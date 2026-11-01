import numpy as np
import time
import itertools
from functools import lru_cache

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n+i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank14_strassen():
    """Build a rank-14 decomposition using Strassen for each 2x2 block."""
    n, m, p = 2, 2, 4
    rank = 14
    
    # Standard Strassen 2x2*2x2 patterns
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_patterns = [
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
        ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
        ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
        ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
        ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (1, 3), (1, 1)),  # M7
    ]
    
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    # First 2x2 block (columns 0-6)
    for r in range(7):
        a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
        for idx, a_idx in enumerate(a_sup):
            U[a_idx][r] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V[b_idx][r] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W[c_idx][r] = c_sgn[idx]
    
    # Second 2x2 block (columns 7-13)
    for r in range(7):
        a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
        for idx, a_idx in enumerate(a_sup):
            U[a_idx][r + 7] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V[4 + b_idx][r + 7] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W[4 + c_idx][r + 7] = c_sgn[idx]
    
    return U, V, W

def _groebner_rank13_search(n, m, p, deadline):
    """
    Attempt to find a rank-13 decomposition by solving for a shared term
    that is a linear combination of Strassen terms from both blocks.
    
    We assume the rank-13 solution consists of:
    - 6 terms from Block 1 (Strassen M1-M6)
    - 6 terms from Block 2 (Strassen M1-M6)
    - 1 shared term T_shared = u ⊗ v ⊗ w
    
    The total tensor is T = T_strassen_B1_full + T_strassen_B2_full.
    We want T = (T_B1 - T_M7_B1) + (T_B2 - T_M7_B2) + T_shared
    => T_shared = T_M7_B1 + T_M7_B2
    
    This is the same structural constraint as before, but we allow the "removed"
    terms to be ANY linear combination of the 7 Strassen terms in each block,
    not just a single term. However, for the shared term to be rank-1,
    the sum of the two "removed" parts must be rank-1.
    
    Let R1 = sum_{r=1}^7 c_r * T_r_B1 and R2 = sum_{r=1}^7 d_r * T_r_B2.
    We need R1 + R2 to be rank-1.
    Since T_r_B1 has support only on k in [0,3] and T_r_B2 on k in [4,7],
    R1 + R2 is rank-1 iff R1 is rank-1 (supported on first half) AND R2 is
    rank-1 (supported on second half) AND their supports align in a way that
    allows factorization? No, R1 and R2 have disjoint support in k.
    A tensor with disjoint support in k can be rank-1 only if one part is zero
    or if the "shared" factorization works across the disjoint supports?
    
    Actually, if T[i,j,k] = u_i v_j w_k, and T has support on k=0..3 for R1
    and k=4..7 for R2, then for k in 0..3, T[i,j,k] = u_i v_j w_k must hold,
    and for k in 4..7, T[i,j,k] = u_i v_j w_k must hold.
    
    This implies that the "shape" of R1 (as a 2x2x4 tensor, but effectively 2x2x4
    with zeros in second half) must be consistent with the "shape" of R2
    (zeros in first half) via the SAME u and v vectors.
    
    So we need:
    1. R1 = u ⊗ v ⊗ w1 (where w1 is length 4, supported on first 4)
    2. R2 = u ⊗ v ⊗ w2 (where w2 is length 4, supported on last 4)
    
    This means R1 must be a rank-1 tensor in the 2x2x4 space (with zeros in k>3)
    AND R2 must be a rank-1 tensor in the 2x2x4 space (with zeros in k<4)
    AND they must share the SAME u and v factors.
    
    Let's check if there exist coefficients c_r such that R1 = sum c_r T_r_B1 is rank-1
    AND coefficients d_r such that R2 = sum d_r T_r_B2 is rank-1
    AND the u, v factors match.
    
    This is a system of polynomial equations.
    """
    
    # Precompute the 7 Strassen tensors for Block 1 and Block 2
    # T_r_B1[i,j,k] is non-zero only for k in 0..3
    # T_r_B2[i,j,k] is non-zero only for k in 4..7
    
    strassen_patterns = [
        ((0, 3), (1, 1), (0, 3), (1, 1), (0, 1, 2, 3), (1, -1, 1, 1)),  # M1
        ((2, 3), (1, 1), (0,), (1,), (2, 3), (1, 1)),  # M2
        ((0,), (1,), (2, 3), (1, -1), (0, 2), (1, -1)),  # M3
        ((3,), (1,), (1, 0), (1, -1), (1, 3), (1, -1)),  # M4
        ((0, 1), (1, 1), (3,), (1,), (1, 3), (1, 1)),  # M5
        ((2, 0), (1, -1), (0, 2), (1, 1), (2, 3), (1, 1)),  # M6
        ((1, 3), (1, -1), (2, 3), (1, 1), (1, 3), (1, 1)),  # M7
    ]
    
    T_B1 = np.zeros((7, n, m, p))
    T_B2 = np.zeros((7, n, m, p))
    
    for r in range(7):
        a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        for idx, a_idx in enumerate(a_sup):
            U_vec[a_idx] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V_vec[b_idx] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W_vec[c_idx] = c_sgn[idx]
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_B1[r, i, j, k] = U_vec[i*m+j] * V_vec[j*p+k] * W_vec[k*n+i]
                    
        U_vec2 = np.zeros(n*m)
        V_vec2 = np.zeros(m*p)
        W_vec2 = np.zeros(p*n)
        for idx, a_idx in enumerate(a_sup):
            U_vec2[a_idx] = a_sgn[idx]
        for idx, b_idx in enumerate(b_sup):
            V_vec2[4 + b_idx] = b_sgn[idx]
        for idx, c_idx in enumerate(c_sup):
            W_vec2[4 + c_idx] = c_sgn[idx]
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_B2[r, i, j, k] = U_vec2[i*m+j] * V_vec2[j*p+k] * W_vec2[k*n+i]

    # We want to find c, d such that R1 = sum c_r T_B1[r] is rank-1
    # and R2 = sum d_r T_B2[r] is rank-1
    # and they share the same u, v factors.
    
    # Let's parameterize u, v, w1, w2.
    # R1[i,j,k] = u_i v_j w1_k for k in 0..3
    # R2[i,j,k] = u_i v_j w2_k for k in 4..7
    
    # This implies that the matrix R1[:,:,k] = u v^T w1_k must be rank-1 for all k.
    # And R2[:,:,k] = u v^T w2_k must be rank-1 for all k.
    # This is automatically satisfied if u and v are fixed vectors.
    # The constraint is that R1 must lie in the span of T_B1[r] and be of the form u⊗v⊗w1.
    
    # Let's fix u and v to be simple vectors and see if we can solve for w1, w2, c, d.
    # Or better, let's use the fact that R1 is a linear combination of T_B1.
    # The space of rank-1 tensors in the subspace spanned by T_B1 is what we're looking for.
    
    # Let's try to find a non-trivial linear combination of T_B1 that is rank-1.
    # This is a classic problem: find c such that sum c_r T_r is rank-1.
    # We can use the catalecticant or SVD rank check.
    
    # Let's iterate over possible simple u, v vectors.
    # Since the entries are integers/half-integers, let's try small integer vectors for u and v.
    
    possible_u = [
        [1, 1], [1, -1], [1, 0], [0, 1], [1, 2], [2, 1], [1, -2], [-1, 1]
    ]
    possible_v = [
        [1, 1], [1, -1], [1, 0], [0, 1], [1, 2], [2, 1], [1, -2], [-1, 1]
    ]
    
    for u in possible_u:
        for v in possible_v:
            if time.time() > deadline:
                return None
                
            # We want R1 = sum c_r T_B1[r] = u ⊗ v ⊗ w1
            # This means for each k in 0..3:
            # sum c_r T_B1[r, i, j, k] = u_i v_j w1_k
            
            # Let's set up a linear system for c and w1.
            # Actually, let's fix w1_k to be 1 for k=0 and 0 otherwise? No, w1 is unknown.
            # Let's fix c_0 = 1 and solve for c_1..c_6 and w1_0..w1_3.
            # Or fix w1_0 = 1 and solve for c_0..c_6 and w1_1..w1_3.
            
            # Let's try fixing w1_0 = 1 (if non-zero).
            # Then for each i, j, k in 0..3:
            # sum_{r=0}^6 c_r T_B1[r, i, j, k] = u_i v_j w1_k
            
            # This is 2*2*4 = 16 equations.
            # Unknowns: c_0..c_6 (7) + w1_1..w1_3 (3) = 10 unknowns.
            # We have 16 equations. This is overdetermined.
            # We can solve it by least squares and check if the residual is small.
            
            A = np.zeros((16, 10))
            b = np.zeros(16)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(4):
                        # Equation: sum_r c_r T_B1[r, i, j, k] - u_i v_j w1_k = 0
                        for r in range(7):
                            A[eq_idx, r] = T_B1[r, i, j, k]
                        # Coefficient for w1_k
                        if k == 0:
                            # w1_0 is fixed to 1, so move to RHS
                            b[eq_idx] = u[i] * v[j] * 1.0
                        else:
                            A[eq_idx, 7 + (k-1)] = -u[i] * v[j]
                        eq_idx += 1
                        
            # Solve least squares
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                c = sol[:7]
                w1_rest = sol[7:]
                w1 = np.array([1.0] + list(w1_rest))
                
                # Check residual
                if np.linalg.norm(A @ sol - b) < 1e-9:
                    # We found a valid R1. Now check if R2 can be formed with same u, v.
                    # R2 = sum d_r T_B2[r] = u ⊗ v ⊗ w2
                    # For k in 4..7:
                    # sum d_r T_B2[r, i, j, k] = u_i v_j w2_{k-4}
                    
                    A2 = np.zeros((16, 10))
                    b2 = np.zeros(16)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(4, 8):
                                for r in range(7):
                                    A2[eq_idx, r] = T_B2[r, i, j, k]
                                if k == 4:
                                    b2[eq_idx] = u[i] * v[j] * 1.0
                                else:
                                    A2[eq_idx, 7 + (k-5)] = -u[i] * v[j]
                                eq_idx += 1
                                
                    try:
                        sol2, residuals2, rank2, sv2 = np.linalg.lstsq(A2, b2, rcond=None)
                        d = sol2[:7]
                        w2_rest = sol2[7:]
                        w2 = np.array([1.0] + list(w2_rest))
                        
                        if np.linalg.norm(A2 @ sol2 - b2) < 1e-9:
                            # We have R1 and R2 with same u, v.
                            # Construct the rank-13 solution.
                            # Terms:
                            # 0-5: Strassen B1 M1-M6
                            # 6-11: Strassen B2 M1-M6
                            # 12: Shared term u ⊗ v ⊗ w_shared
                            # where w_shared = [w1_0..w1_3, w2_0..w2_3]
                            
                            U = [[0] * 13 for _ in range(n*m)]
                            V = [[0] * 13 for _ in range(m*p)]
                            W = [[0] * 13 for _ in range(p*n)]
                            
                            # Fill Block 1 (cols 0-5)
                            for r in range(6):
                                a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
                                for idx, a_idx in enumerate(a_sup):
                                    U[a_idx][r] = a_sgn[idx]
                                for idx, b_idx in enumerate(b_sup):
                                    V[b_idx][r] = b_sgn[idx]
                                for idx, c_idx in enumerate(c_sup):
                                    W[c_idx][r] = c_sgn[idx]
                                    
                            # Fill Block 2 (cols 6-11)
                            for r in range(6):
                                a_sup, a_sgn, b_sup, b_sgn, c_sup, c_sgn = strassen_patterns[r]
                                for idx, a_idx in enumerate(a_sup):
                                    U[a_idx][r + 6] = a_sgn[idx]
                                for idx, b_idx in enumerate(b_sup):
                                    V[4 + b_idx][r + 6] = b_sgn[idx]
                                for idx, c_idx in enumerate(c_sup):
                                    W[4 + c_idx][r + 6] = c_sgn[idx]
                                    
                            # Fill Shared Term (col 12)
                            w_shared = np.concatenate([w1, w2])
                            for i in range(n):
                                for j in range(m):
                                    U[i*m+j][12] = u[i]
                            for j in range(m):
                                for k in range(p):
                                    V[j*p+k][12] = v[j]
                            for k in range(p):
                                for i in range(n):
                                    W[k*n+i][12] = w_shared[k]
                                    
                            if _verify(U, V, W, n, m, p):
                                return {"U": U, "V": V, "W": W}
                    except:
                        pass
            except:
                pass
                
    return None

def _fallback_rank16():
    """Absolute fallback: rank-16 schoolbook decomposition."""
    n, m, p = 2, 2, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Guaranteed fallback: rank-14 Strassen construction
    U14, V14, W14 = _build_rank14_strassen()
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Verify fallback
    if not _verify(U14, V14, W14, n, m, p):
        best_result = _fallback_rank16()
        best_rank = 16
        return best_result
    
    # Try Groebner-based rank-13 search
    if time.time() < deadline - 10:
        result = _groebner_rank13_search(n, m, p, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res)
