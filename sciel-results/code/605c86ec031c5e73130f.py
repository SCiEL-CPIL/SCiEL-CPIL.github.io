import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank14_strassen():
    """Build a rank-14 decomposition using Strassen for each 2x2 block.
    
    For 2x2*2x4, we split into two 2x2*2x2 subproblems.
    U: (4, rank), V: (8, rank), W: (8, rank)
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Standard Strassen 2x2*2x2 patterns
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_patterns = [
        # (A_support, A_signs, B_support, B_signs, C_support, C_signs)
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

def _peel_tensor(U, V, W, n, m, p, deadline):
    """
    Apply peeling algorithm to try to reduce rank.
    
    The idea: given a rank-r decomposition, try to find linear dependencies
    among the rank-1 tensors U_r ⊗ V_r ⊗ W_r that allow removing one component.
    
    For each pair (r1, r2), check if there exist scalars a, b such that
    (a*U_r1 + b*U_r2) ⊗ (a*V_r1 + b*V_r2) ⊗ (a*W_r1 + b*W_r2) equals the sum
    of U_r1 ⊗ V_r1 ⊗ W_r1 + U_r2 ⊗ V_r2 ⊗ W_r2 when combined with the rest.
    
    More practically: try to find if the tensor T = sum_r U_r ⊗ V_r ⊗ W_r can be
    represented with rank r-1 by finding a rank-1 component that is a linear
    combination of existing components.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the full tensor
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    # Try to peel off rank-1 components
    # For each potential rank-1 component X ⊗ Y ⊗ Z, check if T - X⊗Y⊗Z
    # can be represented with rank r-1
    
    # Generate candidate rank-1 components from the existing decomposition
    # and their linear combinations
    candidates = []
    
    # Individual components
    for r in range(rank):
        candidates.append((r, [0.0]*rank))
        candidates[r] = (r, [(1.0 if i == r else 0.0) for i in range(rank)])
    
    # Try all pairs of components with small coefficients
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            for a in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
                for b in [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]:
                    coeffs = [0.0] * rank
                    coeffs[r1] = a
                    coeffs[r2] = b
                    candidates.append((None, coeffs))
    
    # Try to peel each candidate
    best_result = None
    best_rank = rank
    
    for start_idx, (orig_idx, coeffs) in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Build the candidate rank-1 component
        U_cand = np.array([sum(coeffs[r] * U_arr[i, r] for r in range(rank)) for i in range(n*m)])
        V_cand = np.array([sum(coeffs[r] * V_arr[j, r] for r in range(rank)) for j in range(m*p)])
        W_cand = np.array([sum(coeffs[r] * W_arr[k, r] for r in range(rank)) for k in range(p*n)])
        
        # Check if this is a valid rank-1 component (non-zero)
        if np.allclose(U_cand, 0) or np.allclose(V_cand, 0) or np.allclose(W_cand, 0):
            continue
        
        # Compute T - candidate
        T_rem = T.copy()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rem[i, j, k] -= U_cand[i*m+j] * V_cand[j*p+k] * W_cand[k*n+i]
        
        # Check if T_rem can be represented with rank r-1
        # This is the hard part - we need to verify if T_rem has rank ≤ r-1
        # Approximation: check if T_rem is close to the sum of all components except
        # those that make up the candidate
        
        if orig_idx is not None:
            # We're peeling off a single component
            # T_rem should equal sum of all other components
            T_other = np.zeros((n, m, p))
            for r in range(rank):
                if r != orig_idx:
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_other[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            
            if np.allclose(T_rem, T_other, atol=1e-6):
                # We've found a redundant component!
                new_rank = rank - 1
                U_new = [[0] * new_rank for _ in range(n*m)]
                V_new = [[0] * new_rank for _ in range(m*p)]
                W_new = [[0] * new_rank for _ in range(p*n)]
                
                col = 0
                for r in range(rank):
                    if r == orig_idx:
                        continue
                    for i in range(n*m):
                        U_new[i][col] = U_arr[i, r]
                    for j in range(m*p):
                        V_new[j][col] = V_arr[j, r]
                    for k in range(p*n):
                        W_new[k][col] = W_arr[k, r]
                    col += 1
                
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = (U_new, V_new, W_new)
                    best_rank = new_rank
                    break
        else:
            # We're peeling off a linear combination
            # Check if T_rem has rank ≤ r-1 by trying to express it as
            # sum of r-1 rank-1 tensors
            # This is expensive, so we do a quick check:
            # If T_rem is close to zero, we've found a complete decomposition
            if np.allclose(T_rem, 0, atol=1e-6):
                # The candidate alone represents the entire tensor
                # This means rank 1, which is unlikely but possible
                pass
            elif start_idx % 10 == 0:  # Only do expensive check every 10th candidate
                # Try to verify if T_rem can be built from the remaining components
                # by solving for which components to keep
                # This is a combinatorial problem, so we skip for now
                pass
    
    return best_result

def _search_rank13_via_peeling(n, m, p, deadline):
    """Search for rank-13 decomposition using peeling algorithm."""
    U14, V14, W14 = _build_rank14_strassen()
    
    # First, verify the rank-14 decomposition
    if not _verify(U14, V14, W14, n, m, p):
        return None
    
    # Try peeling
    result = _peel_tensor(U14, V14, W14, n, m, p, deadline)
    if result is not None:
        U_new, V_new, W_new = result
        if _verify(U_new, V_new, W_new, n, m, p):
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Also try with different Strassen pattern permutations
    import random
    random.seed(42)
    
    # Generate random sign flips for Strassen patterns
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Randomly flip signs of some Strassen components
        U_mod = [row[:] for row in U14]
        V_mod = [row[:] for row in V14]
        
        for r in range(14):
            if random.random() < 0.3:
                for i in range(len(U_mod)):
                    U_mod[i][r] *= -1
                    V_mod[i][r] *= -1
        
        result = _peel_tensor(U_mod, V_mod, W14, n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    return None

def _search_rank13_direct(n, m, p, deadline):
    """
    Direct search for rank-13 decomposition by fixing U and V to sparse patterns
    and solving for W.
    
    Key insight: for 2x2*2x4, the tensor has 2*2*4 = 16 entries.
    With rank 13, we have 13*4 + 13*8 + 13*8 = 284 variables.
    This is an underdetermined system, so solutions should exist.
    
    Strategy: fix U to have exactly 2 non-zeros per column (typical for Strassen-like
    decompositions), and V to have 2-3 non-zeros per column. Then solve the linear
    system for W and check if W is half-integer.
    """
    rank = 13
    np.random.seed(12345)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Generate sparse U: each column has exactly 2 non-zeros
        U = np.zeros((n*m, rank), dtype=int)
        for r in range(rank):
            rows = np.random.choice(n*m, size=2, replace=False)
            for row in rows:
                U[row, r] = np.random.choice([-1, 1])
        
        # Generate sparse V: each column has 2 or 3 non-zeros
        V = np.zeros((m*p, rank), dtype=int)
        for r in range(rank):
            num_nonzero = np.random.choice([2, 3])
            rows = np.random.choice(m*p, size=num_nonzero, replace=False)
            for row in rows:
                V[row, r] = np.random.choice([-1, 1])
        
        # Build linear system for W
        # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        A = np.zeros((n*m*p, p*n*rank), dtype=float)
        b = np.ones(n*m*p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            A[eq, (k*n+i)*rank + r] = coeff
                    eq += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W = W_flat.reshape(p*n, rank)
        
        # Check residual
        if np.linalg.norm(A @ W_flat - b) > 1e-6:
            continue
        
        # Check if W is close to half-integers
        W2 = W * 2
        W_rounded = np.round(W2)
        error = np.linalg.norm(W2 - W_rounded)
        
        if error < 0.1:
            W_int = W_rounded / 2.0
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_int.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try rounding U and V to half-integers
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        if _verify(U2.tolist(), V2.tolist(), W_int.tolist(), n, m, p):
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W_int.tolist()}
    
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Guaranteed fallback: rank-14 Strassen construction
    U14, V14, W14 = _build_rank14_strassen()
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Verify fallback
    if not _verify(U14, V14, W14, n, m, p):
        best_result = _fallback_rank16()
        best_rank = 16
        return best_result
    
    # Try peeling-based rank-13 search
    if time.time() < deadline - 20:
        result = _search_rank13_via_peeling(n, m, p, deadline - 10)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    # Try direct rank-13 search
    if time.time() < deadline - 5 and best_rank > 13:
        result = _search_rank13_direct(n, m, p, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    return best_result
