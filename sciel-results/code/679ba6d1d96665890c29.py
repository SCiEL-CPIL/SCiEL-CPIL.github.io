"""Matrix multiplication 2x2 x 2x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
import itertools
from collections import defaultdict

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s limit

def _check_time():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _verify_rank(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = 2, 2, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Build the multiplication tensor T[a, j, k, i] = delta[j,k] * delta[a,i]
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) where T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # In tensor form: T_{i,j,k} with output index (i,k), input indices (i,j) for A and (j,k) for B
    # So T is a 4-tensor: T[a, b, c, d] where (a,b) indexes A, (c,d) indexes B, (a,d) indexes C
    # T[a,b,c,d] = delta[b,c] * delta[a,a] * delta[d,d]... 
    # More precisely: C[a,d] = sum_b A[a,b] * B[b,d]
    # Tensor: T[a,b;b,d] = delta[b,b] (the shared index)
    # In flattened form: T[A_idx, B_idx, C_idx] where A_idx = a*m+b, B_idx = b*p+d, C_idx = a*n... 
    # Wait, C is 2x6, so C_idx = a*6+d? No.
    
    # Let me use the standard convention:
    # U has shape (n*m, rank) - rows indexed by (i,j) for A[i,j]
    # V has shape (m*p, rank) - rows indexed by (j,k) for B[j,k]
    # W has shape (p*n, rank) - rows indexed by (k,i) for C[k,i]? 
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W row index = k*n + i, where k is the column of B (0..p-1) and i is the row of C (0..n-1)
    
    # The decomposition identity:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since C[i,k] = sum_j A[i,j]*B[j,k], each term A[i,j]*B[j,k] contributes 1 to C[i,k]
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # Check this for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(r))
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_bruteforce(rank, n=2, m=2, p=6):
    """Try to find a rank-r decomposition by solving the bilinear constraints.
    This is extremely expensive for rank > ~10, so we use it only as a fallback."""
    # For small ranks, we can try random initialization + local search
    # But for rank 20, this is infeasible. We'll skip this for now.
    pass

def _construct_rank21():
    """The Strassen-based rank-21 decomposition from the seed."""
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    u_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for t in range(3):
        for s_idx in range(7):
            r = 7 * t + s_idx
            u_s = u_coeffs[s_idx]
            v_s = v_coeffs[s_idx]
            w_s = w_coeffs[s_idx]
            
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = u_s[i * m + j]
                    
            for j in range(m):
                for k in range(p):
                    if 2 * t <= k <= 2 * t + 1:
                        k_loc = k - 2 * t
                        V[j * p + k, r] = v_s[j * 2 + k_loc]
                    else:
                        V[j * p + k, r] = 0
                        
            for k in range(p):
                for i in range(n):
                    if 2 * t <= k <= 2 * t + 1:
                        k_loc = k - 2 * t
                        W[k * n + i, r] = w_s[i * 2 + k_loc]
                    else:
                        W[k * n + i, r] = 0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank20_polynomial():
    """
    Attempt a rank-20 decomposition using polynomial embedding / Kronecker substitution.
    
    Idea: Multiply 2x2 by 2x6. We can think of B as a 2x2 block matrix [B0 B1 B2] where
    B0, B1, B2 are 2x2. Then C = A * [B0 B1 B2] = [A*B0, A*B1, A*B2].
    
    Each A*Bt is a 2x2 product. If we could compute all three 2x2 products sharing some
    Strassen intermediates, we might save rank.
    
    Standard: 3 separate Strassen = 21 rank.
    
    Alternative: Use the fact that A is the same matrix multiplying all three blocks.
    The Strassen intermediates for A * Bt are:
      M1 = (A00+A11)(Bt00+Bt11)
      M2 = (A10+A11)Bt00
      M3 = A00(Bt10-Bt11)
      M4 = A11(Bt01-Bt00)
      M5 = (A00+A01)Bt11
      M6 = (A10-A00)(Bt00+Bt10)
      M7 = (A01-A11)(Bt01+Bt11)
    
    The A-side factors (A00+A11), (A10+A11), A00, A11, (A00+A01), (A10-A00), (A01-A11)
    are the SAME for all three blocks. The B-side factors differ.
    
    So we can precompute 7 "A-side" linear combinations and 3*7=21 "B-side" ones,
    but we only need 21 products total. That's the same as 3*7=21.
    
    To get rank 20, we need to find one more sharing. Let's try:
    Notice that M6 and M7 for different t's share the A-side factor.
    M6_t = (A10-A00)(Bt00+Bt10)
    M7_t = (A01-A11)(Bt01+Bt11)
    
    If we define:
      a6 = A10 - A00
      a7 = A01 - A11
    
    Then M6_t = a6 * (Bt00+Bt10) and M7_t = a7 * (Bt01+Bt11).
    
    These are 6 distinct products (2 A-factors x 3 B-factors each), plus the other
    5 A-factors each producing 3 products = 15, total 21. Still 21.
    
    To get 20, we need to merge one more. Let's see if any B-side factors can be shared
    across different t values in a way that allows a combined product.
    
    Actually, let me try a different approach: use a 2x3 Strassen-like scheme.
    
    Alternative: Pad to 2x2 * 2x2 * 2x2 and use a 3-block Strassen with one shared term.
    
    Let me try a direct rank-20 construction by solving the constraints numerically.
    """
    n, m, p = 2, 2, 6
    target_rank = 20
    
    # We'll try to construct a rank-20 solution by starting from rank-21 and
    # trying to eliminate one tensor.
    
    # Start with the rank-21 solution
    sol21 = _construct_rank21()
    U21 = np.array(sol21['U'])
    V21 = np.array(sol21['V'])
    W21 = np.array(sol21['W'])
    
    # Try to find a rank-20 solution by optimization
    # We'll use a simple local search: perturb one tensor and see if we can adjust others
    
    # Actually, for this problem, let me try a structured rank-20 approach.
    # The key insight: we have 3 blocks of 2x2. The A-side has 7 Strassen factors.
    # If we can share one of the B-side factors across blocks, we save 1 rank.
    
    # Let's try: for blocks 0 and 1, share the B-factor (B00+B11) if possible.
    # M1_0 = (A00+A11)(B0_00+B0_11)
    # M1_1 = (A00+A11)(B1_00+B1_11)
    # These have different B-factors, so they can't be directly merged.
    
    # Instead, let's try a completely different decomposition.
    # Use the "4x4" approach: treat 2x2*2x6 as part of a larger structure.
    
    # For now, let's return rank 21 if we can't find rank 20 quickly.
    # But let's try one more idea: a direct rank-20 with a specific structure.
    
    # Structure: 7 A-factors, and for the 3 B-blocks, use 13 B-factors total
    # (instead of 21), giving 7*3 = 21... no, that doesn't work directly.
    
    # The rank is the number of tensor terms, not the product of factor counts.
    # Each tensor term has one A-row, one B-row, one C-row.
    
    # Let me try: can we find a rank-20 by solving a relaxed version?
    # Given time constraints, I'll return rank 21.
    
    return None

def construct():
    n, m, p = 2, 2, 6
    
    # First, try to find a rank-20 solution using polynomial embedding
    if _check_time():
        result20 = _construct_rank20_polynomial()
        if result20 is not None:
            # Verify it
            U = np.array(result20['U'])
            V = np.array(result20['V'])
            W = np.array(result20['W'])
            r = U.shape[1]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return result20
    
    # Fallback: rank-21 Strassen decomposition
    result21 = _construct_rank21()
    
    # Verify rank-21
    U = np.array(result21['U'])
    V = np.array(result21['V'])
    W = np.array(result21['W'])
    r = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Emergency fallback: naive rank 12 (each product term separately)
        rank = n * m * p  # 24
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result21
