import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10s margin before 120s kill

    def _to_list(arr):
        return arr.tolist()

    def _verify(U, V, W, T):
        """Verify if U, V, W exactly decompose T."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T: (n, m, p)
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Note: The task description says W[k*n+i, r].
        # Let's check the indices carefully.
        # Matrix mult: C[i, k] = sum_j A[i, j] B[j, k]
        # Tensor T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
        # So T[i, j, k] = 1.
        # Decomposition: T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[i, k, r] ?
        # The interface specifies shapes:
        # U: (n*m, rank) -> index i*m + j
        # V: (m*p, rank) -> index j*p + k
        # W: (p*n, rank) -> index k*n + i
        # Let's verify the einsum.
        # U is flattened (i, j). V is flattened (j, k). W is flattened (k, i).
        # We want to reconstruct T[i, j, k].
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's use numpy to verify efficiently.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # sum over r
        recon = np.einsum('ijr,jkr,pir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(recon, T, atol=1e-9)

    def _round_half(arr):
        return np.round(arr * 2) / 2

    def _check_and_return(U, V, W, T):
        """Round to half-integers and verify."""
        U_r = _round_half(U)
        V_r = _round_half(V)
        W_r = _round_half(W)
        if _verify(U_r, V_r, W_r, T):
            return {"U": _to_list(U_r), "V": _to_list(V_r), "W": _to_list(W_r)}
        return None

    # Target: 3x3 * 3x8 = 3x8
    # Tensor T is ones (3, 3, 8)
    T = np.ones((n, m, p))

    # Strategy:
    # 1. Try to find a low-rank decomposition for 3x3x4 (half the width).
    #    If we find rank R for 3x3x4, we can construct 3x3x8 with rank 2R.
    #    Schoolbook 3x3x4 is 36. Schoolbook 3x3x8 is 72.
    #    If we find R=19 for 3x3x4, total rank is 38.
    #    If we find R=18, total rank is 36.
    #    Current seed program failed to find anything better than schoolbook (score 1.0 = 72? No, seed score 1.0 means it returned the seed value. The seed value is likely the schoolbook rank or a known baseline. Wait, the prompt says "score is the rank". "1.0 equals the seed program". The seed program returns schoolbook rank 72? Let's check the seed program's fallback.
    #    Seed fallback: rank = n*m*p = 3*3*8 = 72.
    #    So seed score corresponds to rank 72? Or is the score normalized?
    #    "score is the rank". "1.0 equals the seed program".
    #    If the seed program returns rank 72, and the score is 1.0, then the score is likely `rank / seed_rank` or just the rank itself if the seed rank is 1? No, "lower is better".
    #    "The evaluator measures the raw value (lower is better). The search score is seed/value".
    #    If value=72 and score=1.0, then seed=72.
    #    So if we find rank 38, score = 72/38 = 1.89.
    #    If we find rank 21 (if possible for 3x3x8 directly? Unlikely), score = 72/21 = 3.42.
    
    # Let's focus on finding a good decomposition for 3x3x4.
    # Known results:
    # Rank(2x2x2) = 7 (Strassen)
    # Rank(3x3x3) is between 19 and 23. Best known is 19? 
    # Actually, for 3x3x3, the best known upper bound is 19 (Landsberg?). No, it's 19? 
    # Let's check: Strassen 7 for 2. 3x3x3 is at least 19.
    # For 3x3x4, the rank is at least Rank(3x3x3) = 19?
    # Actually, 3x3x4 can be seen as two 3x3x2 blocks? No.
    # We can split 3x3x8 into two 3x3x4.
    # Let's try to find a rank-19 or rank-20 decomposition for 3x3x4.
    # If we find rank 19 for 3x3x4, we get rank 38 for 3x3x8.
    # If we find rank 18 for 3x3x4, we get rank 36 for 3x3x8.
    
    T_334 = np.ones((3, 3, 4))
    
    best_decomp_334 = None
    best_rank_334 = 36 # Schoolbook
    
    # We will use a randomized local search (ALS) with multiple restarts.
    # To improve success rate, we can try to initialize with known patterns or random.
    # Since 3x3x3 rank 19 is a known result, 3x3x4 might be 19 or 20.
    
    def _als_decomp(T, rank, max_iter=2000, restarts=50, seed=42):
        """
        Attempt to find a rank decomposition using ALS.
        T is shape (n, m, p).
        Returns (U, V, W) if found, else None.
        """
        n, m, p = T.shape
        rng = np.random.RandomState(seed)
        
        for r in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize
            U = rng.randn(n, rank) * 0.5
            V = rng.randn(m, rank) * 0.5
            W = rng.randn(p, rank) * 0.5
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U
                # U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r] / sum_{j,k} V[j,r]^2 W[k,r]^2
                # Efficient computation:
                # Temp_U[i, r] = sum_j sum_k T[i,j,k] V[j,r] W[k,r]
                # This is a matmul-like operation.
                # T is (n, m, p). V is (m, rank). W is (p, rank).
                # For each i, Temp_U[i, :] = sum_j (T[i, j, :] * V[j, :]) dot W ?
                # No.
                # Temp_U[i, r] = sum_j T[i, j, :] dot (V[j, r] * W[:, r]) ?
                # Let's use einsum.
                # 'ijk, jr, kr -> ir'
                Temp_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
                Denom_U = np.einsum('jr, kr, jr, kr -> r', V, W, V, W)
                U_new = Temp_U / (Denom_U[None, :] + 1e-12)
                
                # Update V
                Temp_V = np.einsum('ijk, ir, kr -> jr', T, U_new, W)
                Denom_V = np.einsum('ir, kr, ir, kr -> r', U_new, W, U_new, W)
                V_new = Temp_V / (Denom_V[None, :] + 1e-12)
                
                # Update W
                Temp_W = np.einsum('ijk, ir, jr -> kr', T, U_new, V_new)
                Denom_W = np.einsum('ir, jr, ir, jr -> r', U_new, V_new, U_new, V_new)
                W_new = Temp_W / (Denom_W[None, :] + 1e-12)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                Recon = np.einsum('ir, jr, kr -> ijk', U, V, W)
                Error = np.linalg.norm(Recon - T)
                if Error < 1e-8:
                    res = _check_and_return(U, V, W, T)
                    if res:
                        return res
                    # If rounding fails, it might be a spurious solution or we need to refine.
                    # For now, break this restart.
                    break
            
            # End of iterations for this restart
            res = _check_and_return(U, V, W, T)
            if res:
                return res
                
        return None

    # Try ranks for 3x3x4
    # Lower bound for 3x3x4 is at least 19 (since 3x3x3 is 19).
    # Let's try 19, 20, 21...
    
    for rank in range(19, 37):
        if time.time() > deadline:
            break
        res = _als_decomp(T_334, rank, max_iter=500, restarts=20, seed=rank)
        if res:
            best_decomp_334 = res
            best_rank_334 = rank
            break # Found the minimal rank we can get with this method
    
    if best_decomp_334:
        U1 = np.array(best_decomp_334['U'])
        V1 = np.array(best_decomp_334['V'])
        W1 = np.array(best_decomp_334['W'])
        
        # Construct 3x3x8
        # A (3x3) * B (3x8) = C (3x8)
        # B = [B1, B2] where B1, B2 are 3x4
        # C = [C1, C2] where C1, C2 are 3x4
        # C1 = A * B1, C2 = A * B2
        # We can reuse the same U, V for both halves, but W is split.
        # U is 3x3xR (flattened 9xR), V is 3x4xR (flattened 12xR), W is 4x3xR (flattened 12xR)
        # For 3x3x8:
        # U_full: 9x2R
        # V_full: 12x2R
        # W_full: 24x2R (since p=8, n=3 -> 8*3=24)
        
        R = best_rank_334
        total_rank = 2 * R
        U_full = np.zeros((9, total_rank))
        V_full = np.zeros((12, total_rank))
        W_full = np.zeros((24, total_rank))
        
        # First half (k=0..3)
        U_full[:, 0:R] = U1
        V_full[:, 0:R] = V1
        # W1 is shape (12, R) corresponding to (k, i) for k=0..3, i=0..2
        # In W_full, the first 12 rows correspond to k=0..3.
        W_full[0:12, 0:R] = W1
        
        # Second half (k=4..7)
        U_full[:, R:2*R] = U1
        V_full[:, R:2*R] = V1
        # W1 is shape (12, R) corresponding to (k', i) for k'=0..3, i=0..2
        # In W_full, the next 12 rows correspond to k=4..7.
        # So we just copy W1 again.
        W_full[12:24, R:2*R] = W1
        
        if _verify(U_full, V_full, W_full, T):
            return {"U": _to_list(U_full), "V": _to_list(V_full), "W": _to_list(W_full)}
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
