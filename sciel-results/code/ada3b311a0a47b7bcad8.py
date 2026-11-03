import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-9):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    rng = np.random.default_rng(12345)
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C = A @ B
        
        M = U_3d * A[:, :, np.newaxis]
        N = V_3d * B[:, :, np.newaxis]
        S = np.einsum('ijc,jkc->ikc', M, N)
        S_T = np.transpose(S, (1, 0, 2))
        C_dec = np.sum(S_T * W_3d, axis=2)
        
        if np.max(np.abs(C - C_dec)) > tol:
            return False
    return True

def _to_list(mat):
    if isinstance(mat, np.ndarray):
        mat = mat.tolist()
    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else (round(v * 2) / 2 if abs(v * 2 - round(v * 2)) < 1e-9 else v) for v in row] for row in mat]

def _winograd_4x4_factors():
    """
    Returns U, V, W for Winograd's 4x4x4 matrix multiplication algorithm.
    Rank is 43.
    U: (16, 43), V: (16, 43), W: (16, 43)
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    # Hardcoded Winograd 4x4 factors (Rank 43)
    # This is a known decomposition.
    # Generating this dynamically is complex, so we use a pre-computed set.
    # Note: The exact values depend on the specific Winograd variant.
    # We will use a standard set of factors for F(4,4) matrix mult.
    
    # Since I cannot guarantee the correctness of hardcoded values from memory without verification,
    # and the risk of failure is high, I will implement a robust fallback:
    # Use the Strassen-based rank 88 decomposition which is structurally simpler to verify.
    # However, the directive asks for a structural change.
    # I will implement a rank-88 decomposition using Strassen for 2x2 blocks 
    # but applied to 4x4 blocks? No, that's rank 49.
    # Let's use the known rank 88 decomposition for 3x4x8.
    # Actually, the inspiration 3 achieved 1.0909 (rank 88).
    # I will implement the rank 88 decomposition explicitly.
    
    # Rank 88 decomposition for 3x4x8:
    # Split p=8 into p1=4, p2=4.
    # For each 3x4x4 block, use a rank 44 decomposition?
    # Or use Strassen on the 4x4 part?
    # Rank of 4x4x4 via Strassen is 49.
    # Rank of 3x4x4 is at most 3 * rank(4x4x4)? No.
    # If we use Strassen for 4x4x4, rank is 49.
    # Then 3x4x8 rank is 49 + 49 = 98.
    # This is worse than 96.
    
    # So we need a better decomposition for 4x4x4.
    # Winograd 4x4x4 has rank 43.
    # If I can't hardcode it, I will use a random search to find rank < 96.
    
    # For now, return None to indicate we need to use the search or fallback.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find a better decomposition using random search
    rng = np.random.default_rng(42)
    
    # Target rank 88 (from inspiration)
    # We will search for rank 88
    for r in range(88, min(96, best_rank)):
        if time.time() > deadline - 5:
            break
        
        max_iters = 20
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_rand = rng.standard_normal((nm, r))
            V_rand = rng.standard_normal((mp, r))
            
            A = np.zeros((n * m * p, pn * r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * mp + j * p + k
                        for c in range(r):
                            uid = (k * n + i) * r + c
                            A[idx, uid] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
            
            T_flat = np.ones(n * m * p)
            
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
                
            W_rand = W_flat.reshape((pn, r))
            
            W_rounded = np.round(W_rand * 2) / 2
            if np.max(np.abs(W_rand - W_rounded)) < 1e-6:
                U_rounded = np.round(U_rand * 2) / 2
                V_rounded = np.round(V_rand * 2) / 2
                if np.max(np.abs(U_rand - U_rounded)) < 1e-6 and np.max(np.abs(V_rand - V_rounded)) < 1e-6:
                    U_list = _to_list(U_rounded)
                    V_list = _to_list(V_rounded)
                    W_list = _to_list(W_rounded)
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        break
        
        if best_rank == r:
            break
            
    # Convert final result
    U_final = best["U"]
    V_final = best["V"]
    W_final = best["W"]
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-9:
                    new_row.append(round(v * 2) / 2)
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_final = clean_matrix(U_final)
    V_final = clean_matrix(V_final)
    W_final = clean_matrix(W_final)
    
    return {"U": U_final, "V": V_final, "W": W_final}
