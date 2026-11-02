import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.ravel()
    T_2d = T.reshape(U_dim * V_dim, W_dim)
    T_3d = T.reshape(U_dim, V_dim * W_dim)
    T_3d_v = T.reshape(V_dim, U_dim * W_dim)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True
    
    def round_and_verify(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None
    
    # --- Strassen-based rank-42 construction ---
    # For 2x4x6, use Strassen on 2x2x2 blocks.
    # The idea: decompose the computation into Strassen subproblems.
    # 
    # A (2x4) = [A11 A12; A21 A22] where each Ai is 1x2
    # B (4x6) = [B11 B12 B13; B21 B22 B23] where each Bj is 2x2
    # 
    # C = A * B = [C11 C12 C13; C21 C22 C23]
    # C1j = A11*B1j + A12*B2j
    # C2j = A21*B1j + A22*B2j
    #
    # Each product of 1x2 by 2x2 is a 1x2 result.
    # We can use Strassen-like combinations.
    #
    # Actually, let's use a simpler valid construction:
    # rank(2,4,6) <= rank(2,4,4) + rank(2,4,2)
    # For 2x4x4: use 2 Strassen 2x2x2 blocks => rank <= 14
    # For 2x4x2: use 2 Strassen 2x2x2 blocks => rank <= 14  
    # Total: 28? But we need to be careful about the structure.
    #
    # Let me just use the schoolbook as initial and refine with ALS.
    # The cell finding says 42 is achievable, so let's start there.
    
    # Start with schoolbook rank 48
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school), dtype=np.float64)
    V_school = np.zeros((V_dim, rank_school), dtype=np.float64)
    W_school = np.zeros((W_dim, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_school
    
    # Try to reduce rank using ALS from schoolbook initialization
    # Start by trying to project to lower ranks
    
    for target_rank in range(24, rank_school):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Use schoolbook columns as initial basis, try to find a lower-rank representation
        # Initialize with first target_rank schoolbook columns
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            seed = attempt * 1000 + target_rank
            np.random.seed(seed)
            
            # Initialize with a mix of schoolbook and random
            if attempt < 5:
                # Use subset of schoolbook columns
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
            else:
                # Random initialization
                U = np.random.randn(U_dim, target_rank) * 0.5
                V = np.random.randn(V_dim, target_rank) * 0.5
                W = np.random.randn(W_dim, target_rank) * 0.5
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline - 20:
                    break
                
                # Fix U and V, solve for W
                M = np.zeros((U_dim * V_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(U[:, r], V[:, r])
                    M[:, r] = outer.ravel()
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M.T, T_2d.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Fix V and W, solve for U
                N = np.zeros((V_dim * W_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(V[:, r], W[:, r])
                    N[:, r] = outer.ravel()
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(N.T, T_3d.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Fix U and W, solve for V
                P = np.zeros((U_dim * W_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(U[:, r], W[:, r])
                    P[:, r] = outer.ravel()
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(P.T, T_3d_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Check error periodically
                if it % 30 == 0 or it == 299:
                    # Compute residual efficiently
                    error = 0.0
                    for a in range(U_dim):
                        for b in range(V_dim):
                            for c in range(W_dim):
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                error += (val - T[a, b, c]) ** 2
                    
                    if error < 1e-6:
                        result = round_and_verify(U, V, W)
                        if result is not None:
                            U_r, V_r, W_r = result
                            found = True
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            break
                
                if found:
                    break
            
            if found:
                break
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_school
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
