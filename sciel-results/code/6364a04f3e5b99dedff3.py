"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Precompute the list of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # Convert to index form: u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    u_indices = [i * m + j for i, j, k in triples]
    v_indices = [j * p + k for i, j, k in triples]
    w_indices = [k * n + i for i, j, k in triples]
    
    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the tensor."""
        for idx in range(num_triples):
            ui, vi, wi = u_indices[idx], v_indices[idx], w_indices[idx]
            s = 0.0
            for c in range(rank):
                s += U[ui][c] * V[vi][c] * W[wi][c]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def try_rank(rank, max_iters=5000, seed=42):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with small random values
        U[:] = np_rng.uniform(-1, 1, (n * m, rank))
        V[:] = np_rng.uniform(-1, 1, (m * p, rank))
        W[:] = np_rng.uniform(-1, 1, (p * n, rank))
        
        # Convert to lists for verification
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Check initial
        if verify_decomp(U_list, V_list, W_list, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Use a gradient-free local search approach
        # We'll use a simple hill-climbing with random perturbations
        
        best_error = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current error
            error = 0.0
            for idx in range(num_triples):
                ui, vi, wi = u_indices[idx], v_indices[idx], w_indices[idx]
                s = np.dot(U[ui], V[vi] * W[wi])
                error += (s - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if error < 1e-12:
                    U_list = best_U.tolist()
                    V_list = best_V.tolist()
                    W_list = best_W.tolist()
                    if verify_decomp(U_list, V_list, W_list, rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            # Random perturbation
            alpha = 0.1 * (1.0 - it / max_iters) + 0.01
            mask_u = (np_rng.rand(n * m, rank) < 0.1)
            mask_v = (np_rng.rand(m * p, rank) < 0.1)
            mask_w = (np_rng.rand(p * n, rank) < 0.1)
            
            U[mask_u] += alpha * np_rng.randn(n * m, rank)[mask_u]
            V[mask_v] += alpha * np_rng.randn(m * p, rank)[mask_v]
            W[mask_w] += alpha * np_rng.randn(p * n, rank)[mask_w]
        
        # Final check
        if best_error < 1e-12:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            if verify_decomp(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from schoolbook down
    # For (2,3,7), the minimum rank is known to be at most 42 (schoolbook)
    # Let's try a few lower ranks
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks in decreasing order
    for rank in range(schoolbook_rank - 1, max(10, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
