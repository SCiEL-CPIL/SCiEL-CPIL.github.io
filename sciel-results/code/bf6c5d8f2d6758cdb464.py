import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    def solve_decomp_225(rank, seed=0):
        """
        Solve for 2x2x5 decomposition with given rank.
        Returns U, V, W or None.
        """
        np.random.seed(seed)
        total_U = n * m
        total_V = m * 5
        total_W = 5 * n
        
        def residuals(params):
            U = params[:rank*total_U].reshape(rank, total_U).T
            V = params[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
            W = params[rank*(total_U+total_V):].reshape(rank, total_W).T
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(5):
                        val = np.sum(U[i*m+j] * V[j*5+k] * W[k*n+i])
                        res.append(val - 1.0)
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        # Try multiple random initializations
        for trial in range(100):
            # Use small random values
            params0 = np.random.randn(rank * (total_U + total_V + total_W)) * 0.5
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=2000, xtol=1e-15, ftol=1e-15)
                cost = result.cost
                if cost < 1e-10:
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(5):
                                val = np.sum(U_r[i*m+j] * V_r[j*5+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                if cost < best_cost:
                    best_cost = cost
            except:
                pass
        
        return None

    # Try to find rank 14 for 2x2x5
    U5, V5, W5 = None, None, None
    start_time = time.time()
    
    # Try rank 14
    if time.time() - start_time < 60:
        U5, V5, W5 = solve_decomp_225(14, seed=42)
    
    # If rank 14 fails, try rank 15, 16, etc.
    if U5 is None and time.time() - start_time < 90:
        for r in range(15, 21):
            U5, V5, W5 = solve_decomp_225(r, seed=42+r)
            if U5 is not None:
                break
    
    if U5 is None:
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
    
    # Construct full decomposition
    # We have a decomposition for 2x2x5.
    # We need 2x2x15.
    # Split 15 into 5 + 5 + 5.
    # Use 3 copies of the 2x2x5 decomposition.
    
    rank_block = U5.shape[1]
    total_rank = rank_block * 3
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for block_idx in range(3):
        offset = block_idx * rank_block
        # U is the same for all blocks (since A is 2x2)
        U_full[:, offset:offset+rank_block] = U5
        
        # V and W are shifted for each block of 5 columns
        # V maps (j, k) where k is 0..14
        # Block 0: k=0..4
        # Block 1: k=5..9
        # Block 2: k=10..14
        
        # V5 has shape (m*5, rank_block) i.e., (10, rank_block)
        # Indices in V5: row = j*5 + k_local
        # In V_full: row = j*15 + k_global
        
        for j in range(m):
            for k_local in range(5):
                k_global = block_idx * 5 + k_local
                V_full[j * p + k_global, offset:offset+rank_block] = V5[j * 5 + k_local]
        
        # W5 has shape (5*n, rank_block) i.e., (10, rank_block)
        # Indices in W5: row = k_local*n + i
        # In W_full: row = k_global*n + i
        
        for k_local in range(5):
            k_global = block_idx * 5 + k_local
            for i in range(n):
                W_full[k_global * n + i, offset:offset+rank_block] = W5[k_local * n + i]
                
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
