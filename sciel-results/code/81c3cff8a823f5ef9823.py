import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    time_limit = 140  # seconds, safety margin before 150s kill
    
    def is_valid(U, V, W):
        """Check if U, V, W is a valid decomposition"""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if val != 1:
                        return False
        return True
    
    def build_schoolbook():
        """Build the schoolbook rank-36 decomposition"""
        rank = n * m * p  # 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Search from low rank upward using structured random search
    # Try ranks from 23 to 35, for each rank try multiple random initializations
    
    # First, try a more systematic approach with better heuristics
    for R in range(23, 36):
        if time.time() - start_time > time_limit - 5:
            break
        
        # For this rank, try multiple random initializations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Generate sparse random U, V with small integer entries
            # Use different sparsity patterns for diversity
            density = 0.2 + 0.3 * (attempt % 5) / 5.0  # Vary density
            
            U = np.zeros((n * m, R), dtype=int)
            V = np.zeros((m * p, R), dtype=int)
            
            # Fill U with random ±1 entries
            for _ in range(int(n * m * R * density)):
                i = np.random.randint(0, n * m)
                r = np.random.randint(0, R)
                U[i, r] = np.random.choice([-1, 1])
            
            # Fill V with random ±1 entries
            for _ in range(int(m * p * R * density)):
                j = np.random.randint(0, m * p)
                r = np.random.randint(0, R)
                V[j, r] = np.random.choice([-1, 1])
            
            # Build linear system for W
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            num_vars = p * n * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_base = k * n + i
                        for r in range(R):
                            var_idx = w_base * R + r
                            A_mat[eq_idx, var_idx] = U[u_idx, r] * V[v_idx, r]
                        eq_idx += 1
            
            # Solve for W using least squares
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if solution is close to integers
                W_int = np.round(W_flat)
                
                # Verify strictly
                if np.allclose(A_mat @ W_int, b_vec, atol=1e-10):
                    W_sol = W_int.reshape((p * n, R))
                    
                    # Final verification with exact integer arithmetic
                    if is_valid(U, V, W_sol):
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_sol.tolist()
                        }
            except:
                pass
    
    # If no better found, return schoolbook
    return build_schoolbook()

# EVOLVE-BLOCK-START
# (No changes to interface)
# EVOLVE-BLOCK-END
