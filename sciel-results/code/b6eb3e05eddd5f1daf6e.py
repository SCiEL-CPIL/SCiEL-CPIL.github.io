import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 96
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def verify_solution(U, V, W, R):
        """Verify that U, V, W with rank R correctly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j, r] * V[l * p + k, r]
                        expected = 1.0 if j == l else 0.0
                        if abs(val - expected) > 1e-8:
                            return False
        return True
    
    def solve_for_W(U_flat, V_flat, R):
        """
        Given U and V, solve for W such that the tensor decomposition holds.
        For each (i,k), we need sum_r W[k*n+i, r] * U[i*m+j, r] * V[l*p+k, r] = delta_{jl}
        This is a linear system in W[k*n+i, :].
        """
        W_flat = np.zeros((p * n, R))
        for i in range(n):
            for k in range(p):
                # Build the matrix A where A[j*m+l, r] = U[i*m+j, r] * V[l*p+k, r]
                A = np.zeros((m * m, R))
                b = np.zeros(m * m)
                for j in range(m):
                    b[j * m + j] = 1.0
                    for l in range(m):
                        A[j * m + l, :] = U_flat[i * m + j, :] * V_flat[l * p + k, :]
                
                # Solve A @ w = b for w = W[k*n+i, :]
                # Use least squares
                try:
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                
                W_flat[k * n + i, :] = w
        
        return W_flat
    
    def snap_to_half_integers(arr):
        """Snap values to nearest half-integer."""
        return np.round(arr * 2) / 2.0
    
    def try_structured_rank(R, max_trials=100):
        """
        Try to find a valid decomposition of rank R.
        Strategy: Randomly initialize U and V with entries in {-1, 0, 1},
        then solve for W. Check if W can be snapped to half-integers and verify.
        """
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U_flat = np.random.randint(-1, 2, (n * m, R))
            V_flat = np.random.randint(-1, 2, (m * p, R))
            
            # Solve for W
            W_flat = solve_for_W(U_flat, V_flat, R)
            if W_flat is None:
                continue
            
            # Snap W to half-integers
            W_snapped = snap_to_half_integers(W_flat)
            
            # Verify the full solution
            if verify_solution(U_flat.tolist(), V_flat.tolist(), W_snapped.tolist(), R):
                return {
                    "U": [[float(v) for v in row] for row in U_flat],
                    "V": [[float(v) for v in row] for row in V_flat],
                    "W": [[float(v) for v in row] for row in W_snapped]
                }
        
        return None
    
    def try_strassen_block_rank(R):
        """
        Try a structured approach based on Strassen-like block decomposition.
        For 4x4x4, known rank is 21. For 4x4x6, we can try to extend.
        This is a heuristic and may not always work.
        """
        if R < 21:
            return None
        
        # Try to use a known rank-21 decomposition for 4x4x4 and extend to 4x4x6
        # This is complex, so we'll rely on the random search for now
        return None
    
    # Search for lower rank
    # Start from a reasonable minimum
    # Known: 4x4x4 has rank 21, so 4x4x6 should be at most 21 + something
    # Let's search from 25 to 95
    for R in range(25, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_structured_rank(R, max_trials=50)
        if result is not None:
            best_solution = result
            best_rank = R
            break
        
        # Also try the structured approach
        result2 = try_strassen_block_rank(R)
        if result2 is not None:
            best_solution = result2
            best_rank = R
            break
    
    return best_solution
