"""Program for matrix multiplication tensor decomposition task."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    Strategy: Exploit the direct-sum structure. The 3x3 * 3x6 multiplication can be viewed as
    two independent 3x3 * 3x3 multiplications (one for output columns 0,1,2 and one for 3,4,5).
    We attempt to find a low-rank decomposition for the 3x3*3x3 case (target rank 23, best known)
    and combine two copies to get rank 46 for the full problem.
    If that fails, we fall back to schoolbook rank 54.
    """
    n, m, p = DIMENSIONS  # 3, 3, 6
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    def is_valid_decomposition_full(U, V, W, rank):
        """Check if U,V,W of given rank exactly decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def find_3x3x3_decomposition(rank, seed=42, max_time=60):
        """
        Find a rank-rank decomposition for 3x3x3 matrix multiplication.
        Returns (U, V, W) as lists or None.
        U: (9, rank), V: (9, rank), W: (9, rank)
        Constraint: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] == 1
        """
        start_t = time.time()
        rng = np.random.RandomState(seed)
        
        # The tensor T has T[i*3+j, j*3+k, k*3+i] = 1
        # We want to minimize ||sum_r u_r v_r^T w_r^T - T||
        
        # Strategy: Random initialization + local search / optimization
        # Since we need exact integer/half-integer solutions, random float search is unlikely to hit them directly.
        # However, we can try to find a float solution and then round, or use algebraic structures.
        
        # Given the difficulty of finding rank-23 3x3x3 exactly via random search in limited time,
        # we will attempt a few random restarts with a simple gradient-free local search.
        
        best_residual = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random small values
            U = rng.randn(9, rank) * 0.5
            V = rng.randn(9, rank) * 0.5
            W = rng.randn(9, rank) * 0.5
            
            # Local search: try to reduce the residual
            # Residual R[i,j,k] = sum_r U[i*3+j, r] V[j*3+k, r] W[k*3+i, r] - 1
            
            for iter in range(1000):
                if time.time() > deadline - 10:
                    break
                
                # Compute current residual
                residual = np.zeros((3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                            residual[i,j,k] = s - 1.0
                
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-12:
                    # Found a solution!
                    # Check if it can be rounded to integers/half-integers
                    # For now, return as is if it's close enough
                    return U.tolist(), V.tolist(), W.tolist()
                
                if max_res < best_residual:
                    best_residual = max_res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Simple local search step: perturb one random element
                # This is very slow, but let's try a few iterations
                # Actually, gradient descent would be better, but let's keep it simple
                # to avoid complex dependencies.
                
                # Pick a random matrix and element
                mat_idx = rng.randint(0, 3)
                if mat_idx == 0:
                    idx = (rng.randint(0, 9), rng.randint(0, rank))
                    U[idx] += rng.randn() * 0.01
                elif mat_idx == 1:
                    idx = (rng.randint(0, 9), rng.randint(0, rank))
                    V[idx] += rng.randn() * 0.01
                else:
                    idx = (rng.randint(0, 9), rng.randint(0, rank))
                    W[idx] += rng.randn() * 0.01
        
        # Check if best found is good enough
        if best_residual < 1e-9:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None

    # Attempt to find a rank-23 decomposition for 3x3x3
    # This is the critical step. If successful, we get rank 46 for 3x3x6.
    # If not, we fall back to schoolbook 54.
    
    # Try rank 23
    result_3x3 = find_3x3x3_decomposition(23, seed=42, max_time=60)
    
    if result_3x3 is not None:
        U1, V1, W1 = result_3x3
        # Construct the full 3x3x6 decomposition
        # Block 1: k=0,1,2
        # Block 2: k=3,4,5
        
        rank_total = 46
        U_full = [[0.0] * rank_total for _ in range(9)]
        V_full = [[0.0] * rank_total for _ in range(18)]
        W_full = [[0.0] * rank_total for _ in range(18)]
        
        # Copy block 1 (columns 0-22)
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r] = U1[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    V_full[j*3+k][r] = V1[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[k*3+i][r] = W1[k*3+i][r]
        
        # Copy block 2 (columns 23-45)
        # Note: The indices for V and W shift for the second block
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U_full[i*3+j][r+23] = U1[i*3+j][r]
            for j in range(3):
                for k in range(3):
                    # k in 3x3x3 corresponds to k+3 in 3x3x6
                    V_full[j*3+(k+3)][r+23] = V1[j*3+k][r]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3+i][r+23] = W1[k*3+i][r]
        
        # Verify the full decomposition
        if is_valid_decomposition_full(U_full, V_full, W_full, rank_total):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: Schoolbook decomposition (rank 54)
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
# EVOLVE-BLOCK-END
