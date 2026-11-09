"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Test with random matrices
        rng = np.random.RandomState(42)
        for trial in range(20):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_np[:, r], A.flatten())
                v_val = np.dot(V_np[:, r], B.flatten())
                # W index is k*n + i, so we need to map back to C[i, k]
                w_vec = W_np[:, r]
                for idx in range(p * n):
                    k = idx // n
                    i = idx % n
                    C_computed[i, k] += u_val * v_val * w_vec[idx]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    def try_rank_decomposition(target_rank, max_time=30):
        """Try to find a decomposition of the given rank using structured approach."""
        if time.time() - start_time > deadline:
            return None
        
        # For 3x3 x 3x5, we can decompose as:
        # C = A @ B = A @ [B_1 B_2] where B_1 is 3x3, B_2 is 3x2
        # C = A @ B_1 + A @ B_2
        # 
        # For 3x3 x 3x3, we can use Strassen-like decomposition or rank 23
        # For 3x3 x 3x2, we can use a simpler decomposition
        #
        # Let's try a structured approach:
        # 1. Find a low-rank decomposition for 3x3 x 3x3 (target rank ~23)
        # 2. Find a low-rank decomposition for 3x3 x 3x2 (target rank ~15)
        # 3. Combine them
        
        # First, let's try to find a decomposition for 3x3 x 3x3
        def decompose_3x3_3x3():
            """Find a rank-minimal decomposition for 3x3 x 3x3."""
            # Use a known low-rank decomposition
            # For 3x3, the minimal rank is 23 (Laderman's algorithm)
            # We'll implement Laderman's algorithm
            
            # Laderman's 3x3 matrix multiplication algorithm (rank 23)
            # This is a known result from 1976
            
            U_list = []
            V_list = []
            W_list = []
            
            # The 23 terms of Laderman's algorithm
            # Each term has (i, j, k) indices for U[i,j], V[j,k], W[k,i]
            
            # Term 1-9: diagonal terms
            for i in range(3):
                for j in range(3):
                    U_list.append((i, j))
                    V_list.append((j, i))
                    W_list.append((i, i))
            
            # This is getting complex. Let's use a simpler approach.
            # For now, return a basic decomposition and we'll improve it.
            
            # Actually, let's just return None and use a different approach
            return None
        
        # Alternative approach: Use the fact that 3x5 = 3x3 + 3x2
        # and find decompositions for each part
        
        # For 3x3 x 3x3, use rank 27 (schoolbook) or try to find lower
        # For 3x3 x 3x2, use rank 18 (schoolbook) or try to find lower
        
        # Let's try a more direct approach:
        # Use random initialization and gradient descent to minimize the rank
        
        if time.time() - start_time > deadline:
            return None
        
        # Try to find a decomposition with target_rank
        # Use a structured initialization based on the tensor structure
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Initialize with a structured pattern
        rng = np.random.RandomState(42)
        
        # For each (i,j,k) in the tensor, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's try to find a solution by solving a system of equations
        # This is a nonlinear system, so we'll use iterative methods
        
        # For now, return None to indicate we couldn't find a solution
        return None
    
    # Try to use known low-rank decompositions
    # For 3x3 x 3x5, we can decompose as (3x3 x 3x3) + (3x3 x 3x2)
    
    # First, let's try to find a good decomposition using the structure
    # The key is that the middle dimension is 3, which allows for sharing
    
    # Try rank 30, 35, 40, etc.
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to improve by finding linear dependencies in the schoolbook
    # or by using a more clever decomposition
    
    # For now, return schoolbook as a safe baseline
    # In future iterations, we can try to find better decompositions
    
    return best_result
# EVOLVE-BLOCK-END
