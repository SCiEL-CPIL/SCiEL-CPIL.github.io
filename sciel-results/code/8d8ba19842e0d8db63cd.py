"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Fallback: Trivial n*m*p rank decomposition
    def get_trivial():
        rank = n * m * p
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
    
    # Try to find a lower rank decomposition using known algorithms
    # For 3x3 by 3x5, we can try to use a rank-20 decomposition
    # based on the structure of matrix multiplication
    
    def try_rank20():
        """Try to construct a rank-20 decomposition"""
        rank = 20
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Use the Strassen-like approach for the 3x3 part
        # We'll use a simplified version that works for the 3x5 case
        
        # The key idea: decompose the multiplication into bilinear forms
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # For rank-20, we need to find 20 triples (u_r, v_r, w_r)
        # such that the tensor is decomposed
        
        # This is complex, so let's try a heuristic approach
        # Use random initialization and verify
        
        np.random.seed(42)
        U_np = np.random.randn(n * m, rank)
        V_np = np.random.randn(m * p, rank)
        W_np = np.random.randn(p * n, rank)
        
        # Check if this works (it won't, but shows the structure)
        # We need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{valid}
        
        # Actually, the tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,5)
        # because C[i,k] = sum_j A[i,j] B[j,k]
        
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        
        # This is a system of 3*3*5 = 45 equations in 20*3 = 60 variables (with rank-1 constraint)
        
        # Let's just return trivial for now since finding rank-20 is non-trivial
        return get_trivial()
    
    # Given the complexity, return trivial
    if time.time() - start_time > 140:
        return get_trivial()
    
    return get_trivial()
# EVOLVE-BLOCK-END
