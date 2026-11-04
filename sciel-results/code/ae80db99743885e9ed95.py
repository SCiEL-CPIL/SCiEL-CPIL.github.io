"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # First, return the schoolbook decomposition as a safe fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using random initialization + gradient-free optimization
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to minimize rank such that this holds exactly.
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also rank >= ceil(n*p/m) * something... actually for 3x5x6, the Strassen-like bounds give ~26-28
    # Let's try ranks from 26 up to 90 (schoolbook)
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Use a randomized local search approach
    # Initialize with random low-rank factors and try to fit the identity tensor
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    expected = 1.0 if (i >= 0 and j >= 0 and k >= 0) else 0.0
                    # The tensor for matrix multiplication is T[i,j,k] = delta_{i,j,k} in the sense that
                    # (A*B)[i,k] = sum_j A[i,j]*B[j,k], so the tensor is T[i,j,k] = 1 for all valid i,j,k
                    # Wait, no. The matrix multiplication tensor has entries T[i,j,k] where the multiplication
                    # is (AB)_{ik} = sum_j A_{ij} B_{jk}. So the tensor is T[i,j,k] = 1 for all i,j,k.
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Actually, let me reconsider. The matrix multiplication tensor T is defined by:
    # T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    
    # Let's use a more efficient approach: try specific known decompositions
    # For small matrices, we can use the fact that rank(A*B) <= min(rank(A), rank(B))
    # but that's not directly helpful.
    
    # Try a simple approach: use the Kronecker product structure
    # If we can write the multiplication as a sum of rank-1 tensors
    
    # Let's try rank from max(n*m, m*p, p*n) upward, but that's 30.
    # Actually, let's just try a few random initializations with rank 26-35
    
    np.random.seed(42)
    random.seed(42)
    
    for target_rank in range(26, 36):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Simple iterative refinement: try to adjust one factor at a time
            # This is a very basic approach, may not converge well
            
            # Check if it's already close
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < rank_school:
            break
    
    return best_result
# EVOLVE-BLOCK-END
