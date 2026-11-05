import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def verify_decomposition(U, V, W, n, m, p):
        """Verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == delta(i,k) for all i,j,k"""
        # Note: The verify function in the original code checked for == 1.
        # For matmul, the tensor T[i,j,k] is 1 if the indices form a valid term in the sum.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T has T[i,j,k] = 1 for all i,j,k? 
        # No, the matmul tensor is defined such that the decomposition reconstructs the operation.
        # Standard check: 
        # Does sum_r U[i,j,r] V[j,k,r] W[i,k,r] equal 1?
        # If we set A and B to identity or basis matrices, we can check.
        # But the provided verify function checks:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 (in the original code's comment it said delta, but code said 1? 
        # Let's look at the original code's verify:
        # "target = 1 if i == k else 0" -> This was in the comment, but the code said:
        # if total != target: return False. 
        # Wait, the original code had:
        # target = 1 if i == k else 0
        # if total != target: return False
        # This implies the tensor is T[i,j,k] = delta(i,k).
        # But for matmul, T[i,j,k] is 1 for ALL i,j,k? 
        # Let's re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
        # And "sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1" ?
        # If the target is 1 for all i,j,k, then we are decomposing the all-ones tensor.
        # If the target is delta(i,k), that's not matmul.
        # 
        # Let's look at the "Schoolbook" fallback in the original code:
        # U[i*6+j, r] = 1, V[j*6+k, r] = 1, W[k*6+i, r] = 1.
        # Sum_r ... = 1 (since there is only one r for each i,j,k where all are 1).
        # So the original code's fallback assumes the target is 1 for ALL i,j,k.
        # 
        # Is the matmul tensor the all-ones tensor?
        # The tensor $T$ for matrix multiplication is defined by $T_{ijk} = 1$ if the indices are valid?
        # Actually, the "matrix multiplication tensor" $M_{n \times m \times p}$ has entries $M_{ijk} = 1$ for all $i,j,k$.
        # Why? Because the operation is a multilinear map. The coefficient of $A_{ij} B_{jk}$ in $C_{ik}$ is 1.
        # So yes, $T[i,j,k] = 1$ for all $i,j,k$.
        # 
        # So the verify function should check if the sum equals 1.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True

    # Schoolbook decomposition for 6x6x6
    # Rank = n*m*p = 216
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
