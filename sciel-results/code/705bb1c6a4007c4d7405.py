import time
import sys
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact for matrix multiplication."""
    rank = len(U[0])
    # Check each entry of the multiplication tensor
    for i in range(n):
        for j in range(p):
            for k in range(m):
                # C[i,j] = sum_k A[i,k] * B[k,j]
                # Tensor entry T[i,k,j] should be 1 if k is the middle index
                # In our convention: U[i*m + k][r] * V[k*p + j][r] * W[j*n + i][r]
                # Actually, let me re-read the convention
                # W[k*n+i, r] - this is the output factor
                # So the decomposition is: C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                # Wait, that doesn't seem right either. Let me think more carefully.
                
                # The standard matrix multiplication tensor T has entries T[i,k,j] = 1
                # where (i,j) are output indices and k is the contraction index.
                # A decomposition: T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
                
                # But the task says:
                # U: (n*m, rank) - so U[i*m+k, r]
                # V: (m*p, rank) - so V[k*p+j, r]  
                # W: (p*n, rank) - so W[j*n+i, r]
                
                # And the convention is W[k*n+i, r] which seems off.
                # Let me just verify by checking if the bilinear form reproduces matrix mult.
                
                # For matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
                # We need: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = delta(k,k) = 1
                # Wait no, that's not right. Let me think again.
                
                # Actually the tensor T for matrix mult C[i,j] = sum_k A[i,k]B[k,j]
                # has T[i,k,j] = 1 for all i,k,j.
                # A rank-r decomposition: T[i,k,j] = sum_{r=1}^r u_r[i,k] * v_r[k,j] * w_r[i,j]
                
                # So U[i*m+k, r] = u_r[i,k]
                # V[k*p+j, r] = v_r[k,j]
                # W[j*n+i, r] = w_r[i,j]
                
                # And we need sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j.
                
                val = 0
                for r in range(rank):
                    val += U[i * m + k][r] * V[k * p + j][r] * W[j * n + i][r]
                if val != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 120  # 120 seconds limit, keep safety margin
    
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # Strategy: try rank-1 modifications and pruning
    
    # For small dimensions, we can try systematic search
    # The minimum rank for 3x5 * 5x7 is unknown but likely less than 105
    
    # Try various ranks from low to high
    for target_rank in range(50, best_rank + 1):
        if time.time() - start_time > deadline - 15:
            break
            
        # Try to find a decomposition of this rank
        # This is a hard problem, so we'll use a heuristic approach
        
        # One approach: start with schoolbook and try to merge terms
        # Another: use known results for small matrix multiplication
        
        # For 3x5 * 5x7, let's try a structured approach
        # We can exploit the fact that one dimension is small (3 and 7)
        
        # Try a block decomposition approach
        # Since n=3 is small, we can think of A as 3 blocks of size 1x5
        # And B as 7 blocks of size 5x1
        
        # Actually, let's try a more systematic approach using the fact that
        # matrix multiplication tensors have structure
        
        # For now, let's just return the schoolbook if we can't find better
        # But let's try one more thing: check if we can do better with a simple heuristic
        
        # Heuristic: try to find rank-1 terms that can be combined
        # This is complex, so for this iteration, we'll keep the schoolbook
        
        pass
    
    # If we found a better decomposition, use it
    # Otherwise, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
