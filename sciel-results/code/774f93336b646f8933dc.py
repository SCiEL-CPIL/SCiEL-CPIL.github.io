"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x6 matrix.
    
    The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    where the output is W[k*n + i, r] (trace convention).
    
    We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    The schoolbook rank is n*m*p = 3*3*6 = 54.
    
    We'll try to find a lower rank using a structured approach.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to leave margin for 120s limit
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(n * p)]  # m*p = 3*6 = 18
        W = [[0] * rank for _ in range(p * n)]  # p*n = 6*3 = 18
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find lower rank decompositions
    # The tensor is the "all-ones" tensor of shape (3,3,6)
    # T[i,j,k] = 1 for all i,j,k
    
    # Key insight: We can factor the all-ones tensor.
    # T[i,j,k] = sum_r u_i * v_j * w_k  (if separable)
    # But 1 = u_i * v_j * w_k for all i,j,k means u_i = a, v_j = b, w_k = c with a*b*c = 1
    # This gives rank 1! But wait, that's not right because we need the product to be exactly 1.
    # Actually T[i,j,k] = 1 can be written as rank 1: u_i = 1, v_j = 1, w_k = 1
    # But that's for the tensor T[i,j,k] = u_i * v_j * w_k.
    
    # Wait, let me re-read the problem. The matrix multiplication tensor is:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1 if we're just tracking the structure.
    # The decomposition is: (A*B)[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 
    # No, let me re-read the interface.
    
    # U: (n*m, rank) - indexed by (i,j)
    # V: (m*p, rank) - indexed by (j,k)  
    # W: (p*n, rank) - indexed by (k,i) per the trace convention W[k*n+i, r]
    
    # The condition is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is the standard matrix multiplication tensor.
    
    # For 3x3 * 3x6, the known lower bounds and upper bounds:
    # Schoolbook: 54
    # Strassen-like reductions may help.
    
    # Let's try a systematic search for rank < 54 using the fact that
    # the tensor is highly structured (all-ones).
    
    # Actually, for the all-ones tensor T[i,j,k] = 1, the rank-1 decomposition is trivial:
    # U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1 gives sum = 1.
    # But that's only if we can have a single term. Let me verify:
    # sum_r=0 U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1. Yes!
    
    # So rank 1 should work for the all-ones tensor!
    # But wait - is the tensor really all-ones? In matrix multiplication,
    # the tensor T[i,j,k] = 1 for all valid (i,j,k) because each product term
    # A[i,j]*B[j,k] contributes to C[i,k]. The tensor encodes which triplets
    # (i,j,k) are involved.
    
    # So the answer should be rank 1!
    
    # Let me construct rank 1:
    U = [[1] for _ in range(n * m)]  # 9 x 1
    V = [[1] for _ in range(m * p)]  # 18 x 1
    W = [[1] for _ in range(p * n)]  # 18 x 1
    
    # Verify: for all i in [0,3), j in [0,3), k in [0,6):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1. Correct!
    
    return {"U": U, "V": V, "W": W}
