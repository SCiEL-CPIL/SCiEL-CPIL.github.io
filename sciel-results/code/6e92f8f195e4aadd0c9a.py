"""
Matrix multiplication 3x3x4 tensor decomposition.
Goal: find a rank-minimal exact decomposition.
Known: schoolbook rank = 36, border rank bound ~20, exact rank unknown but >20.
Strategy: use a structured approach that exploits the asymmetry (3x3x4).
We use a hybrid: start from a known low-rank construction if available,
otherwise use a gradient-based approach to minimize rank.

Actually, for 3x3x4, the exact rank is known to be 22 (from Alon et al. or similar).
But let's use a practical approach: since we need exact integer/half-integer entries,
and the problem is small, we can try to construct a decomposition of rank 22 or lower.

However, given the 150s time limit and the need for exactness, let's use a smarter approach:
1. Try to use a known algebraic construction if we can derive one.
2. Otherwise, use a randomized search with rounding to integers/half-integers.

Given the complexity, let's implement a practical solver that:
- Starts with the schoolbook decomposition (rank 36)
- Attempts to reduce rank by finding linear dependencies among the rank-1 tensors
- Uses exact arithmetic with fractions for verification

But for simplicity and reliability within the time limit, let's use a different strategy:
Since the task allows integer/half-integer entries, and the matrix multiplication tensor
has a known structure, we can use the fact that for multiplying m x n by n x p matrices,
there are known sub-cubic algorithms, but for 3x3x4 the best known is likely still 22-24.

Let's implement a randomized search that tries to find a low-rank decomposition:
- Use random integer vectors for U, V, W with rank r
- Check if they exactly decompose the tensor
- Start with r=20 and increase until we find a solution

This is computationally expensive, so we'll use a smarter initialization.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # The tensor T_{i,j,k} = delta_{i,k} for matrix multiplication C = AB
    # where A is n x m, B is m x p, C is n x p
    # Actually, the multiplication tensor is: T(a,b) = sum over (i,j,k) a_i * b_j * delta_{i,k} * c_k
    # In the standard form: T = sum_{i,j,k} e_i^* \otimes f_j \otimes g_k where e_i^* is the i-th basis of A^*, f_j of B^*, g_k of C^*
    # With the trace convention W[k*n + i, r], the third index is k*n + i.
    
    # Let's define the tensor explicitly
    # T[i, j, k*n + i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # All other entries are 0
    
    # We want to find U (N x rank), V (M x rank), W (P x rank) such that:
    # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the indices match the multiplication, else 0
    # Actually, the condition is:
    # (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor satisfies: T_{i,j,k} = A_{i,j} * B_{j,k} * C_{k,i} ... no.
    # 
    # The correct tensor decomposition: 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is equivalent to: sum_j A_{i,j} B_{j,k} - C_{i,k} = 0
    # The tensor is: T = sum_{i,j,k} e_i \otimes f_j \otimes g_k where the multiplication is:
    # <T, a \otimes b \otimes c> = sum_{i,j,k} a_i b_j c_{k} delta_{i,k} ... 
    # 
    # Let me restate: The bilinear map (A,B) -> C where C = A*B.
    # The tensor T satisfies: <T, a \otimes b \otimes c^*> = sum_{i,j,k} a_{i,j} b_{j,k} c_{i,k}
    # where a_{i,j} is A_{i,j}, b_{j,k} is B_{j,k}, c_{i,k} is C_{i,k}
    # 
    # So the tensor is: T = sum_{i,j,k} E_{i,j} \otimes F_{j,k} \otimes G_{i,k}
    # where E_{i,j} is the matrix with 1 at (i,j), F_{j,k} has 1 at (j,k), G_{i,k} has 1 at (i,k)
    # 
    # In the factorization: T = sum_r U_r \otimes V_r \otimes W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # 
    # The condition is: for all (i,j,k):
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 if this corresponds to a valid term, else 0
    # Actually, every (i,j,k) triple contributes exactly once to C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 for ALL i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right either. The tensor should satisfy:
    # For the bilinear map: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The associated trilinear form is: sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # So T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # This means: T = sum_{i,j,k} E_{i,j} \otimes F_{j,k} \otimes G_{i,k}
    # where E_{i,j} has 1 at position (i,j) in the n*m vector, etc.
    # 
    # So the condition for the decomposition is:
    # For all (i,j,k) in [0,n)x[0,m)x[0,p):
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    # And for all other index combinations (which don't exist since we're summing over all i,j,k), it's automatically satisfied.
    # 
    # Wait, the tensor only has non-zero entries at (i*m+j, j*p+k, k*n+i) for each (i,j,k).
    # So the condition is simply:
    # For all i in range(n), j in range(m), k in range(p):
    #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    
    # Let's verify this is correct by checking dimensions:
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (12, rank)  
    # W has shape (p*n, rank) = (12, rank)
    # For each (i,j,k), we access U[i*m+j], V[j*p+k], W[k*n+i]
    # This looks correct.
    
    # Try ranks from low to high
    for rank in range(20, 37):
        if time.time() > DEADLINE:
            break
        
        # Try to find a decomposition of this rank
        # Use a random search with integer/half-integer values
        best_solution = None
        
        # Number of attempts based on remaining time
        remaining = DEADLINE - time.time()
        num_attempts = min(100, max(10, int(remaining * 2)))
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE:
                break
            
            # Generate random integer or half-integer values
            # Scale to keep numbers small
            scale = 1
            U = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(N)]
            V = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(M)]
            W = [[random.choice([-scale, 0, scale]) for _ in range(rank)] for _ in range(P)]
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_solution = {"U": U, "V": V, "W": W}
                return best_solution
        
        if best_solution:
            return best_solution
    
    # Fallback: return the schoolbook decomposition
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
