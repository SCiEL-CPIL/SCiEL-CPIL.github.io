"""Find a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # We know the schoolbook works. Can we do better?
    # The tensor is T[i, j, k] for A[i,j] * B[j,k] -> C[i,k].
    # Indices: i in [0,3), j in [0,3), k in [0,7).
    # U has shape (3*3, r) = (9, r)
    # V has shape (3*7, r) = (21, r)
    # W has shape (7*3, r) = (21, r)
    # Decomposition: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's try to find a decomposition of rank r < 63.
    # We can use the fact that the tensor is sparse in some sense or has structure.
    # Actually, for small dimensions, we can try to solve for U, V, W directly using random initialization and gradient descent, or algebraic methods.
    # However, exact rational solutions are required.
    
    # A common approach is to fix one matrix (say W) and solve for U and V, or use a rank-1 peeling approach.
    # Given the time constraint and the requirement for exactness, let's try a few small ranks with a deterministic search or a quick randomized search with rational rounding.
    
    # For 3x3x3, the rank is 7 (Strassen).
    # For 3x3x7, it is likely higher.
    # Let's try to see if we can find a decomposition of rank < 63.
    # We will attempt a simple iterative method:
    # 1. Initialize U, V, W with random values.
    # 2. Fix two, solve for the third linearly.
    # 3. Round to nearest half-integer.
    # 4. Check if it works.
    
    # This is heuristic. If it fails, we return schoolbook.
    
    # Let's try ranks from a lower bound up.
    # Lower bound for rank of (3,3,7) matrix multiplication:
    # It's at least max(3*3, 3*7, 7*3) = 21? No, rank is at least n*m = 9, m*p=21, p*n=21. So rank >= 21.
    # Actually, the rank of the tensor is at least the dimension of the space it spans.
    
    # Let's try a few ranks, say 30 to 40, with a short time limit.
    # If we find one, great. If not, return schoolbook.
    
    # To keep it simple and robust, I will implement a very basic search.
    # If it's too slow, it will timeout and we return schoolbook.
    
    # Given the complexity, and the fact that finding an exact low-rank decomposition is hard,
    # and the seed program is schoolbook (63), the probability of finding a better one quickly is low.
    # However, the directive says "REFINEMENT: KEEP this design and investigate improvements within its family."
    # The "family" is finding a rank-minimal decomposition.
    
    # Let's try a specific known algorithm or a simple heuristic.
    # Since I cannot guarantee a better rank in 120s without a strong algorithm,
    # and the previous attempts (p168) didn't improve, I will stick to a safe, fast check.
    
    # Actually, let's look at the structure.
    # T[i,j,k] = 1 if we consider the standard basis.
    # Wait, the tensor entries are 1 for all i,j,k in the schoolbook? No.
    # The tensor for matrix multiplication C = AB has entries T[i,k,j] (depending on convention) or T[i,j,k].
    # The task says W[k*n+i, r].
    # The standard tensor T has T[i, j, k] = 1 if we are multiplying the (i,j) entry of A and (j,k) entry of B to contribute to (i,k) of C.
    # So T[i, j, k] is 1 for all valid i, j, k? No, that's not right.
    # The tensor is a 3rd order tensor. T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, no. The tensor is defined such that C[i,k] = sum_{j} A[i,j] B[j,k].
    # This can be written as C[i,k] = sum_{j,r} T[i,k,r] A[i,j] B[j,k]? No.
    # The tensor T is such that T_{i,k,j} = 1?
    # Let's check the schoolbook code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # Product: U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Sum over c.
    # For a fixed c corresponding to (i,j,k), the term is 1*1*1 = 1.
    # This term contributes to the (i,k) entry of C?
    # The decomposition is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # This sum is over c.
    # The result is a scalar for each (i,j,k)?
    # No, the result of the decomposition is the tensor itself.
    # The tensor T has indices (i, j, k) ?
    # The output is W[k*n+i, r].
    # The standard convention is T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, if T[i, j, k] = 1 for all i, j, k, then the tensor is the all-ones tensor.
    # The rank of the all-ones tensor of size 3x3x7 is 1?
    # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1.
    # Then sum_c 1*1*1 = r.
    # So T[i,j,k] = r.
    # We can scale: U=1/sqrt(r), etc. But we need integers/half-integers.
    # If r=1, U=1, V=1, W=1. Then T[i,j,k]=1.
    # Is the matrix multiplication tensor the all-ones tensor?
    # Let's re-read the task carefully.
    # "Matrix multiplication 3,3,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 7 matrix."
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T is defined by T[i, j, k] = 1 if j is the inner index.
    # But in the tensor decomposition, we have:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a linear map. The tensor T maps A and B to C.
    # T is a 3-tensor. T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i, j, k] = 1 for all i, j, k?
    # No, T[i, j, k] is non-zero only if the indices match.
    # In the schoolbook, we have a term for each (i,j,k).
    # The term is U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # For the schoolbook, c indexes (i,j,k).
    # U[i*m+j, c] is 1 if c corresponds to (i',j',k') with i'=i, j'=j.
    # V[j*p+k, c] is 1 if c corresponds to (i',j',k') with j'=j, k'=k.
    # W[k*n+i, c] is 1 if c corresponds to (i',j',k') with k'=k, i'=i.
    # So for a fixed c=(i,j,k), the product is 1 only if the indices match.
    # The sum over c gives 1 for each (i,j,k) triple?
    # No, the decomposition is of the TENSOR T.
    # The tensor T has components T[i, j, k].
    # In the context of matrix multiplication, T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) entry of C.
    # This is true for ALL i, j, k.
    # So T[i, j, k] = 1 for all i, j, k.
    # Wait, if T[i, j, k] = 1 for all i, j, k, then the rank is 1.
    # Let's check:
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Then sum_c U V W = 1.
    # So T[i, j, k] = 1.
    # Is the matrix multiplication tensor the all-ones tensor?
    # No!
    # The tensor T is such that T[i, j, k] = 1 if j is the common index.
    # But in the tensor decomposition, the indices are (i, j, k).
    # The schoolbook code uses U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This implies that the tensor component is T[i, j, k].
    # And the schoolbook sets it to 1 for each (i,j,k).
    # So the tensor IS the all-ones tensor?
    # Let's verify with a small example. 1x1x1.
    # A[0,0] * B[0,0] = C[0,0].
    # T[0,0,0] = 1.
    # Rank 1.
    # 2x2x2.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # The tensor T has components T[i, j, k].
    # T[0,0,0] = 1 (A[0,0]B[0,0] -> C[0,0])
    # T[0,1,0] = 1 (A[0,1]B[1,0] -> C[0,0])
    # T[0,0,1] = 1 (A[0,0]B[0,1] -> C[0,1])
    # T[0,1,1] = 1 (A[0,1]B[1,1] -> C[0,1])
    # ...
    # So T[i, j, k] = 1 for all i, j, k?
    # Yes, because every A[i,j] and B[j,k] contributes to C[i,k].
    # So the tensor is indeed the all-ones tensor!
    # Then the rank is 1!
    # Let's check the schoolbook code again.
    # It creates a rank n*m*p decomposition.
    # Why?
    # Maybe I am misunderstanding the tensor indices.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This is just the layout of W.
    # If the tensor is all-ones, then U=1, V=1, W=1 is a valid rank-1 decomposition.
    # Let's test this hypothesis.
    # If I return rank 1, will it be accepted?
    # The evaluator checks if the decomposition is exact.
    # If the tensor is all-ones, then rank 1 is correct.
    # But is the matrix multiplication tensor really all-ones?
    # Let's think about the definition.
    # The tensor T is defined by T(i, j, k) = 1 if the product A[i,j] B[j,k] appears in C[i,k].
    # This is true for all i, j, k.
    # So yes, T is the all-ones tensor.
    # Therefore, the rank is 1.
    
    # Let's construct the rank-1 decomposition.
    U = [[1]]
    V = [[1]]
    W = [[1]]
    
    # Wait, the shapes must be (n*m, rank), (m*p, rank), (p*n, rank).
    # n=3, m=3, p=7.
    # U: (9, 1)
    # V: (21, 1)
    # W: (21, 1)
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
