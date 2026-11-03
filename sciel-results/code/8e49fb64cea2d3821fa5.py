import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    # Build the schoolbook rank-36 decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # The tensor condition is: for all (i,j) and (j',k):
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{k,k'}
    # when i=i' and k=k', and 0 otherwise.
    
    # We want to find linear dependencies: if sum_r alpha_r * U[:,r] = 0, sum_r alpha_r * V[:,r] = 0, 
    # and sum_r alpha_r * W[:,r] = 0, we can eliminate those components.
    # More precisely, we look for a vector alpha such that the "product tensor" 
    # sum_r alpha_r * (U[:,r] outer V[:,r] outer W[:,r]) = 0.
    
    # This is equivalent to finding alpha in the null space of a certain linear map.
    # Let's construct the matrix M where each row corresponds to a (i,j,k) triple and each column to r,
    # with M[(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But that's not quite right. The condition is more subtle.
    
    # Actually, the correct formulation: define a 4-index tensor T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need: for all (i,j) and (j',k), sum_r T[i,j,k,r] * delta_{j,j'} = delta_{i,i'}*delta_{k,k'}
    # This means: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'}*delta_{k,k'} when j=j'
    
    # To find rank reductions, we can try to find a set of coefficients alpha_r such that
    # the "effective tensor" sum_r alpha_r * U[:,r] outer V[:,r] outer W[:,r] vanishes.
    # This is a system of linear equations in alpha.
    
    # Let's build the linear system. The tensor has dimensions (n*m) x (m*p) x (p*n).
    # Flatten it: index = (i*m+j) * (m*p) * (p*n) + (j'*p+k) * (p*n) + (k'*n+i')
    # But we only care about specific entries where the indices match the matrix multiplication pattern.
    
    # Simpler approach: try to find dependencies by checking if we can express one rank-1 component
    # as a linear combination of others.
    
    # For each component r, check if it's in the span of the others.
    # Build a matrix where each row is a "flattened" version of the rank-1 tensor.
    
    # The rank-1 tensor for component r is: T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need to check if T_r is in the span of {T_s : s != r}
    
    # Build a matrix A where A[entry, s] = T_s[entry] for all entries
    # Then check if column r is in the span of the other columns.
    
    # Number of entries: n*m * m*p * p*n = 3*3 * 3*4 * 4*3 = 9 * 12 * 12 = 1296
    
    # Let's build this matrix and check for dependencies.
    
    # Create all (i,j,k) triples
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i, j, k))
    
    num_entries = len(entries)
    
    # Build the matrix
    A = np.zeros((num_entries, rank))
    for idx, (i, j, k) in enumerate(entries):
        for r in range(rank):
            A[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Find the rank of A
    # Use SVD to find the null space
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        # The null space is spanned by the rows of Vt corresponding to zero singular values
        tol = 1e-10
        null_space = Vt[s < tol]
        
        if null_space.shape[0] > 0:
            # We found dependencies. Let's try to use them to reduce the rank.
            # Each null space vector gives a linear combination that vanishes.
            # We can use these to eliminate components.
            
            # For now, just report that we found dependencies
            pass
    except Exception as e:
        pass
    
    # Alternative approach: use the fact that for matrix multiplication, there are known
    # algebraic identities. For 3x3x4, we can try to use Winograd-style optimizations.
    
    # Let's try a different approach: use a structured decomposition.
    # For 3x3 matrix multiplication, the rank is 23 (best known).
    # For 3x3x4, we can use the 3x3x3 decomposition for the first 3 columns,
    # and add 9 more for the 4th column, giving rank 32.
    
    # But we can do better. Let's try to use a 2x2 block decomposition.
    # Split the 3x4 matrix into two 3x2 blocks.
    # Then C = A * B = A * [B1 | B2] = [A*B1 | A*B2]
    
    # For each 3x2 multiplication, we can use a rank-? decomposition.
    # The rank for 3x2 matrix multiplication is at most 6 (schoolbook).
    # But we can do better using Winograd's method.
    
    # Actually, let's just try to find a better decomposition using a search.
    
    # Use a random search to find a low-rank decomposition.
    # Start with rank 30 and try to find a valid decomposition.
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce the rank by finding linear dependencies
    # Use the null space approach more carefully
    
    # Check if we can reduce by 1
    for target_rank in range(rank - 1, max(20, rank - 10), -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a rank-target_rank decomposition
        # Use a greedy approach: start with the full decomposition and try to eliminate components
        
        # For each component, check if it can be eliminated
        # This is complex, so let's use a simpler heuristic
        
        pass
    
    # For now, return the schoolbook solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
