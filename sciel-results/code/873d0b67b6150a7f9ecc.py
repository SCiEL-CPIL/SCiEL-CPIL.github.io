import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Use the explicit rank-24 construction via slice rank method
    # Partition the 7 output columns into: {0,1,2}, {3,4,5}, {6}
    # But use a smarter decomposition
    
    # Actually, let's construct a valid decomposition with rank 24
    # Strategy: For each of the 7 columns, we need to compute C[:,k] = A @ B[:,k]
    # We can group columns and share computation
    
    # Let's use a direct algebraic construction
    # The tensor T[i,j,k] = 1 for all i,j,k in respective ranges
    
    # For 3x3x3, minimal rank is 21. For 3x3x7, let's try to find rank 24.
    
    # Construction idea: 
    # Use 3 groups of columns: G0={0,1,2}, G1={3,4,5}, G2={6}
    # For G0 and G1, use rank-9 decompositions (since 3x3x3 can be done in rank 9? No, it's 21)
    
    # Let me just build a valid decomposition with a reasonable rank
    # and verify it works.
    
    # Start with a known valid decomposition and try to minimize rank
    
    # For now, let's construct rank 24 explicitly
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Construction: use 8 factors for each of 3 column-groups
    # Group 0: columns 0,1,2 -> factors 0-7
    # Group 1: columns 3,4,5 -> factors 8-15
    # Group 2: column 6 -> factors 16-23
    
    # For a 3x3x3 multiplication, we can use a rank-9 decomposition (not minimal but valid)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Simple rank-9 decomposition for 3x3x3:
    # For each (i,k) pair, have one factor
    # U[i,j,r] = 1 if r corresponds to (i,k) and j is free... no that doesn't work
    
    # Let me use the standard bilinear decomposition
    # For 3x3x3: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is rank 9 with:
    # For each j in 0..2, k in 0..2: factor r = j*3+k
    # U[i, j*3+k] = A[i,j] -> U[i*m + j, r] where r encodes (j,k)
    # Wait, the tensor is T[i,j,k] and we decompose as sum U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    # For the bilinear form: T[i,j,k] = delta(i,j,k) is not right
    # The multiplication tensor: M[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entry is M[i,j,k] = 1 if this term contributes, i.e., always 1 for valid i,j,k
    # But the structure is: contribution to C[i,k] from A[i,j]*B[j,k]
    
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait no, it should equal the tensor value which is 1 for the multiplication tensor
    
    # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k (it's the identity tensor in this context)
    # No wait - the multiplication tensor T has T[i,j,k] = 1 iff the term A[i,j]*B[j,k] contributes to C[i,k]
    # For standard multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all i,j,k
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # For rank 9 (one factor per j,k pair won't work directly)
    
    # Let me use a different approach: construct U, V, W such that the condition holds
    
    # For factor r corresponding to "multiply A[i,j] by B[j,k] and add to C[i,k]":
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for the specific (i,j,k) and 0 otherwise
    
    # This is hard with few factors. Let me use the schoolbook approach but optimized
    
    # Schoolbook: rank = n*m*p = 63
    # For each (i,j,k): one factor
    # U[i*m+j, r(i,j,k)] = 1
    # V[j*p+k, r(i,j,k)] = 1  
    # W[k*n+i, r(i,j,k)] = 1
    
    # This gives rank 63. Can we do better?
    
    # Let me try to find a rank-24 decomposition by solving the system
    
    # Actually, let me just return a valid decomposition and see what rank we get
    # The current code returns rank 63. Let me try to improve it.
    
    # For now, let's construct a valid rank-63 decomposition (schoolbook)
    # and then try to reduce it
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    # Verify
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
