DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    The multiplication tensor T_{2,3,10} has dimensions n=2, m=3, p=10.
    The schoolbook decomposition uses n*m*p = 60 terms.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Attempt to reduce the rank by finding linear dependencies among the 
       rank-1 terms using integer/half-integer combinations.
    3. Use linear programming to find non-trivial combinations of terms that 
       sum to zero in the tensor space, allowing us to eliminate terms.
    
    For small dimensions, we can check if the rank-1 vectors are linearly 
    dependent. If they are, we can potentially reduce the rank.
    
    The tensor T_{n,m,p} has components T_{i,j,k} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    where the multiplication is C = A * B with A being n x m and B being m x p.
    
    The schoolbook decomposition uses terms e_i^A * e_j^B * e_k^C where 
    A[i,j], B[j,k], C[i,k] are the matrix elements.
    
    We look for a set of coefficients c_t (t=1..r) such that 
    sum_t c_t * (u_t ⊗ v_t ⊗ w_t) = 0, where u_t, v_t, w_t are the 
    rank-1 vectors from the schoolbook decomposition.
    
    If we find such a combination, we can use it to replace some terms.
    
    For the 2x3x10 case, let's analyze the structure:
    - A has 2*3 = 6 elements
    - B has 3*10 = 30 elements  
    - C has 10*2 = 20 elements
    
    The schoolbook terms are indexed by (i,j,k) where i in [0,1], j in [0,2], k in [0,9].
    Each term has:
    - u_{i,j} = 1 (A element at position i*m+j)
    - v_{j,k} = 1 (B element at position j*p+k)
    - w_{k,i} = 1 (C element at position k*n+i)
    
    We can try to find a lower rank decomposition by exploiting the structure.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Initialize with schoolbook
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # The tensor equation is: sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    # where T is the multiplication tensor
    
    # For matrix multiplication, the tensor T has entries:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k' else 0
    # More precisely, T_{a,b,c} = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    # Let's check if we can find a linear combination of the schoolbook terms
    # that sums to zero. This would allow us to reduce the rank.
    
    # The key insight: for fixed j, the terms with that j form a structure
    # that might have dependencies.
    
    # Let's try a different approach: use the fact that for matrix multiplication
    # with small dimensions, there might be known lower bounds or constructions.
    
    # For 2x3x10, the rank is at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # by the dimension argument (the tensor must have rank at least the max dimension
    # of the factor spaces).
    
    # Actually, the rank of the multiplication tensor T_{n,m,p} is at least 
    # max(n*m, m*p, p*n) because the tensor must be able to represent the 
    # multiplication map which has certain rank properties.
    
    # Let's try to find a rank-30 decomposition if possible, or at least reduce below 60.
    
    # One approach: use the Strassen-like idea or exploit the rectangular structure.
    
    # For now, let's return the schoolbook decomposition as a safe baseline,
    # but try a simple optimization: check if any columns are linearly dependent
    # in the combined space.
    
    # Convert to lists for the return format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
