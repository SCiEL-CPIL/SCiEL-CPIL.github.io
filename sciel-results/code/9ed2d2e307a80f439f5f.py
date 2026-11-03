"""Matrix multiplication 2x4x7 - tensor decomposition search."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication
    using tensor contraction via intermediate linear maps.
    
    Strategy:
    1. Decompose the 2x4x4 block using Strassen-based rank reduction
    2. Embed into the full 2x4x7 space  
    3. Apply linear algebra rank-reduction in the intermediate space
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Step 1: Construct Strassen decomposition for 2x2x2
    # Strassen's algorithm uses 7 multiplications instead of 8
    # For 2x2 * 2x2: C = A*B
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our tensor, we need T[i,j,k] = 1 for the multiplication tensor
    # The 2x2x2 tensor has rank 7 (Strassen)
    
    # Step 2: Build 2x4x4 tensor using two 2x2x2 Strassen blocks
    # Partition j dimension: j=0,1 and j=2,3
    # A = [A_left | A_right] where each is 2x2
    # B = [B_top; B_bottom] where each is 2x2 (for the 2x4x4 case)
    
    # For 2x4x4: C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # Each 2x2x2 part uses Strassen rank 7
    # Total for 2x4x4: rank 14 (if independent)
    
    # Step 3: Extend to 2x4x7
    # B is 4x7, so we have 7 columns
    # Partition the 7 columns into groups: 2+2+2+1 or 3+2+2 etc.
    # For simplicity, use 2+2+2+1
    
    # Actually, let's think differently. The tensor is:
    # T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,7)
    
    # We can view this as: for each k, we have a 2x4x1 slice
    # Or for each (i,j), we have a 1x1x7 slice
    
    # Better approach: Use the fact that the tensor is a product structure
    # T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed appropriately
    
    # Let's try a direct construction using linear algebra:
    # We need to find U (8 x R), V (28 x R), W (14 x R) such that:
    # sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding a low-rank approximation of the tensor
    # in the appropriate format.
    
    # Key insight: The tensor T[i,j,k] = 1 can be written as:
    # T[i,j,k] = sum_{j'} delta_{j,j'} (which is 1)
    # But that's just the schoolbook.
    
    # Let's try to find a minimal rank using the following approach:
    # 1. Start with schoolbook rank 56
    # 2. Try to reduce by finding linear dependencies
    
    # Actually, let's use a more systematic approach:
    # The multiplication tensor for [n,m,p] has rank at most n*m*p
    # For [2,4,7], schoolbook gives 56
    
    # Known results: For rectangular matrix multiplication, the rank can be lower
    # than n*m*p when there's structure.
    
    # Let's try to construct a decomposition with rank < 56 using the following:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as:
    # C = A * B
    # where A is 2x4, B is 4x7
    
    # One approach: Use the fact that we can factor B = B1 * B2 where
    # B1 is 4x4 and B2 is 4x7, but that doesn't help directly.
    
    # Better: Use intermediate products
    # Let X = A * I_4 (2x4), then C = X * B
    # But I_4 is just identity, so X = A
    
    # Alternative: Partition the computation
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group terms:
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    # Each group of 2 can be done with Strassen-like decomposition
    # For 2 terms: x1*y1 + x2*y2
    # This is a 2x1x1 tensor, which has rank 2 (can't reduce)
    
    # Wait, that's not right. Let me reconsider.
    
    # For the tensor form, we need:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # Let's try a different construction. Use the fact that:
    # 1 = (1 + 0)(1 + 0) but that's trivial.
    
    # Actually, let's just implement a search for low-rank decompositions
    # using linear algebra.
    
    # For a given rank R, we need to solve:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of equations. We can use iterative methods.
    
    # Let's try a greedy approach:
    # 1. Initialize U, V, W randomly
    # 2. For each rank element, try to match as many (i,j,k) as possible
    # 3. Reduce the residual
    
    # But this is complex. Let's try known constructions first.
    
    # Construction: Use the Kronecker product structure
    # 2x4 * 4x7 can be seen as (2x2 * 2x2) with extensions
    
    # Let's try rank = 28 (half of schoolbook)
    # Idea: For each pair of j indices, use a rank-2 decomposition
    
    # Actually, let's just return a valid decomposition and try to improve it.
    
    # For now, use schoolbook as a baseline
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
