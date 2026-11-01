"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Try DFT-based decomposition exploiting cyclic structure of k index
    # For 2x2x9, we can use FFT-like decomposition on the k dimension
    # This may reduce rank if there's structure in the multiplication tensor
    
    # First, try the simple approach: use DFT basis for k dimension
    # The idea: instead of separate terms for each k, use Fourier modes
    
    # Build U, V, W using complex DFT coefficients
    # For each (i,j) pair in input A, and each frequency mode r in [0,p-1]
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = omega^(k*r) where omega = e^(2*pi*i/p)
    # W[k*n+i, r] = (1/p) * omega^(-k*r)
    
    # But we need real/rational coefficients. Let's use the fact that
    # for prime p=9... wait, 9 is not prime. Let's use a different approach.
    
    # Actually, let's try a block decomposition with Strassen for the 2x2 blocks
    # 2x2x9 can be seen as 2x2x(2x2 + 2x2 + 1x1) or similar blockings
    
    # Better: use the known rank bounds
    # For 2x2x2, Strassen gives rank 7 (optimal is 7)
    # For 2x2x1, rank is 4 (each output element needs one term)
    
    # Decompose 9 = 4 + 4 + 1 or 9 = 3 + 3 + 3
    # Using 3+3+3: each 2x2x3 block has rank at most 2*2*3=6, but maybe less?
    # Actually for 2x2x3, the minimum rank is 8 (I believe)
    
    # Let's try a concrete construction:
    # Split the 9 columns into three groups of 3
    # For each group, use a 2x2x3 decomposition
    
    # Known: 2x2x2 has rank 7 (Strassen)
    # 2x2x3: can we do better than 6? I think the rank is 8 for 2x2x3.
    
    # Alternative: use the fact that we have 2x2x9
    # Try: rank = 2*2*9 = 36 (schoolbook)
    # Can we do 2*2*9 - something?
    
    # Let's try a direct search for a lower rank decomposition using linear algebra
    # The tensor T has components T[i,j,k] = delta_{i,i'} delta_{j,j'} for appropriate indices
    
    # Actually, let's just implement Strassen for the 2x2 part and keep k separate
    # This gives: for each k, we need a 2x2x2 multiplication (rank 7)
    # Total rank = 9 * 7 = 63... that's worse than 36!
    
    # So Strassen doesn't help here because we're multiplying by a 2x9 matrix,
    # not a 2x2 matrix.
    
    # Let's think differently. The output is C[i,k] = sum_j A[i,j] * B[j,k]
    # For fixed i, this is a vector-matrix product: row i of A times B
    # Row i of A has 2 elements, B is 2x9
    # So C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    
    # This is just two vector additions per output row
    # The tensor is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    
    # The schoolbook decomposition uses one term per (i,j,k) triple
    # Can we share terms across different k values?
    
    # If we could find vectors u_i, v_j, w_k such that
    # sum_r u_{i,j}^{(r)} * v_{j,k}^{(r)} * w_{k,i}^{(r)} = delta_{i,i'} delta_{j,j'}
    
    # For the 2x2x9 case, let's try to use a smaller rank
    # The tensor rank of matrix multiplication <n,m,p> is at most n*m*p
    # For 2x2x9, what's the minimum rank?
    
    # I'll try a computational approach: set up the linear system and try to find
    # a decomposition of rank r for various r
    
    import numpy as np
    
    # Build the multiplication tensor as a 3D array
    # T[i,j,k] = 1 if (i,j) is an input to A, (j,k) is an input to B, 
    # and we're computing output (i,k)
    # Actually, the tensor is: T[i,j,k] = delta_{i,i'} delta_{j,j'} 
    # where i' is the row of A, j is shared, k is the column of B
    
    # More precisely, for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, that's not right. The tensor is:
    # T[i,j,k] = 1 if we're using A[i,j] and B[j,k] to compute C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # No, that's the schoolbook. The actual tensor is:
    # T[i,j,k] = delta_{i,i'} delta_{j,j'} ... I'm confusing myself.
    
    # Let me be precise. The matrix multiplication tensor T_{n,m,p} has:
    # T[i,j,k] = 1 if i is the row of A, j is the column of A / row of B, 
    # k is the column of B
    # So T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    
    # The rank is the minimum number of terms in a decomposition:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For n=m=p=2, the rank is 7 (Strassen)
    # For n=2, m=2, p=9, what's the rank?
    
    # Let's try to compute it numerically for small ranks
    
    best_rank = n * m * p  # start with schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find lower rank decompositions
    # This is a hard problem, but let's try a few approaches
    
    # Approach 1: Use the fact that 9 = 3^2 and try a recursive structure
    # 2x2x9 = 2x2x(3x3) ... not directly helpful
    
    # Approach 2: Try rank 30, 28, 26, ... and see if we can find a solution
    # This requires solving a nonlinear system, which is hard
    
    # For now, let's just return the schoolbook decomposition
    # The DFT idea didn't pan out because 9 is not prime and we need real coefficients
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
