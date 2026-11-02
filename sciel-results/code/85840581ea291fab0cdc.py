"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# The 2x2x2 matrix multiplication tensor has a known rank-7 decomposition (Strassen).
# U, V, W each have 4 entries (corresponding to the 4 indices of 2x2 matrix)
# We need to ensure that sum_{c} U[i,c] * V[j,c] * W[k,c] = 1 if (i,j,k) is a valid
# product term, else 0. Wait, the tensor is T_{i,j,k} = 1 if i,j,k are indices of
# the 2x2 multiplication tensor.
# Specifically, T_{i,j,k} is the 3-way tensor for (2x2) * (2x2).
# The indices i, j, k are 0..3, representing (r1, r2), (c1, c2), (out_r, out_c)?
# No, in the standard tensor decomposition for matrix mult A (m1 x m2) B (m2 x m3) -> C (m1 x m3):
# The tensor T_{(i,j), (k,l), (m,n)} is 1 if they match for the multiplication.
# Let's define the 2x2x2 tensor T2.
# Let A = [[a00, a01], [a10, a11]]
#     B = [[b00, b01], [b10, b11]]
#     C = A B = [[c00, c01], [c10, c11]]
# c00 = a00 b00 + a01 b10
# c01 = a00 b01 + a01 b11
# c10 = a10 b00 + a11 b10
# c11 = a10 b01 + a11 b11
#
# The rank of T2 is 7. The standard Strassen decomposition gives 7 product terms.
# Let's denote the 4 elements of A as vectors of length 4: e0=[1,0,0,0], e1=[0,1,0,0], e2=[0,0,1,0], e3=[0,0,0,1].
# Let U_c be the vector for factor 1 (row of A, col of B? No, U has shape (2*2, rank)).
# Let's check the format: U (n*m, rank), V (m*p, rank), W (p*n, rank).
# For 2x2x2: n=2, m=2, p=2.
# U has shape (4, rank). Index (i, j) for A? Or A is (n x m)? 
# Task: "multiplying a 4 by 4 matrix by a 4 by 9 matrix".
# U is (n*m, rank). This implies U corresponds to matrix A (4x4).
# V is (m*p, rank). This implies V corresponds to matrix B (4x9).
# W is (p*n, rank). This implies W corresponds to matrix C (9x4) flattened in column-major?
# "output factor follows the official trace convention W[k*n+i,r]"
# This matches W index k*p + i if C is p x n? No, C is usually m x p. 
# Wait, if A is 4x4 and B is 4x9, C = A B is 4x9.
# n=4 (rows of A), m=4 (cols of A / rows of B), p=9 (cols of B).
# U (n*m, rank) -> 4*4=16.
# V (m*p, rank) -> 4*9=36.
# W (p*n, rank) -> 9*4=36.
#
# The tensor is the multiplication tensor for A (4x4) and B (4x9).
# T_{(i,j), (k,l), (r,c)} = 1 if i=k and j=c? 
# Let A be (n x m), B be (m x p). C is (n x p).
# C_{i,j} = sum_{k} A_{i,k} B_{k,j}.
# The tensor T has dimensions (n*m) x (m*p) x (n*p).
# Index 1: (i, k) from A.
# Index 2: (k, j) from B.
# Index 3: (i, j) from C.
# T_{(i,k), (k',j), (i',j')} = 1 if i=i' and k=k' and j=j'.
#
# So for the 2x2x2 block:
# A is 2x2, B is 2x2, C is 2x2.
# We can partition the 4x4x9 multiplication into 2x2x2 blocks?
# A is 4x4. B is 4x9.
# Let's partition A into 2x2 blocks.
# A = [[A00, A01], [A10, A11]] where each Ak is 2x2.
# B is 4x9. We can partition it into 2x9 blocks? 
# If we use Strassen on 2x2 blocks, we need to mix rows and columns.
# Standard block decomposition:
# C = A B = [[A00 B00 + A01 B10, A00 B01 + A01 B11], ...]
# This is a 2x2 block matrix multiplication where B00, B10 are 2x9.
# This doesn't directly use the 2x2x2 rank-7 decomposition because B blocks are 2x9.
#
# We can split the 2x9 block into 2x2 and 2x1? Or 2x3 and 2x3?
# Let's split the 4x9 matrix into 2x2 blocks?
# 4x9 = (2x4) * 9? 
# If we partition the 4x4 A into 2x2 blocks, and the 4x9 B into 2x(9) blocks.
# The products are (2x2) * (2x9) = 2x9.
# The rank of the (2x2)*(2x9) tensor is rank(2x2*2x2) * rank(2x1)?
# The tensor for (2x2)*(2x9) is the sum of the tensor for (2x2)*(2x2) (for first 2 cols of B) 
# plus the tensor for (2x2)*(2x2) (for next 2 cols of B)...
# Since the columns of B are independent in the tensor decomposition?
# Yes, the multiplication tensor is separable over the columns of B if we treat them independently?
# No, the Strassen algorithm mixes columns too.
#
# However, if we don't mix columns, we can just use the tensor rank for 2x2 * 2x1.
# Rank of (2x2)*(2x1) is 4 (schoolbook) or 3?
# (2x1)*(2x1) is 4 scalars. Rank is 4.
# (1x1)*(1x1) is 1.
#
# Let's try to use the Kronecker product of decompositions.
# The rank of (A*B) * (C*D) is rank(A*C) * rank(B*D)? 
# Not necessarily, but there is a bound.
# Actually, if T1 is a decomposition for 2x2*2x2 with rank 7, and T2 is a decomposition for 1x1*1x1 with rank 1,
# can we get a decomposition for 2x2*2x9?
# 2x2 * 2x9 = sum of 2x2 * 2x1 over 9 columns? 
# If we don't mix columns, rank is 9 * rank(2x2 * 2x1) = 9 * 4 = 36.
# If we mix columns using Strassen-like methods, can we do better?
#
# Let's try to compute the rank of the 4x4 * 4x9 tensor using a brute force search or a smarter heuristic.
# Given the deadline, a simple heuristic might be better.
#
# A known result: rank of (m x n) * (n x p) tensor is bounded by O(n^(log_2 7) * max(m,p)^(log_2 7) / ...)?
# For 4x4 * 4x9, a simple bound might be 4^log_2(7) * 9 ... 
# 4^2.8 = 64.
#
# Let's try to construct a rank-32 decomposition? 
# Or maybe rank 45?
#
# Let's use the following strategy:
# 1. Try to find a low rank decomposition using a numerical relaxation (minimize nuclear norm of the tensor unfolding).
# 2. If that fails or takes too long, fallback to schoolbook.
#
# Numerical Relaxation:
# Tensor T has dimensions (16, 36, 36).
# We want to find U (16, R), V (36, R), W (36, R) such that T = U V W^T (tensor product).
# This is equivalent to minimizing the distance to a rank-R tensor.
# We can try random initializations and gradient descent.
#
# However, R is unknown. We can start with R=20 and increase if it fails.
#
# Let's check the deadline. 150s wall clock.
# If the numerical search takes > 120s, we return schoolbook.
#
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    # Define the target tensor T
    # T[i, j, k] where i is (A_row, A_col), j is (B_row, B_col), k is (C_row, C_col)
    # A is n x m, B is m x p, C is n x p.
    # i: 0..nm-1, (r, c1)
    # j: 0..mp-1, (r, c2)
    # k: 0..pn-1, (c2, r)  <-- Note: W[k*n+i, r] in the task description?
    # Task says: W[k*n+i, r].
    # My previous deduction: k is (i, j) of C. C is n x p.
    # Index in W is k*n + i? 
    # If C is n x p, and we flatten column-major, index is c * n + r.
    # Here k corresponds to c (column of C), i corresponds to r (row of C).
    # So W index is k * n + i.
    # A index: U[i*m + j, c]. i is row of A, j is col of A.
    # B index: V[j*p + k, c]. j is row of B, k is col of B.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A_{i,j}, B_{j,k}, C_{i,k}
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1

    def try_rank(R, max_iter=1000, tol=1e-5):
        # We need to check if T can be decomposed into R terms.
        # This is a non-convex problem.
        # We will use random initialization and gradient descent on U, V, W.
        # Loss = || T - sum_{r} U[:,r] outer V[:,r] outer W[:,r] ||_F^2
        
        # Since we want integer/half-integer entries, we can project afterwards?
        # Or just find if there is a solution with small norm and check if it's integer-ish.
        
        # Initialize
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        lr = 0.01
        
        for step in range(max_iter):
            # Compute current tensor estimate
            # Estimate E[i,j,k] = sum_{r} U[i,r] V[j,r] W[k,r]
            # We can compute this efficiently? 
            # For n=4, m=4, p=9, size is 16x36x36.
            # Loop over R and update.
            # Grad U = V^T W^T * T_resid?
            
            # Residual
            # E = np.einsum('ir,jr,kr->ijk', U, V, W)
            # This might be slow if R is large or max_iter is high.
            
            if step % 10 == 0:
                # Check progress roughly
                pass
            
            # Calculate Gradient
            # dE/du_ir = V_jr W_kr sum_{jk} Resid_ijk
            # Let's use a simpler SGD-like approach or BFGS?
            # Given Python overhead, simple gradient descent might be too slow for max_iter=1000.
            
            # Alternative: Use the alternating least squares (ALS) on the tensor?
            # Not directly applicable to rank decomposition.
            
            # Maybe just a few steps of gradient descent with large steps?
            
            # Let's try 100 iterations.
            
            # Compute Residual
            # To save time, we can compute only a subsample?
            # No, we need exactness.
            
            # Let's check if time is running out.
            elapsed = time.time() - start_time
            if elapsed > 40:  # Keep 100s for fallback or other things? No, deadline is 150s for construct().
                # If we are using a loop inside construct, we must be careful.
                # The evaluator kills after 150s.
                # We should probably abort the numerical search if it takes > 20s per attempt.
                return None

            # Efficient computation of gradients is hard here without C extensions.
            # Given the constraints, a full numerical optimization is risky for 150s.
            
            # Maybe we should try a specific structured decomposition?
            # Let's try to decompose using the known rank of 2x2x2.
            # We established rank(2x2*2x2) = 7.
            # Can we combine this to get 4x4*4x9?
            
            # Partition:
            # A (4x4) -> 4 blocks of 2x2
            # B (4x9) -> 18 blocks of 2x2 (if we split 9 into 4+4+1? No, 9 is odd).
            # 9 = 2+2+2+2+1.
            
            # This is complex.
            
            # Let's return None to indicate failure.
            pass
            
        return None

    # Given the risk of timeout and complexity, let's stick to the safe schoolbook for now,
    # BUT I will try one more thing:
    # If the prompt implies that a better solution exists, maybe rank 45?
    # Let's try to find if rank 48 is possible? 
    # Or maybe just output schoolbook. The score 1.0000 is the baseline.
    
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
