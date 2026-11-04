"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Schoolbook rank = 2*3*5 = 30
    
    # Strategy: Embed the 2x3 * 3x5 multiplication into a 3x3 * 3x3 multiplication.
    # Let A be 2x3, B be 3x5.
    # Pad A to 3x3: A' = [[A], [0, 0, 0]] (3 rows, 3 cols)
    # Pad B to 3x3: B' = [[B[:, :3]], [0, 0, 0]] (3 rows, 3 cols)
    # Note: B is 3x5. We take the first 3 columns for the 3x3 core. 
    # Wait, the multiplication A*B results in a 2x5 matrix.
    # The 3x3 product A'*B' will be a 3x3 matrix.
    # Let's analyze the mapping more carefully.
    
    # The tensor for matrix multiplication C = A * B is defined by C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T is indexed by (i, j, k) where T[i, j, k] = 1.
    # Dimensions: i in [0, n), j in [0, m), k in [0, p).
    # Here n=2, m=3, p=5.
    
    # We want to use a rank-23 decomposition for 3x3 matrix multiplication.
    # Let the 3x3 multiplication be C' = A' * B', where A' is 3x3 and B' is 3x3.
    # C'[i', k'] = sum_{j'} A'[i', j'] B'[j', k'].
    # The tensor T' for 3x3 has T'[i', j', k'] = 1 for i', j', k' in {0, 1, 2}.
    
    # We can map our 2x3x5 problem to a subset of the 3x3x3 problem?
    # Actually, the rank of a sub-tensor is less than or equal to the rank of the full tensor.
    # If we can express our tensor as a restriction of the 3x3 tensor, we can use the 3x3 decomposition.
    
    # Let's define the mapping:
    # Our indices: i in {0, 1}, j in {0, 1, 2}, k in {0, 1, 2, 3, 4}.
    # 3x3 indices: i' in {0, 1, 2}, j' in {0, 1, 2}, k' in {0, 1, 2}.
    
    # This direct mapping doesn't work because p=5 > 3.
    # However, we can split the 5 columns into two parts: 3 and 2.
    # Or, we can just use the schoolbook for the extra parts and Strassen for the core?
    # No, the rank of the sum of tensors is not simply the sum of ranks, but we can construct a block decomposition.
    
    # Alternative approach: 
    # The rank of M_{2,3,5} is known to be less than 30.
    # Is it 23? 
    # Let's check if we can embed M_{2,3,5} into M_{3,3,5} or similar.
    # Actually, a simpler structural change:
    # Use the known rank-23 decomposition for 3x3 matrices.
    # We can compute A (2x3) * B (3x5) by:
    # 1. Pad A to 3x3 by adding a row of zeros. A_pad = [[A], [0,0,0]]
    # 2. We cannot simply pad B to 3x3 because B has 5 columns.
    #    But we can split B into B1 (3x3) and B2 (3x2).
    #    C = A * B1 + A * B2.
    #    A * B1 is 2x3. A * B2 is 2x2.
    #    
    #    Let's compute A * B1 using Strassen (rank 23) on the 3x3 block.
    #    Let's compute A * B2 using schoolbook (rank 2*3*2 = 12).
    #    Total rank <= 23 + 12 = 35. This is worse than 30.
    
    # What if we split A?
    # A is 2x3. We can't really split it to use Strassen efficiently without padding.
    
    # Let's look at the dimensions again.
    # n=2, m=3, p=5.
    # Maybe we can use a rank-15 decomposition for 2x2x2 (Strassen) and combine?
    # 
    # Actually, let's look at the "Inspiration 2" which scored 1.0714.
    # 1.0714 = 30 / 28. So it found a rank 28 decomposition.
    # 1.0714 approx 30/28.
    # 
    # Can we do better than 28?
    # The lower bound for M_{2,3,5} is not well established but likely around 20-25.
    # 
    # Let's try a different structural approach: 
    # Use the explicit rank-23 Strassen decomposition for 3x3 matrices.
    # We can compute the 2x3 * 3x5 product by embedding it in a 3x3 * 3x5 product?
    # No, let's stick to the prompt's suggestion: "map ... onto the standard 3x3 by 3x3 multiplication tensor".
    # 
    # How to map 2x3 * 3x5 to 3x3 * 3x3?
    # It's not a direct subspace embedding because p=5 != 3.
    # However, we can treat the 5 columns as two separate multiplications:
    # C[:, 0:3] = A * B[:, 0:3]  (2x3 * 3x3 -> 2x3)
    # C[:, 3:5] = A * B[:, 3:5]  (2x3 * 3x2 -> 2x2)
    #
    # For the first part (2x3 * 3x3):
    # Pad A to 3x3: A_pad = [[A], [0,0,0]].
    # Compute A_pad * B1 (3x3 * 3x3). Result is 3x3.
    # The top 2 rows of the result are A * B1.
    # This uses rank 23 (Strassen for 3x3).
    #
    # For the second part (2x3 * 3x2):
    # Pad B2 to 3x3: B2_pad = [[B2], [0,0,0]]? No, B2 is 3x2. Pad to 3x3 by adding a column of zeros.
    # Pad A to 3x3: A_pad = [[A], [0,0,0]].
    # Compute A_pad * B2_pad (3x3 * 3x3). Result is 3x3.
    # The top 2 rows and left 2 cols are A * B2.
    # This uses rank 23 (Strassen for 3x3).
    #
    # Total rank = 23 + 23 = 46. Worse than 30.
    
    # This suggests that splitting and using full Strassen blocks is inefficient for rectangular matrices.
    
    # Let's reconsider the rank 28 result from Inspiration 2.
    # How did it get 28?
    # 30 - 2 = 28. Maybe it found 2 efficient rank-1 decompositions?
    # Or maybe it used a specific algorithm.
    
    # Let's try to construct a rank-25 or rank-24 decomposition manually if possible, 
    # or fall back to a robust search for rank < 30.
    
    # Given the time limit and complexity, a "structural change" that is safe is to implement 
    # a known lower-rank algorithm if one exists, or a more efficient search.
    #
    # However, there is a known result: 
    # The rank of M_{2,3,5} is at most 25? 
    # Actually, let's look at the bilinear complexity.
    # 
    # Let's try a different strategy: 
    # Use the "Coppersmith-Winograd" style or just a random search with better initialization?
    # The current program just returns schoolbook.
    #
    # Let's try to implement a rank-28 decomposition explicitly if we can find one, 
    # or use a solver to find a rank-R decomposition for R=28, 27, etc.
    #
    # Since I cannot run code here, I will implement a solver that attempts to find a 
    # rank-R decomposition using alternating least squares, but with a tighter 
    # constraint set (integers/half-integers) and a better initialization based on 
    # the schoolbook structure.
    #
    # Actually, a very effective structural change for these specific small dimensions 
    # is to use the fact that the tensor is sparse or has symmetry.
    #
    # Let's try to return a rank-28 decomposition. 
    # One way to get 28: 
    # Use Strassen for the 2x2 core (rank 7) and schoolbook for the rest?
    # A (2x3) * B (3x5).
    # Split B into B1 (3x2) and B2 (3x3).
    # A * B1: 2x3 * 3x2. Rank <= 2*3*2 = 12.
    # A * B2: 2x3 * 3x3. 
    #   Pad A to 3x3? No.
    #   Use Strassen for 3x3? Rank 23.
    #   Total 12 + 23 = 35.
    #
    # What if we split A?
    # A1 (1x3), A2 (1x3).
    # A1 * B: 1x3 * 3x5. Rank <= 1*3*5 = 15.
    # A2 * B: 1x3 * 3x5. Rank <= 15.
    # Total 30.
    #
    # Let's use the rank-7 Strassen for 2x2.
    # We can compute A * B by:
    # 1. Take the first 2 columns of B: B1 (3x2).
    # 2. Compute A * B1. A is 2x3. B1 is 3x2.
    #    Pad A to 3x3? No.
    #    Pad B1 to 3x3? No.
    #    
    #    Let's embed 2x3 * 3x2 into 3x3 * 3x3.
    #    A_pad (3x3) = [[A], [0,0,0]]
    #    B1_pad (3x3) = [[B1], [0,0,0]]? No, B1 is 3x2. Pad to 3x3 by adding col of zeros.
    #    Product C_pad = A_pad * B1_pad (3x3).
    #    Top-left 2x2 block is A * B1.
    #    Rank 23 (Strassen).
    #    
    #    Remaining part: B2 (3x3) (columns 3,4,5 of B).
    #    Compute A * B2.
    #    Pad A to 3x3. B2 is 3x3.
    #    Product C2_pad = A_pad * B2 (3x3).
    #    Top 2 rows are A * B2.
    #    Rank 23 (Strassen).
    #    
    #    Total 46.
    
    # This approach is not working to beat 30.
    
    # Let's look at the "Inspiration 2" score 1.0714 again.
    # 30 / 1.0714 = 28.
    # So a rank 28 decomposition exists.
    # How to get 28?
    # 30 - 2 = 28.
    # Maybe 2 of the schoolbook terms can be combined?
    # Or maybe a different splitting.
    
    # Let's try to find a rank-28 decomposition using a solver in the code.
    # We will try ranks from 28 down to 20.
    # We will use a randomized approach with projection onto the integer/half-integer lattice.
    
    def construct_rank_R(R):
        """
        Attempt to find a rank-R decomposition.
        Returns (U, V, W) or None.
        """
        n, m, p = DIMENSIONS
        # Target tensor T[i,j,k] = 1
        # We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to sum to 1.
        
        # Initialize with schoolbook and perturb
        # Schoolbook has rank n*m*p = 30.
        # We need to reduce rank to R.
        
        # Strategy: Start with random U, V, W of shape (n*m, R), (m*p, R), (p*n, R).
        # Use ALS (Alternating Least Squares) to minimize the error.
        # Then round to half-integers and check.
        
        # To make it faster, we can fix one matrix and solve for the others?
        # No, it's a 3-way tensor decomposition.
        
        # Let's use a simple random search with local optimization.
        # Given the small dimensions (6, 15, 10), we can afford some computation.
        
        np.random.seed(42)
        
        # Initial guess: Scale schoolbook?
        # Or just random.
        
        U = np.random.randn(n*m, R)
        V = np.random.randn(m*p, R)
        W = np.random.randn(p*n, R)
        
        # Normalize to prevent overflow
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        # ALS iteration
        for it in range(50):
            # Update U
            # For each row i*m+j, we want U[i*m+j, :] * (V[j*p+k, :] * W[k*n+i, :]) to approximate T[i,j,k]
            # This is tricky because V and W are coupled.
            # 
            # Standard ALS for CP decomposition:
            # Fix V, W. Solve for U.
            # For a given index (i, j), the row of U is u_ij.
            # The contribution to the tensor is u_ij_r * v_jk_r * w_ki_r.
            # We want sum_r u_ij_r * (v_jk_r * w_ki_r) = T[i,j,k] for all k.
            # Let M_k = v_jk_r * w_ki_r (element-wise product of row j of V and row k of W? No.)
            # V is (m*p, R). Row index is j*p+k.
            # W is (p*n, R). Row index is k*n+i.
            # So for fixed i, j, we have a vector of length R for U.
            # And for each k, we have a vector of length R: V[j*p+k, :] * W[k*n+i, :] (element-wise).
            # We want U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1 for all k.
            # This is a linear system for U[i*m+j, :].
            # Let A_ij be a matrix of size p x R.
            # A_ij[k, r] = V[j*p+k, r] * W[k*n+i, r].
            # We want A_ij @ U[i*m+j, :] = 1 (vector of ones of length p).
            
            # Solve for all i, j.
            for i in range(n):
                for j in range(m):
                    A_ij = np.zeros((p, R))
                    for k in range(p):
                        A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A_ij @ u = ones(p)
                    # Use least squares
                    try:
                        u, residuals, rank, sv = np.linalg.lstsq(A_ij, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Similarly for V
            for j in range(m):
                for k in range(p):
                    A_jk = np.zeros((n, R))
                    for i in range(n):
                        A_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, residuals, rank, sv = np.linalg.lstsq(A_jk, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Similarly for W
            for k in range(p):
                for i in range(n):
                    A_ki = np.zeros((m, R))
                    for j in range(m):
                        A_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check error
            # Compute tensor
            T_recon = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i,j,k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            
            error = np.linalg.norm(T_recon - np.ones((n,m,p)))
            if error < 1e-6:
                # Found a solution! Now round to half-integers.
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                T_recon_int = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon_int[i,j,k] = np.dot(U_int[i*m+j, :], V_int[j*p+k, :] * W_int[k*n+i, :])
                
                if np.allclose(T_recon_int, np.ones((n,m,p)), atol=1e-9):
                    return U_int, V_int, W_int
                else:
                    # Rounding failed. Try to fix?
                    # For now, return None and try next R.
                    return None
        
        return None
    
    # Try ranks from 28 down to 20
    for R in range(28, 19, -1):
        res = construct_rank_R(R)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
