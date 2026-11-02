"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find better decomposition using known constructions
    # For 3x3x3, Laderman's algorithm gives rank 23
    # For 3x3x4, we can try to extend or use block methods
    
    # Strategy: Try random low-rank decompositions and verify
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # must equal 1 if (i,j,k) is a valid index triple, else 0
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the MM tensor."""
        n, m, p = DIMENSIONS
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank):
        """Faster verification using numpy."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # This is equivalent to: the tensor T = sum_r outer(u_r, v_r, w_r) equals the MM tensor
        # where u_r is the r-th column of U, etc.
        # Let's compute the reconstructed tensor and compare
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this as: for each r, outer product of vectors
        # But faster: reshape and use einsum
        U_reshaped = U_arr.reshape(n, m, rank)  # U[i,j,r]
        V_reshaped = V_arr.reshape(m, p, rank)  # V[j,k,r]
        W_reshaped = W_arr.reshape(p, n, rank)  # W[k,i,r]
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)
    
    # Try Laderman-inspired approach for 3x3x3 part
    # Laderman's 23-term algorithm for 3x3x3:
    # The 23 multiplications are:
    # M1 = A11*(B11+B22+B33)
    # M2 = A22*(B11+B22+B33)
    # M3 = A33*(B11+B22+B33)
    # M4 = A12*(B21+B31)
    # M5 = A23*(B13+B33)
    # M6 = A31*(B12+B22)
    # M7 = (A11+A22)*(B22+B33)
    # M8 = (A11+A33)*(B11+B33)
    # M9 = (A22+A33)*(B11+B22)
    # M10 = (A12+A23)*(B23+B31)
    # M11 = (A23+A31)*(B12+B13)
    # M12 = (A31+A12)*(B21+B32)
    # M13 = (A11+A12+A22+A23)*(B22+B33)  -- actually let me look this up properly
    
    # Let me use a different approach: random search with local optimization
    # Start with schoolbook and try to reduce rank by merging
    
    # Alternative: Use the fact that for 3x3x4, we can view it as 4 separate 3x3x1 problems
    # or use block decomposition
    
    # Let's try a systematic approach: 
    # For 3x3x3, we know rank 23 works. For 3x3x4, minimum is at least 24 (since 3x3x3 needs 23)
    # Upper bound: 36 (schoolbook) or 23+12=35 (do 3x3x3 optimally + 3x3x1 schoolbook)
    # Actually, we can do better by sharing computations.
    
    # Let me try to construct a rank-28 or lower decomposition
    # by extending Laderman's algorithm
    
    # Laderman's algorithm for 3x3x3 (23 terms):
    # Let A = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
    # Let B = [b11 b12 b13; b21 b22 b23; b31 b32 b33]
    # C = A*B
    
    # The 23 products are (using 1-indexed):
    # p1 = a11*(b11+b22+b33)
    # p2 = a22*(b11+b22+b33)  
    # p3 = a33*(b11+b22+b33)
    # p4 = a12*(b21+b31)
    # p5 = a23*(b13+b33)
    # p6 = a31*(b12+b22)
    # p7 = (a11+a22)*(b22+b33)
    # p8 = (a11+a33)*(b11+b33)
    # p9 = (a22+a33)*(b11+b22)
    # p10 = (a12+a23)*(b23+b31)
    # p11 = (a23+a31)*(b12+b13)
    # p12 = (a31+a12)*(b21+b32)
    # p13 = (a11+a12+a22+a23)*(b22+b33)  -- I'm not sure about exact form
    
    # Let me use a verified Laderman construction:
    # From literature, the 23-term formula for 3x3 matrix multiplication:
    
    def laderman_3x3x3():
        """Return U, V, W for 3x3x3 matrix multiplication with rank 23."""
        # Using the standard Laderman decomposition
        # U has 9 rows (3*3), V has 9 rows (3*3), W has 9 rows (3*3)
        # Each has 23 columns
        U = [[0]*23 for _ in range(9)]
        V = [[0]*23 for _ in range(9)]
        W = [[0]*23 for _ in range(9)]
        
        # The 23 terms (using 0-indexed: i,j,k for A[i,j], B[j,k], C[i,k])
        # Term 0: a11*(b11+b22+b33)
        # A[0,0]=1, B[0,0]=1, B[1,1]=1, B[2,2]=1
        # C[0,0] gets contribution, C[0,1] gets... wait, this doesn't directly give C entries
        # 
        # The way it works: each term is u_r * v_r * w_r where
        # u_r[i*m+j] = coefficient for A[i,j] in this term
        # v_r[j*p+k] = coefficient for B[j,k] in this term  
        # w_r[k*n+i] = coefficient for C[i,k] in this term
        # And sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i] = 1 for all i,j,k
        
        # Let me use a known correct Laderman decomposition
        # From "A 23-term algorithm for 3x3 matrix multiplication" by Laderman (1976)
        
        # I'll use the following verified 23-term decomposition:
        # Terms (U_col, V_col, W_col) where each is a vector of length 9
        # U_col[i*3+j] for i,j in 0..2
        # V_col[j*3+k] for j,k in 0..2
        # W_col[k*3+i] for k,i in 0..2
        
        terms = [
            # (U_indices: {i*3+j: val}, V_indices: {j*3+k: val}, W_indices: {k*3+i: val})
            # Term 0
            ({0:1}, {0:1,3:1,6:1}, {0:1}),
            # Term 1
            ({3:1}, {0:1,3:1,6:1}, {3:1}),
            # Term 2
            ({6:1}, {0:1,3:1,6:1}, {6:1}),
            # Term 3
            ({1:1}, {2:1,5:1}, {1:1}),
            # Term 4
            ({5:1}, {1:1,7:1}, {5:1}),
            # Term 5
            ({8:1}, {2:1,4:1}, {8:1}),
            # Term 6
            ({0:1,3:1}, {3:1,6:1}, {0:1,3:1}),
            # Term 7
            ({0:1,6:1}, {0:1,6:1}, {0:1,6:1}),
            # Term 8
            ({3:1,6:1}, {0:1,3:1}, {3:1,6:1}),
            # Term 9
            ({1:1,5:1}, {5:1,2:1}, {1:1,5:1}),
            # Term 10
            ({5:1,8:1}, {2:1,1:1}, {5:1,8:1}),
            # Term 11
            ({8:1,1:1}, {3:1,7:1}, {8:1,1:1}),
            # Term 12
            ({0:1,1:1,3:1,5:1}, {3:1,6:1}, {0:1,3:1,6:1}),
            # Term 13
            ({0:1,6:1,3:1,5:1}, {0:1,6:1}, {0:1,6:1,3:1,5:1}),
            # Term 14
            ({3:1,6:1,0:1,1:1}, {0:1,3:1}, {3:1,6:1,0:1,1:1}),
            # Term 15
            ({1:1,5:1,5:1,8:1}, {5:1,2:1}, {1:1,5:1,5:1,8:1}),
            # Term 16
            ({5:1,8:1,8:1,1:1}, {2:1,1:1}, {5:1,8:1,8:1,1:1}),
            # Term 17
            ({8:1,1:1,1:1,0:1}, {3:1,7:1}, {8:1,1:1,1:1,0:1}),
            # Term 18
            ({0:1,1:1,3:1,5:1,6:1}, {3:1,6:1}, {0:1,3:1,6:1}),
            # Term 19
            ({0:1,6:1,3:1,5:1,0:1}, {0:1,6:1}, {0:1,6:1,3:1,5:1}),
            # Term 20
            ({3:1,6:1,0:1,1:1,3:1}, {0:1,3:1}, {3:1,6:1,0:1,1:1}),
            # Term 21
            ({1:1,5:1,5:1,8:1,1:1}, {5:1,2:1}, {1:1,5:1,5:1,8:1}),
            # Term 22
            ({5:1,8:1,8:1,1:1,5:1}, {2:1,1:1}, {5:1,8:1,8:1,1:1}),
        ]
        
        # This is getting complicated and I'm not sure it's correct.
        # Let me use a different, more reliable approach.
        return None
    
    # Let me use a more reliable approach: start from schoolbook and try to reduce rank
    # by finding linear dependencies among the terms
    
    # Actually, let me try a direct construction for 3x3x4
    # Key insight: we can share some computations between the 4 output columns
    
    # For now, let's try a random search approach with verification
    # We'll try ranks from 24 upward and see if we can find a valid decomposition
    
    # Given time constraints, let's use a smarter approach:
    # 1. Start with schoolbook (rank 36)
    # 2. Try to merge pairs of terms that have overlapping support
    # 3. Verify each merge
    
    # Actually, the most promising approach for a structural change:
    # Use the fact that 3x3x4 can be decomposed using the 3x3x3 Laderman algorithm
    # plus additional terms for the 4th column, but with shared computations
    
    # Let me try a specific construction:
    # Split B into B_left (3x3) and B_right (3x1)
    # Compute A*B_left using Laderman (23 terms)
    # Compute A*B_right using 3 terms (schoolbook for 3x3x1)
    # Total: 26 terms
    # But we can potentially share some terms
    
    # For the 3x3x1 part: C[i,3] = sum_j A[i,j]*B[j,3]
    # This needs 9 scalar multiplications in schoolbook, but we can use 3 terms
    # if we structure it as: for each j, term with A[i,j]*B[j,3] contributing to all i
    # Wait, that's not how it works. Each term is a product of linear forms.
    
    # Let me just try to verify if a rank-26 decomposition exists by construction
    
    # Construction for 3x3x4 with rank 26:
    # Use 23 terms for the first 3 columns (Laderman-like) and 3 terms for the 4th column
    # But the 23 terms must only affect columns 0,1,2 of the output
    
    # This is complex. Let me try a different structural approach:
    # Use block decomposition and known results
    
    # For now, let's implement a search that tries to find low-rank decompositions
    # using random initialization and gradient-free optimization
    
    # Given the time budget, let's try a few specific constructions
    
    # Construction 1: Rank 26 (23 + 3)
    # We need a 3x3x3 decomposition of rank 23 that doesn't interfere with column 3
    # and 3 additional terms for column 3
    
    # Actually, let me just try to find ANY valid decomposition with rank < 36
    # by using a random search with local improvement
    
    # For time efficiency, let's try a specific known construction
    # and verify it
    
    # Let me use a simpler approach: try rank values and use random initialization
    # with verification
    
    for target_rank in [26, 27, 28, 29, 30]:
        if time.time() > deadline - 20:
            break
        
        # Try random initialization
        for trial in range(100):
            if time.time() > deadline - 20:
                break
            
            # Random initialization with small values
            U_rand = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n*m)]
            V_rand = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m*p)]
            W_rand = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_decomposition_fast(U_rand, V_rand, W_rand, target_rank):
                best_result = {"U": U_rand, "V": V_rand, "W": W_rand}
                best_rank = target_rank
                break
        
        if best_rank < 36:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
