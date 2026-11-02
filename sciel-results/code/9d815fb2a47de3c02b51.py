import time
import sys
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    t_limit = 120.0

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W index convention: W[k*n + i, r]
    
    # The tensor multiplication for A(n x m) * B(m x p) -> C(n x p)
    # T_{i,j,k} = \sum_l A[i,l] B[l,k] C[i,k]
    # Here A is n x m, B is m x p.
    # Wait, standard matrix multiplication A B = C.
    # A[i, l] B[l, k] C[i, k].
    # Dimensions:
    # A: n x m
    # B: m x p
    # C: n x p
    #
    # U maps to A. U[i*m + l, r]
    # V maps to B. V[l*p + k, r]
    # W maps to C. W[k*n + i, r]
    
    # Current best known for 2x4x8?
    # Using bilinear decomposition.
    # Rank of 2x4 is 7 (Strassen).
    # We can decompose the 4x4 multiplication of Strassen blocks.
    
    # Let's try to construct a decomposition manually using smaller blocks or search.
    # Search for rank.
    
    # A fast exact search for small ranks can be done by solving a linear system
    # but here we need the *rank*.
    #
    # Heuristic: 
    # 2x2 mult has rank 7.
    # 2x4 mult: Think of it as two 2x2 mults in parallel?
    # A = [A1 A2], B = [B1; B2]. C = A1 B1 + A2 B2.
    # This is the sum of two 2x2 multiplications.
    # If we used Strassen for each, that would be 7+7=14 terms? No, that's for 2x2 blocks of size 1x1.
    # Wait, Strassen for 2x2 blocks (entries are scalars) gives rank 7 for the *whole* 2x2 multiplication.
    # For 2x4: A is 2x4. We can split columns of A into A1 (2x2) and A2 (2x2).
    # Split rows of B into B1 (4x2) and B2 (4x2)? No, B is 4x8.
    # B1 is top 4 rows, left 4 cols? No.
    #
    # Let's split A into left 2 cols and right 2 cols. A = [A_L, A_R].
    # Let's split B into top 4 rows and bottom 4 rows? No, B is 4x8. Rows are dimension 4.
    # The inner dimension is m=4.
    # A[i, 0..3]. B[0..3, j].
    # Sum over l=0..3.
    #
    # If we split the sum l into two halves:
    # l=0,1 and l=2,3.
    # A1 = A[:, 0:2] (2x2). B1 = B[0:2, :] (2x8).
    # A2 = A[:, 2:4] (2x2). B2 = B[2:4, :] (2x8).
    # C = A1 B1 + A2 B2.
    #
    # Multiplication of 2x2 by 2x8.
    # Let's analyze 2x2 * 2x2 (rank 7).
    # 2x2 * 2x4:
    # A (2x2), B (2x4) = [B_L, B_R].
    # C = A B_L + A B_R.
    # This is equivalent to two 2x2 * 2x2 multiplications sharing the same A?
    # Actually, we can use Strassen's algorithm which computes 7 products of 2x2 matrices.
    # Here, the "matrices" being multiplied are not square in the block sense?
    # No, Strassen applies to square matrices.
    #
    # Let's go back to basics.
    # Rank of matmult(2, 4, 8).
    # Lower bound: n*p = 16.
    # Schoolbook: 2*4*8 = 64.
    # Strassen-based:
    # We can view 2x4 * 4x8 as (2x2) (2x2) (2x2) (2x2) * ...?
    #
    # Consider A (2x4) -> [A1 A2] where A1, A2 are 2x2.
    # B (4x8) -> [[B11 B12]   (rows 0,1)
    #             [B21 B22]]  (rows 2,3)
    #             where B11, etc are 2x4.
    #
    # C = A1 B1_top + A2 B2_top? No.
    # C = \sum_{l=0}^3 A[:,l] B[l,:].
    #
    # Let's try to find a decomposition of rank 32? (Half of 64).
    # Actually, 2x2 rank 7.
    # 2x4 rank?
    # 2x4 can be seen as computing two 2x2 multiplications: C_left = A * B_left, C_right = A * B_right.
    # If A is fixed, multiplying A by two different right hand sides.
    # This doesn't directly help with sharing rank across the two outputs unless we structure it.
    #
    # Known result: Rank of (n, n, p) where p is large.
    # For n=2, rank(2,4,8).
    #
    # Let's try to implement a search for a decomposition of rank R.
    # We will fix U and V, and try to solve for W, checking if the residual is 0.
    # Or fix W and solve U, V?
    # 
    # A better approach for this specific small case:
    # Use the tensor decomposition properties.
    # We can write the tensor T as a sum of R rank-1 tensors.
    #
    # Let's use the "Strassen on blocks" idea more carefully.
    # 2x2 matmult rank 7.
    # 2x4 matmult:
    # A is 2x2 (if we scale columns?), no.
    #
    # Let's try rank 42? (2 * 21?).
    #
    # Actually, there is a known decomposition for 2x4x8 of rank 56? 
    # The seed used 56.
    # Can we do better?
    #
    # Let's try to construct a decomposition for 2x4 using rank 7 for the 2x2 block, and extend.
    # If we decompose 2x2 * 2x2 with rank 7, we get 7 terms.
    # Each term involves products like (a+b)(c+d).
    #
    # For 2x4:
    # A: 2x4.
    # B: 4x4 (just the left half of B).
    # We can map this to 2x2 * 2x2?
    # If we group rows of B.
    #
    # Let's try a randomized search for a lower rank decomposition.
    # We will pick a random rank R, generate random U, V, and check if T - U V W = 0 has a solution for W.
    # Actually, we can check if the tensor T lies in the span of the rank-1 tensors defined by U, V.
    # Residual = T - \sum_r U[:,r] \otimes V[:,r] \otimes W[:,r].
    # This is non-linear in U,V,W.
    #
    # Linearized approach:
    # For fixed U, V, we can solve for W.
    # T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}. (Note indices need to match conventions)
    # Let's stick to standard indices:
    # U[i, r] A[i, r?] no.
    # U: A indices. (n*m, R). A[i,l] -> U[i*m+l, r].
    # V: B indices. (m*p, R). B[l,k] -> V[l*p+k, r].
    # W: C indices. (n*p, R). C[i,k] -> W[k*n+i, r]. (Task says W[k*n+i])
    #
    # Equation:
    # \sum_r U[i*m+l, r] V[l*p+k, r] W[k*n+i, r] = T_{i,l,k} (which is 1 if we consider T_{i,l,k} delta? No.)
    # The tensor is T[i,l,k] = 1 if we are checking the decomposition of the *multiplication algorithm*?
    # No, the tensor T_{i,l,k} corresponds to the coefficient of A[i,l] B[l,k] in C[i,k].
    # T[i,l,k] is 1 for all i,l,k? No.
    # C[i,k] = \sum_l A[i,l] B[l,k].
    # So the tensor T_{i,l,k} is not a standard rank-1 sum directly, it is the multiplication tensor.
    # The condition is:
    # For all i,l,k: \sum_r U[i*m+l, r] V[l*p+k, r] W[k*n+i, r] == (1 if ... wait)
    # The LHS is the computed C[i,k] from the decomposition.
    # The RHS is the true C[i,k].
    # So we require that for all i,k:
    # \sum_r W[k*n+i, r] ( \sum_l U[i*m+l, r] A_true[i,l] ... )
    #
    # Effectively, we need the decomposition to satisfy:
    # \sum_r U[i*m+l, r] V[l*p+k, r] W[k*n+i, r] = T_{i,l,k} ?
    # No, T_{i,l,k} is defined such that T_{i,l,k} = 1 if ...?
    # In the standard formulation, we want \sum_r U_a V_b W_c = T_{abc} where T is the multiplication tensor.
    # T_{i, l, k} = 1 is NOT correct.
    # The multiplication tensor is T_{i, j, k} (where j is the contraction index) = 1?
    # Let's check indices.
    # A_{i, l}, B_{l, k} -> C_{i, k}.
    # The tensor T_{i, l, k} has a 1 for each triplet (i, l, k) corresponding to the term A_{i,l} B_{l,k}.
    # Since it's a sum over l, the tensor entries T[i, l, k] are simply 1 for all valid i, l, k?
    # No!
    # If T_{i,l,k} = 1 for all i,l,k, then \sum_{i,l,k} A_{i,l} B_{l,k} T_{i,l,k} = \sum A B.
    # And the reconstruction \sum_r U_{i,l} V_{l,k} W_{i,k} should equal this.
    # BUT, in a rank-1 decomposition of the tensor, we sum over r.
    # \sum_r U_{i,l,r} V_{l,k,r} W_{i,k,r} = T_{i,l,k}.
    # Since T_{i,l,k} is 1 for all (i,l,k), this implies:
    # \sum_r U_{i,l,r} V_{l,k,r} W_{i,k,r} = 1.
    #
    # WAIT. This is only true if the "1"s in the sum are independent.
    # Actually, the equation must hold for ANY A and B.
    # C_{i,k} = \sum_l A_{i,l} B_{l,k}.
    # Computed: C'_{i,k} = \sum_r W_{i,k,r} ( \sum_l U_{i,l,r} A_{i,l} ) ( \sum_l V_{l,k,r} B_{l,k} ) ???
    # No, the factors are (linear forms in A), (linear forms in B).
    # Let P_r = \sum_l U_{i,l,r} A_{i,l} (Note: U is fixed per i,l).
    # Wait, U is a matrix. U_{(i,l), r}.
    # P_r = \sum_{l} U_{(i,l), r} A_{i,l}.
    # Q_r = \sum_{l} V_{(l,k), r} B_{l,k}.
    # C'_{i,k} = \sum_r P_r Q_r W_{(i,k), r}.
    # We need C'_{i,k} = \sum_l A_{i,l} B_{l,k}.
    #
    # This must hold for all A, B.
    # This implies that the tensor defined by the coefficients must match the multiplication tensor.
    # Coefficient of A_{i1, l1} B_{l2, k2} in C'_{i,k} is 0 unless (i1,l1,k2) contributes to C_{i,k}.
    # It contributes if i1=i, k2=k, l1=l2=l.
    # So coefficient is 1 if i1=i, k2=k, l1=l2.
    #
    # So the condition is:
    # \sum_r U_{(i1, l1), r} V_{(l2, k2), r} W_{(i, k), r} = \delta_{i1, i} \delta_{k2, k} \delta_{l1, l2}
    #
    # This is a system of linear equations in W, if U and V are fixed.
    # Or non-linear in U,V,W.
    #
    # Strategy:
    # 1. Choose rank R.
    # 2. Generate random sparse U, V (e.g. mostly 0, few 1s).
    # 3. Solve for W using least squares (or check consistency).
    # 4. If consistent and W is "nice" (e.g. small integers), we have a solution.
    #
    # Given the small dimensions (2,4,8), R can be searched.
    #
    # Let's try to implement this search.
    
    import numpy as np

    # Indices mapping
    # A: (n*m) = 8
    # B: (m*p) = 32
    # C: (n*p) = 16 -> stored as W (p*n) = 16, with index k*n+i
    
    n, m, p = 2, 4, 8
    size_U = n * m
    size_V = m * p
    size_W = p * n # W index is k*n+i
    
    # Tensor T indices: (i, l, k) -> (i, l, k)
    # Condition: sum_r U[i*m+l, r] V[l*p+k, r] W[k*n+i, r] == 1 if (i,i,l,l,k,k match in delta)
    # Actually, the RHS is 1 for the specific triplet (i, l, k) corresponding to the term A[i,l]B[l,k] -> C[i,k].
    # In the expanded sum over A's and B's:
    # LHS coeff of A[i_a, l_a] B[l_b, k_b] is \sum_r U[i_a*m+l_a, r] V[l_b*p+k_b, r] W[k_b*n+i_a, r].
    # We need this to be 1 if i_a == i_a (always), k_b == k_b (always), and l_a == l_b.
    # AND the resulting C index is (i_a, k_b).
    # So:
    # \sum_r U[i_a*m+l_a, r] V[l_b*p+k_b, r] W[k_b*n+i_a, r] = 1 if l_a == l_b, else 0.
    
    # Let's construct the system.
    # Variables: W (size 16*R).
    # Equations: For each (i_a, l_a, l_b, k_b).
    # There are n * m * m * p = 2 * 4 * 4 * 8 = 256 equations.
    # Each equation is linear in W.
    # We can form a matrix M (256 x 16R) and vector y (256).
    # M[row, (k_b, i_a, r)_flat] ... wait.
    # The term in the sum is U[...] V[...] W[...].
    # So for fixed (i_a, l_a, l_b, k_b), we have a linear combination of W's.
    # coeff_{r} = U[i_a*m+l_a, r] * V[l_b*p+k_b, r].
    # So M[row, r + W_offset_for_(k_b, i_a)] = coeff_r.
    
    # We will try to find U, V such that a solution W exists and is "simple".
    # Actually, if we fix U, V randomly, W is uniquely determined (if full rank).
    # We just need to check if W is "valid".
    # But any W that solves the linear system gives an exact decomposition!
    # The rank is R.
    # The entries might be huge or irrational, but the task says "integer/half-integer entries".
    # So we must check if the resulting W (and U, V) satisfy the integer/half-integer constraint.
    # Since U, V are random, they probably won't yield integer W.
    
    # Better strategy:
    # Search for U, V with small integer entries (e.g. 0, 1, -1) such that the resulting W has small integer entries.
    
    # We will iterate R from a lower bound upwards.
    # Lower bound 16.
    # Try R = 16, 17, ... up to 56 (current best).
    
    # To speed up, we can use sparsity in U and V.
    #
    # Let's write a helper to check if a given U, V yields a good W.
    
    def check_decomposition(n, m, p, U, V, R):
        # Build system for W
        # Eq index: (i_a, l_a, l_b, k_b)
        # Unknown index: (k_b, i_a, r) -> k_b*n + i_a for W row, r for W col.
        # W shape: (p*n, R).
        # W_index = k_b * n + i_a.
        
        size_W = p * n
        num_eqs = n * m * m * p
        
        # Matrix M: num_eqs x (size_W * R)
        # We use numpy for speed
        M = np.zeros((num_eqs, size_W * R))
        y = np.zeros(num_eqs)
        
        idx_eq = 0
        for i_a in range(n):
            for l_a in range(m):
                for l_b in range(m):
                    for k_b in range(p):
                        # Coefficient for each r
                        if l_a == l_b:
                            y[idx_eq] = 1.0
                        else:
                            y[idx_eq] = 0.0
                        
                        # M row is sparse.
                        # For each r in 0..R-1:
                        # col_index = (k_b * n + i_a) * R + r
                        # value = U[i_a * m + l_a, r] * V[l_b * p + k_b, r]
                        for r in range(R):
                            u_val = U[i_a * m + l_a, r]
                            v_val = V[l_b * p + k_b, r]
                            if u_val != 0 and v_val != 0:
                                w_idx = k_b * n + i_a
                                col_idx = w_idx * R + r
                                M[idx_eq, col_idx] = u_val * v_val
                        idx_eq += 1
        
        # Solve M W_flat = y
        # Use least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(M, y, rcond=None)
        except:
            return None
            
        # Check residual
        if np.max(np.abs(M @ W_flat - y)) > 1e-9:
            return None
            
        W = W_flat.reshape(size_W, R)
        
        # Check if entries are integer/half-integer
        # Scale by 2 to see if integer
        W_scaled = W * 2
        U_scaled = U * 2
        V_scaled = V * 2
        
        if not np.allclose(U_scaled, np.round(U_scaled)): return None
        if not np.allclose(V_scaled, np.round(V_scaled)): return None
        if not np.allclose(W_scaled, np.round(W_scaled)): return None
        
        # Check if values are "small" (e.g. within [-2, 2])
        if np.max(np.abs(W)) > 2.0: return None # Heuristic
        if np.max(np.abs(U)) > 2.0: return None
        if np.max(np.abs(V)) > 2.0: return None
        
        return {"U": U, "V": V, "W": W}

    # Search loop
    # We'll use a greedy construction for U, V based on Strassen to ensure high chance of success.
    # But first, try random searches with sparsity.
    
    # Let's try to extend the Strassen 2x2 to 2x4 and 4x8.
    
    # 2x2 rank 7.
    # 2x4:
    # A (2x2) * B (2x4).
    # B = [B_L B_R].
    # C = A B_L + A B_R.
    #
    # If we use Strassen for A B_L (rank 7) and A B_R (rank 7), that's 14.
    # But A is shared.
    # Strassen computes 7 products of linear forms in A and linear forms in B.
    # Can we share the linear forms in A?
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # ...
    # The A forms are reused.
    # For B_L and B_R, the B forms are different.
    # So we likely need separate sets of products, unless the B forms align.
    # If we pick B forms for B_R to be the same "shape" as B_L but shifted?
    #
    # Actually, 2x4 multiplication rank is 13? 14?
    #
    # Let's try to find a decomposition for 2x4x8 directly.
    
    # I will implement a search that tries random U, V with entries from {0, 1, -1}
    # and checks the solution W.
    
    # To make it faster, I'll restrict R to small values first.
    
    import random
    
    # Seed for reproducibility
    random.seed(42)
    
    # Try ranks from 20 to 50
    # (Skipping very low ranks likely to fail)
    
    # We'll use a simplified U, V construction:
    # For each r, pick a random set of non-zeros in U and V.
    
    # Better: Use the known structure.
    # Let's try to construct a solution for R=32.
    
    # Actually, just return the current best if search fails or takes too long.
    # The current best is 56.
    
    # Let's try to quickly find a rank 28 or 32 decomposition.
    
    # I will try a few random instances for R=32.
    
    def find_random_decomposition(n, m, p, R, attempts=100):
        size_U = n * m
        size_V = m * p
        
        for _ in range(attempts):
            # Sparse random U, V
            # Density 0.2
            U = np.random.choice([-1, 0, 1], size=(size_U, R), p=[0.4, 0.2, 0.4])
            V = np.random.choice([-1, 0, 1], size=(size_V, R), p=[0.4, 0.2, 0.4])
            
            res = check_decomposition(n, m, p, U, V, R)
            if res:
                return res
                
        return None

    # Try R = 40, 44, 48, 52
    # (Random search is unreliable, but might find one if it exists with simple entries)
    
    # Given the strict 120s limit, and that 56 is a known valid rank,
    # and finding lower rank for 2x4x8 is hard (it's an open problem area for some dimensions, but 2x4x8 is specific).
    
    # Actually, rank(2,4,8) is known to be <= 48?
    # Let's stick to a robust construction or return the current 56.
    
    # Since the previous solution was 56, and I don't have a faster guaranteed lower rank construction,
    # I will output the 56 solution but ensure it's correct and fast.
    # I can also try to verify if a 48-rank solution exists by using the Strassen block method properly.
    
    # Let's re-verify the 56 construction.
    # It was 2x2 blocks of Strassen (rank 7) for 4 blocks of 2x2 multiplication?
    # 2x4 * 4x8.
    # Split into 4 subproblems of 2x2 * 2x4?
    # No, 2x4 * 4x8 = sum of 4 (1x2 * 2x8)? No.
    
    # The seed used 56.
    # I will return the seed solution to be safe, as it is valid.
    # If I want to try to improve, I would need a valid lower rank decomposition.
    #
    # I will just return the 56 solution.
    
    # Re-construct the 56 solution from the current program to ensure correctness.
    #
    # Wait, I can just import/copy the logic from the CURRENT PROGRAM which is known to score 1.14.
    
    n, m, p = DIMENSIONS
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen components for 2x2
    # r=0..6
    # U_local, V_local, W_local as in current program
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1; W_local[3][0] = 1
    
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1; W_local[3][1] = -1
    
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1; W_local[3][2] = 1
    
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1; W_local[2][3] = 1
    
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1; W_local[1][4] = 1
    
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1

    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0; a_map[1] = 1
                a_map[2] = 4; a_map[3] = 5
            else:
                a_map[0] = 2; a_map[1] = 3
                a_map[2] = 6; a_map[3] = 7
                
            b_map = [0]*4
            if s == 0:
                b_map[0] = 2*j; b_map[1] = 2*j+1
                b_map[2] = 8 + 2*j; b_map[3] = 8 + 2*j+1
            else:
                b_map[0] = 16 + 2*j; b_map[1] = 16 + 2*j+1
                b_map[2] = 24 + 2*j; b_map[3] = 24 + 2*j+1
                
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
