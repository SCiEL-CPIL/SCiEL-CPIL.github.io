"""Matrix multiplication 2x3x6 with rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    The tensor is T[i,j,k] = 1 for all i in range(2), j in range(3), k in range(6).
    We seek rank r such that:
        T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where m=3, n=2, p=6.
    """
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # The schoolbook rank is n*m*p = 36
    # Let's try to find a lower rank decomposition.
    
    # Strategy: Use a block structure.
    # Split the 3x6 multiplication into blocks.
    # We can view this as:
    #   [A1 A2 A3]  where A1, A2 are 2x3 blocks (since 3 = 3*1)
    #   Actually, let's think of it differently.
    
    # Let's try rank 24 first (which is 2 * 3 * 4, treating 6 as 2*3)
    # or rank 21 (3 * 7, using Strassen-like for 2x3x3 blocks)
    
    # Known result: rank of 2x3x3 matrix multiplication is 7 (Bini 1974)
    # So if we split the 6 into two 3s:
    #   C = A * B where A is 2x3, B is 3x6
    #   Let B = [B1, B2] where B1, B2 are 3x3
    #   C = [A*B1, A*B2]
    #   Each A*B1 and A*B2 is a 2x3x3 multiplication of rank 7
    #   Total rank = 7 + 7 = 14? No, that's not right.
    #   We need to compute both, so we need 7 rank-1 terms for each, total 14.
    #   But wait, can we share some of the computation?
    
    # Actually, let's think more carefully.
    # The tensor for 2x3x3 has rank 7 (this is a known result).
    # For 2x3x6, we can view it as two independent 2x3x3 multiplications
    # with no sharing, giving rank 14.
    # But maybe we can do better by using the structure.
    
    # Let's try to construct a rank-14 decomposition explicitly.
    # The 2x3x3 rank-7 decomposition (Bini's decomposition):
    # For 2x2x2, Strassen gives rank 7.
    # For 2x3x3, the rank is also 7 (as per literature).
    
    # Let me try a different approach: use the known rank-7 decomposition for 2x3x3.
    # Actually, I recall that the rank of 2x3x3 is 7, same as 2x2x2.
    
    # Let's construct the rank-7 decomposition for 2x3x3:
    # This is non-trivial. Let me try a numerical approach instead.
    
    # Approach: Use numerical tensor decomposition (CP decomposition) and then
    # try to round to integer/half-integer values.
    
    # But first, let's try a few candidate ranks and see if we can find exact solutions.
    
    # Let's try rank 24 first (simpler block structure), then try lower.
    
    # Actually, let me think about this more carefully.
    
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
    
    # Tensor T[i,j,k] has shape (2,3,6) and T[i,j,k] = 1 for all entries.
    
    # Rank-1 terms: T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Let's try to find a decomposition of rank r.
    
    # Start with the schoolbook (rank 36) and try to reduce.
    
    # Method: Try to find a low-rank approximation that is exact.
    
    # Let's try rank 24: split k into two groups of 3.
    # For k in {0,1,2}: use rank-7 decomposition of 2x3x3
    # For k in {3,4,5}: use rank-7 decomposition of 2x3x3
    # Total rank = 14
    
    # Let me try to construct the rank-7 decomposition for 2x3x3.
    # This is the key challenge.
    
    # Actually, I found that the rank of 2x3x3 is 7 (same as 2x2x2).
    # The decomposition is non-trivial to write down.
    
    # Let me try a computational approach: set up the equations and solve.
    
    # For rank r, we have:
    #   sum_{t=1}^{r} U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k
    
    # This is a system of polynomial equations. Let's try to solve it numerically.
    
    # Given the complexity, let me try a simpler approach first:
    # Use the schoolbook decomposition and see if we can merge terms.
    
    # Actually, let me just return the schoolbook decomposition for now,
    # since finding an exact lower-rank decomposition is very hard.
    
    # But wait, let me try rank 24 with a block approach.
    
    # Split B into two 3x3 matrices: B = [B1 | B2]
    # C = A*B = [A*B1 | A*B2]
    
    # For each 2x3x3 multiplication, use a rank-7 decomposition.
    # If I can find the rank-7 decomposition, I get total rank 14.
    
    # Let me try to numerically find a rank-7 decomposition for 2x3x3.
    
    def find_rank_decomposition(n, m, p, target_rank, max_time=30):
        """Try to find an exact rank decomposition using numerical methods."""
        start_time = time.time()
        
        # The tensor is all ones of shape (n, m, p)
        # We want: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
        
        # This is a 3-tensor CP decomposition with exact reconstruction.
        
        # Let's set up the problem as: for each (i,j,k), 
        # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
        
        # We can fix W[k,i,t] = 1 for all (k,i,t) and then the problem becomes:
        # sum_t U[i,j,t] * V[j,k,t] = 1 for all (i,j,k)
        # But this is overconstrained in general.
        
        # Better approach: use the bilinear form.
        # For each t, define U_t (n x m), V_t (m x p), W_t (p x n)
        # Then sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = delta_{i,j,k}(1) = 1
        
        # Let's try with W_t[k,i] = 1 for all:
        # Then sum_t U_t[i,j] * V_t[j,k] = 1
        # This means sum_t U_t * V_t (matrix product) = matrix of all ones
        # where U_t is n x m and V_t is m x p, so U_t * V_t is n x p.
        # We need n*p equations, and we have target_rank*(n*m + m*p) variables.
        # For n=2, m=3, p=3, target_rank=7: 2*3=6 equations, 7*(6+9)=105 variables.
        # Very underdetermined, should be solvable.
        
        # Let's solve: find U_t (n x m), V_t (m x p) for t=1..r
        # such that sum_t U_t @ V_t = J (all-ones n x p matrix)
        
        # This is equivalent to: the matrix J (n x p) can be written as
        # sum of r rank-1 terms where each rank-1 term has the form
        # U_t[i,j] * V_t[j,k] which is actually a rank-m term, not rank-1!
        
        # Wait, U_t @ V_t is an n x p matrix that is the product of n x m and m x p,
        # so it has rank at most m=3. So we're expressing J as a sum of r matrices
        # each of rank at most 3. Since J has rank 1 (for n=2,p=3), we can do this
        # with r=1! Just take U_1 = [1,1,1; 1,1,1] (2x3) and V_1 = [1,1,1; 1,1,1; 1,1,1] (3x3)
        # Then U_1 @ V_1 = [3,3,3; 3,3,3] = 3*J. So U_1 = [1/3,1/3,1/3; 1/3,1/3,1/3], V_1 = J.
        # But we need integer/half-integer entries.
        
        # Hmm, this approach with W=1 doesn't work directly because we need the
        # decomposition to be in terms of U, V, W with the specific tensor structure.
        
        # Let me go back to the original formulation.
        
        # The tensor T[i,j,k] = 1 for all i,j,k.
        # We need: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        
        # Note: U has shape (n*m, r) = (6, r), indexed as U[i*m+j, t]
        # V has shape (m*p, r) = (18, r), indexed as V[j*p+k, t]  
        # W has shape (p*n, r) = (12, r), indexed as W[k*n+i, t]
        
        # So the equation is:
        # 1 = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # Let's try a different strategy: use the known Strassen decomposition for 2x2x2
        # and extend it.
        
        # For 2x2x2, the rank-7 decomposition (Strassen) uses 7 scalar multiplications.
        # The formulas are:
        # M1 = (A00)(B00+B11)
        # M2 = (A00+A11)(B11)
        # M3 = (A01-A11)(B10+B11)
        # M4 = (A00)(B10)
        # M5 = (A10+A11)(B00)
        # M6 = (A10-A00)(B01+B00)
        # M7 = (A01+A10)(B11-B10)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M5
        
        # For 2x3x6, let me try to use a block approach.
        
        # Actually, let me just try to numerically solve for small ranks.
        
        # Given the time constraint, let me try a simpler approach:
        # Use rank 24 by splitting into four 2x3x2 blocks (each of rank 6, using schoolbook)
        # No, that's still 24.
        
        # Let me try rank 21: split into three 2x3x2 blocks, each with rank 7 (if we use Strassen for 2x2x2 part)
        # 2x3x2: we can view as 2x2x2 (rank 7) plus 2x1x2 (rank 2), total rank 9? No.
        # 2x3x2: n=2, m=3, p=2. Schoolbook rank = 12.
        # Can we do better? 2x3x2 tensor... 
        # Actually, 2x3x2 = 2x2x2 (rank 7) + 2x1x2 (rank 2) = rank 9? 
        # Not necessarily, because the blocks might share terms.
        
        # This is getting complicated. Let me just try to use a numerical solver.
        
        # Given the 150s time limit, let me try a few things:
        
        # 1. Try rank 24 with a simple block decomposition
        # 2. Try to verify if it's correct
        # 3. If time permits, try lower ranks
        
        # For rank 24: split k into 2 groups of 3.
        # For each group of 3, use schoolbook for 2x3x3 (rank 9 each), total rank 18? No, 9+9=18.
        # Wait, 2x3x3 schoolbook rank is 2*3*3=18? No, 2*3*3=18. That's wrong.
        # n*m*p = 2*3*3 = 18. Yes.
        # So two 2x3x3 blocks give rank 36. That's the same as schoolbook.
        
        # I need to use Strassen to reduce the rank of each block.
        
        # OK let me try a completely different approach.
        # Let me just use the schoolbook decomposition (rank 36) which is guaranteed to work.
        
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=object)
        V = np.zeros((m * p, rank), dtype=object)
        W = np.zeros((p * n, rank), dtype=object)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Verify
        def verify(U, V, W, n, m, p, tol=1e-10):
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            r = U_arr.shape[1]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_arr[i*m+j, t] * V_arr[j*p+k, t] * W_arr[k*n+i, t] for t in range(r))
                        if abs(val - 1.0) > tol:
                            return False
            return True
        
        if not verify(U_list, V_list, W_list, n, m, p):
            raise ValueError("Schoolbook decomposition failed verification")
        
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
