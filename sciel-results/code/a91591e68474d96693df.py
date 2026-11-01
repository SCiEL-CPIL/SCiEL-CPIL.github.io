"""Matrix multiplication 2x2x6 — rank-minimal exact tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 6
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)

    # The multiplication tensor T[i,j,k] = A[i,j]*B[j,k] for i in 0..1, j in 0..1, k in 0..5
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]

    # Strategy: Try to find a low-rank decomposition.
    # Known bounds: for 2x2x2, Strassen gives rank 7.
    # For 2x2x6, we can think of it as three 2x2x2 blocks in the k dimension.
    # But we can do better by sharing structure.

    # Let's try a direct search approach:
    # The tensor has 2*2*6 = 24 entries.
    # A rank-r decomposition has 4r + 12r + 12r = 28r parameters (but with scaling equivalence).
    # We need rank such that 28r >= 24, so r >= 1. But structure matters.

    # Approach: Use the fact that for each k, we have a 2x2 matrix multiplication.
    # The 2x2x2 tensor can be done in rank 7 (Strassen).
    # For 6 values of k, if independent, rank 42. But we can share.

    # Better approach: Think of B as a 2x6 matrix. The product C = A*B is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k].

    # Let's use a parameterized approach:
    # For each rank component c, U has 4 entries, V has 12 entries, W has 12 entries.
    # The constraint is: for all (i,j,k):
    #   sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta(j fixed) * ... 
    # Actually, the tensor T_{i,j,k} = 1 for all (i,j,k) in the sense that
    # T[i,j,k] * A[i,j] * B[j,k] sums to C[i,k].
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) ... 
    # No, that's not right either.

    # Correct formulation:
    # We want: for all i in {0,1}, k in {0,...,5}:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) ... 
    # No, that's wrong. The correct tensor decomposition is:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if the (i,j,k) entry of the multiplication tensor is 1.
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,5}.

    # This is a system of 2*2*6 = 24 equations.

    # Let's try to find a rank-r solution with small r.
    # For rank 7, we have 7*28 = 196 variables (with scaling, effectively 196 - 24 = 172 degrees of freedom after gauge fixing).
    # 24 equations, so rank 7 should be feasible in principle.

    # Let's use a randomized/iterative approach:
    # 1. Try to construct a rank-r decomposition using a greedy/optimization approach.
    # 2. Use the Strassen-like structure as a starting point.

    # Known result: The minimum rank for 2x2x2 is 7 (Strassen).
    # For 2x2x6, we can embed three 2x2x2 problems. But can we do better?
    # The tensor 2x2x6 can be seen as a 2x2 matrix times a 2x6 matrix.
    # The border rank and rank are studied in literature.

    # Let's try a systematic search for small ranks using rational/integer entries.

    # Approach: Fix a structure and solve for the entries.
    # Let's try rank 7, 8, 9, ... and see if we can find an exact solution.

    # Use a simpler approach: exploit the block structure.
    # C[i,k] for k=0,1,2,3,4,5.
    # For each pair (k1, k2), the sub-problem on {k1, k2} is 2x2x2.
    # If we can find a rank-7 decomposition for each 2x2x2 block and combine them...

    # Actually, let's try a different approach: use the fact that the problem is
    # C = A * B where A is 2x2 and B is 2x6.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].

    # We can precompute some linear combinations of B's columns.
    # Let's define for each k: we need A[i,0]*B[0,k] + A[i,1]*B[1,k].

    # A simple approach with rank 8:
    # For each k, use 2 multiplications (schoolbook for the 2x2 in the j dimension).
    # That gives rank 6*2 = 12. Too many.

    # Better: For pairs of k values, use Strassen.
    # 6 values of k can be grouped into 3 pairs.
    # Each pair uses rank 7 (Strassen for 2x2x2).
    # Total rank: 3*7 = 21.

    # Can we do better by overlapping?
    # Let's try rank 14: 2 * 7. Use Strassen for k={0,1,2} and k={3,4,5}?
    # But that's 2x2x3, not 2x2x2.

    # Let's just try to find the minimum rank by search.

    # For small ranks, we can set up the equations and try to solve them.
    # Use the structure: many entries of U, V, W will be 0, 1, or -1.

    # Let's try a greedy constructive approach for rank 7:
    # We need 7 components. Each component (u_c, v_c, w_c) contributes
    # u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] to T[i,j,k].

    # Let's try to use a known low-rank decomposition.
    # For 2x2x2, Strassen uses:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For 2x2x6, let's extend this.
    # We have B as a 2x6 matrix. Let B = [B0, B1, B2, B3, B4, B5] where Bj is column j (2x1).
    # C = A * B = [A*B0, A*B1, A*B2, A*B3, A*B4, A*B5].

    # For each column, we need a 2x2 * 2x1 multiplication.
    # That's simpler than 2x2x2.

    # 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1].
    # This can be done with rank 3:
    # M1 = A[0,0]*B[0]
    # M2 = A[1,1]*B[1]
    # M3 = (A[0,0]+A[1,1])*(B[0]+B[1])
    # C[0] = M1 + M3 - M2 - A[1,1]*B[0]... hmm, this doesn't work directly.

    # Actually for 2x2 * 2x1, the tensor has 2*2*1 = 4 entries.
    # Minimum rank is 3 (since 2x2x1 is a matrix, and the multiplication tensor has rank 3).
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This is bilinear in (A,B) -> C.
    # Rank 3: 
    #   M1 = (A00+A01)*B0  -> contributes A00*B0 + A01*B0
    #   M2 = (A10+A11)*B1  -> contributes A10*B1 + A11*B1  
    #   M3 = A00*B0        -> need to subtract A01*B0 and add A01*B1...
    # This is getting complicated.

    # Let me just try a computational search for small ranks.

    # For rank r, we have r components. Let's try to find entries that are in {-1, 0, 1}.
    # The constraints are:
    # For each (i, j, k) with i in {0,1}, j in {0,1}, k in {0..5}:
    #   sum_c U[i*2+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1

    # This is 24 equations.

    # Let's try rank 7 first. If we can't find it, try 8, 9, etc.

    def try_rank(r):
        """Try to find a rank-r decomposition with entries in {-1, 0, 1}."""
        if time.time() > deadline:
            return None

        # We'll use a backtracking search with pruning.
        # Variables: U (4*r), V (12*r), W (12*r) with entries in {-1, 0, 1}.
        # That's too many variables for exhaustive search.

        # Instead, let's use a structured approach.
        # Fix U to have a specific pattern and solve for V and W.

        # Alternative: use the fact that each column of (U, V, W) defines a rank-1 tensor.
        # We need the sum of r rank-1 tensors to equal the multiplication tensor.

        # Let's try a randomized approach:
        # 1. Generate random sparse U, V, W with entries in {-1, 0, 1}.
        # 2. Check if the tensor equation is satisfied.
        # 3. If not, adjust.

        # Better: use linear algebra.
        # For a fixed U, the equation becomes linear in V*W (outer product).
        # Let's fix U and solve for V, W.

        # For fixed U, define for each (i,j,k):
        #   sum_c V[j*p+k, c] * W[k*n+i, c] = 1 / U[i*m+j, c]  (if U[i*m+j,c] != 0)
        # This is nonlinear.

        # Let's try a different approach: use known results.
        # The rank of the 2x2x6 matrix multiplication tensor.
        # From literature: R(2x2x6) <= 20 (by various constructions).
        # Let's try to construct one with rank 20.

        # Actually, let's just return a valid decomposition with a known rank.
        # Schoolbook: rank 24.
        # Strassen-based: rank 21 (3 * 7).
        # Let's try to find something better.

        # For now, let's return the rank-21 Strassen-based decomposition.
        return None  # Will be replaced

    # Let's construct the rank-21 decomposition (3 Strassen blocks).
    # Block b covers k = 2b, 2b+1 for b = 0, 1, 2.

    # Strassen 2x2x2 decomposition:
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # For 2x2x2: U[i,j,c], V[j,k,c], W[k,i,c]
    # i,j,k in {0,1}

    # Strassen components:
    # c=0: M1 = (A00+A11)*(B00+B11)
    #   U: U[0,0]=1, U[3,3]=1  (i.e., U[0,0] for (0,0), U[3] for (1,1))
    #   Wait, let me use the standard formulation.
    #
    # Standard Strassen for C = AB, A,B 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7

    # In tensor notation: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # where the result is C[i,k] = sum_j A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k (the multiplication tensor).
    # And C[i,k] = sum_j A[i,j] * B[j,k] * (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] / 1)
    # Wait, I'm confusing myself.

    # The correct statement: We want to compute C = A*B.
    # C[i,k] = sum_j A[i,j]*B[j,k].
    # A rank-r algorithm computes:
    #   For c in 0..r-1:
    #     X_c = linear_combination_of_A (using U_c)
    #     Y_c = linear_combination_of_B (using V_c)
    #     M_c = X_c * Y_c
    #   C[i,k] = sum_c W_c[i,k] * M_c
    #
    # In tensor form: the multiplication tensor is T[i,j,k] = e_i e_j e_k^T (summed appropriately).
    # Actually, the standard tensor decomposition is:
    #   T = sum_c u_c ⊗ v_c ⊗ w_c
    # where T[i,j,k] = sum_c u_c[i] v_c[j] w_c[k] = 1 if (i,j,k) is a valid index for multiplication.
    # For matrix multiplication, T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # And the algorithm computes C[i,k] = sum_j A[i,j] B[j,k] by:
    #   For each c: X_c = sum_j U[j,c] A[?], etc.

    # I think I'm overcomplicating this. Let me just use the direct formulation from the task:
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (12, rank)
    # And the condition is:
    #   For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    #     sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # Let me just hard-code the Strassen-based rank-21 solution.

    # Strassen 2x2x2 U, V, W:
    # Using the standard Strassen:
    # U_strassen[i*2+j, c] for i,j in {0,1}, c in 0..6
    # V_strassen[j*2+k, c] for j,k in {0,1}, c in 0..6
    # W_strassen[k*2+i, c] for k,i in {0,1}, c in 0..6

    # c=0 (M1): X = A00+A11, Y = B00+B11
    #   U: U[0,0]=1 (A00), U[3,0]=1 (A11)  [using i*2+j: (0,0)->0, (1,1)->3]
    #   V: V[0,0]=1 (B00), V[3,0]=1 (B11)  [using j*2+k: (0,0)->0, (1,1)->3]
    #   W: W[0,0]=1 (C00 += M1), W[3,0]=1 (C11 += M1)  [using k*2+i: (0,0)->0, (1,1)->3]
    #
    # c=1 (M2): X = A10+A11, Y = B00
    #   U: U[2,1]=1 (A10), U[3,1]=1 (A11)  [(1,0)->2, (1,1)->3]
    #   V: V[0,1]=1 (B00)
    #   W: W[2,1]=1 (C10 += M2)  [(1,0)->2]
    #
    # c=2 (M3): X = A00, Y = B10-B11
    #   U: U[0,2]=1 (A00)
    #   V: V[2,2]=1 (B10), V[3,2]=-1 (B11)  [(1,0)->2, (1,1)->3]
    #   W: W[2,2]=1 (C10 += M3), W[3,2]=1 (C11 += M3)  [(1,0)->2, (1,1)->3]
    #   Wait, M3 = a00(b10-b11), contributes to c01 and c11.
    #   c01 = M3 + M5, c11 = M1 - M2 + M3 + M7
    #   So W[1,2]=1 (C01), W[3,2]=1 (C11)  [(0,1)->1, (1,1)->3]
    #
    # Let me redo this more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11

    # Strassen:
    # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a10+a11)b00 = a10*b00 + a11*b00
    # M3 = a00(b10-b11) = a00*b10 - a00*b11
    # M4 = a11(b01-b10) = a11*b01 - a11*b10
    # M5 = (a00+a01)b11 = a00*b11 + a01*b11
    # M6 = (a10-a00)(b00+b10) = a10*b00 + a10*b10 - a00*b00 - a00*b10
    # M7 = (a01-a11)(b01+b11) = a01*b01 + a01*b11 - a11*b01 - a11*b11

    # c00 = M1 + M4 - M5 - M7
    #   = (a00*b00 + a00*b11 + a11*b00 + a11*b11) 
    #     + (a11*b01 - a11*b10)
    #     - (a00*b11 + a01*b11)
    #     - (a01*b01 + a01*b11 - a11*b01 - a11*b11)
    #   = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b10 - a00*b11 - a01*b11 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    #   = a00*b00 + a11*b00 + 2*a11*b01 - a11*b10 - 2*a01*b11 - a01*b01 + 2*a11*b11
    #   This doesn't simplify to c00. I must be using the wrong Strassen formula.

    # Let me use the correct Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b10)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7

    # Let me verify c00:
    # M1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M4 = a11*b01 - a11*b10
    # M5 = a00*b11 + a01*b11
    # M7 = a01*b01 + a01*b11 - a11*b01 - a11*b11
    # M1+M4-M5-M7 = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b10 - a00*b11 - a01*b11 - a01*b01 - a01*b11 + a11*b01 + a11*b11
    # = a00*b00 + a11*b00 + 2a11*b01 - a11*b10 - 2a01*b11 - a01*b01 + 2a11*b11
    # This is not c00 = a00*b00 + a01*b10.

    # I think the issue is that in the tensor decomposition for matrix multiplication,
    # the W matrix encodes how the products M_c contribute to C.
    # Let me just look up the standard tensor decomposition for Strassen.

    # Actually, for the purpose of this task, let me just use a simple valid decomposition.
    # Schoolbook rank 24 is valid. Let me try to improve it slightly.

    # For 2x2x6, we can use the following:
    # For each k in 0..5, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is a 2x2 * 2x1 multiplication for each k.
    # The 2x2 * 2x1 tensor has rank 3.
    # So rank 6*3 = 18? No, that's not right because the k dimension couples.

    # Actually, each k is independent, so we can use rank 3 for each k.
    # Total rank: 6 * 3 = 18.

    # Let's construct the rank-18 decomposition:
    # For each k, use 3 components.
    # 2x2 * 2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1]
    # Rank 3 decomposition:
    #   M1 = A[0,0]*B[0]
    #   M2 = A[1,1]*B[1]
    #   M3 = (A[0,0]+A[1,1])*(B[0]+B[1])
    #   C[0] = M1 + M3 - M2 - A[1,1]*B[0]... 

    # Hmm, let me think about this differently.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]

    # Rank 3:
    # M1 = (A[0,0]+A[1,0]) * B[0]  = A[0,0]*B[0] + A[1,0]*B[0]
    # M2 = A[0,1] * (B[0]+B[1])  = A[0,1]*B[0] + A[0,1]*B[1]
    # M3 = A[1,1] * B[1]  = A[1,1]*B[1]
    # C[0] = M1 + M2 - A[1,0]*B[0] - A[0,1]*B[0]... this is getting messy.

    # Let me just use the schoolbook rank 24 for now, and try to optimize.

    # Actually, let me try a rank-16 approach:
    # For each pair of k values (k1, k2), use a 2x2x2 Strassen with rank 7.
    # 3 pairs -> rank 21.

    # Or for 3 k values at a time, use a 2x2x3 decomposition.
    # 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k] for k in {0,1,2}.
    # This is 6 output entries.
    # Rank of 2x2x3 tensor? It's at most 6 (schoolbook) and at least... 
    # We can do: C = A * B where B is 2x3.
    # Use 3 multiplications for each column -> rank 6.
    # Or use a combined approach.

    # Let me just go with the rank-21 Strassen-based solution and verify it.

    # Strassen 2x2x2 tensor decomposition (standard):
    # I'll use a known correct formulation.

    # For 2x2x2, the Strassen decomposition in tensor form:
    # U (4x7), V (4x7), W (4x7)
    # where U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]

    # Component c=0:
    #   U: U[0,0]=1, U[3,0]=1  (a00 + a11)
    #   V: V[0,0]=1, V[3,0]=1  (b00 + b11)
    #   W: W[0,0]=1, W[3,0]=1  (c00 += M0, c11 += M0)
    #   Contribution to T[i,j,k]:
    #     T[0,0,0] += 1*1*1 = 1 (from i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0])
    #     T[0,0,1] += 1*1*0 = 0 (W[1,0]=0)
    #     T[0,1,0] += 0*1*1 = 0 (U[1,0]=0)
    #     T[0,1,1] += 0*1*0 = 0
    #     T[1,0,0] += 0*1*0 = 0
    #     T[1,0,1] += 0*1*1 = 0
    #     T[1,1,0] += 1*1*0 = 0
    #     T[1,1,1] += 1*1*1 = 1 (from i=1,j=1,k=1: U[3,0]*V[3,0]*W[3,0])
    #   So T[0,0,0] += 1, T[1,1,1] += 1.
    #   But we need T[i,j,k]=1 for all i,j,k. So this single component only covers 2 of 8 entries.

    # I see, the Strassen decomposition works because the W matrix has more entries.
    # Let me just look at it as: the product M_c = X_c * Y_c, and then
    # C[i,k] = sum_c W_c[i,k] * M_c.
    # In tensor form: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]... 

    # I think the correct interpretation is:
    # X_c = sum_{i,j} U_c[i,j] * A[i,j]
    # Y_c = sum_{j,k} V_c[j,k] * B[j,k]
    # C[i,k] = sum_c W_c[i,k] * X_c * Y_c
    # = sum_c W_c[i,k] * (sum_{i',j} U_c[i',j] A[i',j]) * (sum_{j',k'} V_c[j',k'] B[j',k'])
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c W_c[i,k] * U_c[i',j] * V_c[j',k'] = delta(i',i) * delta(k',k) for the relevant terms.
    # This is getting too complicated. Let me just use a computational approach.

    # Let me try to find a valid rank-r decomposition by solving the equations.
    # For small r, we can use a mixed-integer approach.

    # For rank r, we have:
    # For all (i,j,k): sum_c U[i*2+j,c] * V[j*6+k,c] * W[k*2+i,c] = 1

    # Let's try rank 7 with entries in {-1, 0, 1} using a SAT-like search.
    # But 24 equations and 7*28 = 196 variables is too many for exhaustive search.

    # Let me just return the schoolbook rank 24 for now, which is guaranteed correct.
    # And add a time check.

    rank = n * m * p  # 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Now try to improve: attempt rank 21 (3 Strassen blocks).
    # For each block b in {0, 1, 2}, covering k = 2b, 2b+1:
    # Use Strassen 2x2x2 decomposition.

    # Strassen 2x2x2 U, V, W (4x7 each):
    # I'll use a verified formulation.
    # 
    # After careful consideration, here's the Strassen tensor decomposition:
    # For C = AB, 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6

    # In tensor form, U[i,j,c], V[j,k,c], W[k,i,c]:
    # c=0 (M0): U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: (0,0)=1, (1,1)=1
    # c=1 (M1): U: (1,0)=1, (1,1)=1; V: (0,0)=1; W: (1,0)=1
    # c=2 (M2): U: (0,0)=1; V: (1,0)=1, (1,1)=-1; W: (0,1)=1, (1,1)=1
    # c=3 (M3): U: (1,1)=1; V: (0,1)=1, (1,0)=-1; W: (0,0)=1, (1,0)=1
    # c=4 (M4): U: (0,0)=1, (0,1)=1; V: (1,1)=1; W: (0,0)=-1, (0,1)=1
    # c=5 (M5): U: (1,0)=1, (0,0)=-1; V: (0,0)=1, (1,0)=1; W: (1,0)=1
    # c=6 (M6): U: (0,1)=1, (1,1)=-1; V: (0,1)=1, (1,1)=1; W: (0,0)=-1, (1,1)=1

    # Let me verify c=0:
    # U[0,0]=1, U[3,0]=1 (i*2+j: (0,0)=0, (1,1)=3)
    # V[0,0]=1, V[3,0]=1 (j*2+k: (0,0)=0, (1,1)=3)
    # W[0,0]=1, W[3,0]=1 (k*2+i: (0,0)=0, (1,1)=3)
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[1,1,1] += U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1
    # Others: T[0,0,1] += U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0 (V[1,0]=V[j=0,k=1]=V[0*2+1]=V[1]=0)
    # This seems correct for the (0,0,0) and (1,1,1) entries.

    # Let me build the full Strassen U, V, W for 2x2x2.
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}

    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]

    # c=0: M0
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1

    # c=1: M1
    U_s[2][1] = 1; U_s[3][1] = 1  # (1,0)=2, (1,1)=3
    V_s[0][1] = 1  # (0,0)=0
    W_s[2][1] = 1  # (1,0)=2

    # c=2: M2
    U_s[0][2] = 1  # (0,0)=0
    V_s[2][2] = 1; V_s[3][2] = -1  # (1,0)=2, (1,1)=3
    W_s[1][2] = 1; W_s[3][2] = 1  # (0,1)=1, (1,1)=3

    # c=3: M3
    U_s[3][3] = 1  # (1,1)=3
    V_s[1][3] = 1; V_s[2][3] = -1  # (0,1)=1, (1,0)=2
    W_s[0][3] = 1; W_s[2][3] = 1  # (0,0)=0, (1,0)=2

    # c=4: M4
    U_s[0][4] = 1; U_s[1][4] = 1  # (0,0)=0, (0,1)=1
    V_s[3][4] = 1  # (1,1)=3
    W_s[0][4] = -1; W_s[1][4] = 1  # (0,0)=0, (0,1)=1

    # c=5: M5
    U_s[2][5] = 1; U_s[0][5] = -1  # (1,0)=2, (0,0)=0
    V_s[0][5] = 1; V_s[2][5] = 1  # (0,0)=0, (1,0)=2
    W_s[2][5] = 1  # (1,0)=2

    # c=6: M6
    U_s[1][6] = 1; U_s[3][6] = -1  # (0,1)=1, (1,1)=3
    V_s[1][6] = 1; V_s[3][6] = 1  # (0,1)=1, (1,1)=3
    W_s[0][6] = -1; W_s[3][6] = 1  # (0,0)=0, (1,1)=3

    # Verify: sum_c U_s[i*2+j,c] * V_s[j*2+k,c] * W_s[k*2+i,c] should be 1 for all i,j,k in {0,1}
    def verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for c in range(7):
                        s += U_s[i*2+j][c] * V_s[j*2+k][c] * W_s[k*2+i][c]
                    if s != 1:
                        return False, (i,j,k,s)
        return True, None

    ok, err = verify_strassen()
    if not ok:
        # Fall back to schoolbook
        return {"U": U, "V": V, "W": W}

    # Now build the rank-21 decomposition for 2x2x6.
    # For each block b in {0, 1, 2}, k in {2b, 2b+1}:
    # Global U[i*2+j, 7*b+c] = U_s[i*2+j][c]
    # Global V[j*6+k, 7*b+c] = V_s[j*2+k_loc][c] where k_loc = k - 2b
    # Global W[k*2+i, 7*b+c] = W_s[k_loc*2+i][c] where k_loc = k - 2b

    rank21 = 21
    U21 = [[0]*rank21 for _ in range(4)]
    V21 = [[0]*rank21 for _ in range(12)]
    W21 = [[0]*rank21 for _ in range(12)]

    for b in range(3):
        for c in range(7):
            col = 7*b + c
            for i in range(2):
                for j in range(2):
                    U21[i*2+j][col] = U_s[i*2+j][c]
            for j in range(2):
                for k_loc in range(2):
                    k_global = 2*b + k_loc
                    V21[j*6+k_global][col] = V_s[j*2+k_loc][c]
            for k_loc in range(2):
                k_global = 2*b + k_loc
                for i in range(2):
                    W21[k_global*2+i][col] = W_s[k_loc*2+i][c]

    # Verify the rank-21 decomposition
    def verify_decomp(Um, Vm, Wm, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(Um[0])):
                        s += Um[i*m+j][c] * Vm[j*p+k][c] * Wm[k*n+i][c]
                    if s != 1:
                        return False, (i,j,k,s)
        return True, None

    ok21, err21 = verify_decomp(U21, V21, W21, n, m, p)
    if ok21:
        return {"U": U21, "V": V21, "W": W21}

    # If rank 21 doesn't work, fall back to schoolbook rank 24
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
