"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    max_err = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = 0.0
                for r in range(rank):
                    val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                err = abs(val - T[a, b, c])
                if err > max_err:
                    max_err = err
    return max_err

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 Strassen decomposition (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A11 + A22)(B11 + B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A21 + A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = A11(B12 - B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = A22(B21 - B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (A11 + A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (A21 - A11)(B11 + B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7 = (A12 - A22)(B21 + B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    return U, V, W

def _block_strassen(n, m, p):
    """3-block Strassen for 2x2 x 2x6, rank 21."""
    U_total = [[0]*21 for _ in range(4)]
    V_total = [[0]*21 for _ in range(12)]
    W_total = [[0]*21 for _ in range(12)]
    U_str, V_str, W_str = _strassen_2x2()
    for block in range(3):
        base = block * 7
        for i in range(4):
            for r in range(7):
                U_total[i][base + r] = U_str[i][r]
        b_map = [2*block, 2*block+1, 6+2*block, 6+2*block+1]
        for j in range(4):
            for r in range(7):
                V_total[b_map[j]][base + r] = V_str[j][r]
        w_map = [4*block, 4*block+2, 4*block+1, 4*block+3]
        for k in range(4):
            for r in range(7):
                W_total[w_map[k]][base + r] = W_str[k][r]
    return U_total, V_total, W_total

def _try_rank20():
    """
    Attempt to find a rank-20 decomposition for M_{2,2,6}.
    
    Strategy: The rank-21 block Strassen has 3 independent sets of 7 multiplications.
    We try to merge them by allowing the A-side linear combinations to be shared
    across blocks. Specifically, we parameterize a rank-20 solution where:
    - U has 4 rows, 20 columns
    - V has 12 rows, 20 columns  
    - W has 12 rows, 20 columns
    
    We use the known structure: for each of the 7 Strassen "patterns", we have
    a U-vector (4-dim). In the rank-21 solution, the same U-vector is repeated
    3 times (once per block). For rank 20, we try to have one of the 7 patterns
    be split differently, or to have a shared pattern that covers parts of two blocks.
    
    Alternative approach: Use the fact that M_{2,2,6} can be viewed as a sub-tensor
    of M_{2,2,2} x M_{1,1,3}. We try a "mixed" decomposition where we use 6 full
    Strassen patterns (18 mults) plus 2 additional shared multiplications.
    
    Most direct approach: Set up the bilinear system and solve.
    For rank r, we need:
      T[a,b,c] = sum_{s=1}^{r} U[a,s] * V[b,s] * W[c,s]
    
    Fix U to have a specific sparse pattern, then solve for V and W linearly.
    
    Let's try: U has the same structure as the 3-block Strassen but with one
    column removed and adjusted. We'll parameterize U with small integer entries
    and solve the resulting linear system for V and W.
    """
    n, m, p = 2, 2, 6
    r = 20
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Strategy: Use a structured U matrix.
    # For each of the 7 Strassen patterns, we have a 4-vector u_s.
    # In rank 21, we use u_s three times (for blocks 0,1,2).
    # For rank 20, we try: use u_s twice for s=0..5 (12 mults) + 
    #   for s=6, use a "combined" pattern that covers block 2 and part of others.
    # This is getting complex. Let's try a more systematic approach.
    
    # Systematic approach: 
    # We know a rank-21 solution. Try to perturb it to get rank 20.
    # Take the rank-21 U, V, W. Remove one column (say column 20) and try to
    # adjust the remaining columns to compensate.
    
    # Better: Direct construction.
    # For M_{2,2,6}, think of B as having 6 columns. We can group them as:
    #   Group 1: cols 0,1 (block 0)
    #   Group 2: cols 2,3 (block 1)  
    #   Group 3: cols 4,5 (block 2)
    
    # The 7 Strassen patterns give linear combinations of A entries.
    # Let u_s be the 4-vector for pattern s (s=0..6).
    # For block t (t=0,1,2), the B-side vector for pattern s is v_{s,t} (12-dim,
    #   but only 4 non-zero entries corresponding to block t's B columns).
    # The W-side vector for pattern s, block t is w_{s,t} (12-dim, only 4 non-zero).
    
    # In rank 21: columns are (s,t) for s in 0..6, t in 0..2.
    #   U[:, (s,t)] = u_s
    #   V[:, (s,t)] = v_{s,t}
    #   W[:, (s,t)] = w_{s,t}
    
    # For rank 20, we need to merge one pair. Suppose we try to have column 20
    # serve a dual purpose. This is hard to do cleanly.
    
    # Alternative: Use a different algebraic structure.
    # M_{2,2,6} has a known rank-20 decomposition based on the "Laser method".
    # The idea: M_{2,2,6} is a sub-tensor of M_{3,3,3} which has rank <= 23.
    # But we need an explicit construction.
    
    # Let me try a computational search:
    # Parameterize U as a 4x20 matrix with entries in {-1,0,1}.
    # For each candidate U, check if the linear system for V,W is consistent.
    # The system: for each (a,b,c) with T[a,b,c]=1, sum_s U[a,s] V[b,s] W[c,s] = 1
    #             for each (a,b,c) with T[a,b,c]=0, sum_s U[a,s] V[b,s] W[c,s] = 0
    # This is bilinear in (V,W), so it's not linear. But if we fix U, it becomes
    # a system of quadratic equations in V and W.
    
    # Better: Fix U and one of V or W, then solve linearly for the other.
    # If we fix U and V, the system becomes linear in W:
    #   For each a,c: sum_b T[a,b,c] * (sum_s U[a,s] V[b,s]) = sum_s U[a,s] W[c,s] * (something)
    # Actually, T[a,b,c] = delta_{a%m, b//p} * delta_{a//m, c//p} * delta_{b%p, c%p}
    # So the constraint is:
    #   sum_s U[a,s] V[b,s] W[c,s] = 1 if (a%m == b//p) and (a//m == c//p) and (b%p == c%p)
    #   sum_s U[a,s] V[b,s] W[c,s] = 0 otherwise
    
    # This is a 3-index constraint. If we fix U and V, for each c:
    #   sum_s W[c,s] * (sum over a,b of T[a,b,c] * U[a,s] V[b,s] ... ) 
    # No, it's not separable that way.
    
    # Let me try a different tactic: use the known rank-21 solution and try to
    # find a rank-20 solution by solving a least-squares problem.
    
    # Actually, let me just try the most promising structural idea:
    # Use 2 full Strassen blocks (14 mults) + a combined block for the third.
    # For the third block, instead of 7 separate mults, use 6 mults by exploiting
    # that we can share some linear combinations.
    
    # This is getting too complex for a clean implementation. Let me try a 
    # numerical approach: set up the rank-20 problem and use scipy to solve.
    
    # For a rank-r decomposition, we have r*4 + r*12 + r*12 = 28r unknowns.
    # The constraints are: 4*12*12 = 576 scalar equations (but many are trivially 0=0).
    # Non-trivial constraints: 2*2*6 = 24 equations of the form sum_s ... = 1
    # And 576-24 = 552 equations of the form sum_s ... = 0.
    
    # This is a system of 576 bilinear equations in 28r unknowns.
    # For r=20: 560 unknowns, 576 equations. Overdetermined but should be solvable.
    
    # Let's use a simple iterative approach:
    # 1. Start with the rank-21 solution, drop one column.
    # 2. Solve for the remaining 20 columns to best fit the tensor.
    # 3. Check if the fit is exact.
    
    U21, V21, W21 = _block_strassen(n, m, p)
    
    # Drop the last column (column 20)
    U20 = [row[:20] for row in U21]
    V20 = [row[:20] for row in V21]
    W20 = [row[:20] for row in W21]
    
    err = _verify(U20, V20, W20, n, m, p)
    if err < 1e-9:
        return {"U": U20, "V": V20, "W": W20}
    
    # That won't work directly. Let's try a different approach:
    # Use the fact that we can add a linear combination of columns to compensate.
    
    # Actually, let me try the most direct approach: 
    # For rank 20, parameterize U with a specific structure and solve.
    
    # Known result: M_{2,2,6} has rank 20. One construction:
    # Use the "border rank" approach or explicit formulas.
    
    # Let me try: U is 4x20. For each of the 20 columns, U[:,s] is one of:
    #   e_0, e_1, e_2, e_3, e_0+e_1, e_0+e_2, e_0+e_3, e_1+e_2, e_1+e_3, 
    #   e_2+e_3, e_0+e_3-e_1, e_0+e_3-e_2, etc.
    # And we solve for V and W.
    
    # This is a large search space. Let me try a more targeted approach.
    
    # Targeted approach: The rank-21 solution has U columns that are 7 distinct 
    # 4-vectors, each repeated 3 times. For rank 20, we need to "merge" one 
    # repetition. 
    # 
    # Idea: Instead of having u_6 appear 3 times (for blocks 0,1,2), have it 
    # appear 2 times, and create a new column that's a combination.
    #
    # Let's try: take the rank-21 solution, and for the 7th pattern (index 6),
    # instead of 3 separate columns, use 2 columns that together cover all 3 blocks.
    # This means we need 2 columns whose V and W vectors span the 3 block-specific
    # vectors.
    
    # This is a subspace problem. Let me check if it's feasible.
    
    # For pattern 6 (M7 = (A12 - A22)(B21 + B22)):
    # u_6 = (0, 1, 0, -1)
    # For block 0: v_6,0 has non-zero at B indices 2,3 (B21,B22 of block 0)
    #   v_6,0[2] = 1, v_6,0[3] = 1 (in the 12-dim B space, these are indices 2,3)
    #   Wait, B indices: block 0 uses B[0],B[1] (row 0) and B[6],B[7] (row 1)
    #   So B21 -> index 6, B22 -> index 7
    #   v_6,0[6] = 1, v_6,0[7] = 1
    # For block 1: v_6,1[8] = 1, v_6,1[9] = 1
    # For block 2: v_6,2[10] = 1, v_6,2[11] = 1
    #
    # Similarly for W:
    # w_6,0: C indices for block 0: C11->W[0], C12->W[2], C21->W[1], C22->W[3]
    #   M7 contributes to C11 and C12: w_6,0[0] = 1, w_6,0[2] = 1
    # w_6,1: w_6,1[4] = 1, w_6,1[6] = 1
    # w_6,2: w_6,2[8] = 1, w_6,2[10] = 1
    
    # Can we find 2 vectors v_a, v_b (12-dim) such that the span of {v_a, v_b}
    # contains {v_6,0, v_6,1, v_6,2}? That requires the 3 vectors to be in a 2-dim
    # subspace, which they're not (they have disjoint supports).
    
    # So this approach won't work directly.
    
    # Let me try a completely different structural idea:
    # Use the "grouping" trick. Instead of grouping B into 3 pairs, group them
    # differently. For example:
    #   Group 1: cols 0, 2, 4 (even columns)
    #   Group 2: cols 1, 3, 5 (odd columns)
    # Then we have two 2x3 sub-matrices. M_{2,2,3} has rank 10 (known result).
    # So total rank = 10 + 10 = 20!
    
    # Let me verify: M_{2,2,3} rank.
    # M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3.
    # Schoolbook rank = 2*2*3 = 12.
    # Strassen-based: split B into 2x2 + 2x1.
    #   2x2 part: rank 7
    #   2x1 part: rank 2 (just 2 mults: A11*B21+A21*B11... no, 2x1 is just 2 mults)
    #   Actually, 2x2 times 2x1: C is 2x1. C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   This can be done with 2 mults? No, it needs 4 mults in schoolbook, but
    #   can be done with fewer. Actually, for 2x2 * 2x1, the rank is 4 (I think).
    #   Let me check: C[0] = A00*B00 + A01*B10, C[1] = A10*B00 + A11*B10
    #   This is a bilinear map from (A00,A01,A10,A11) x (B00,B10) -> (C0,C1).
    #   The tensor has rank 4 (each C entry needs 2 mults, and they share B entries
    #   but not A entries in a way that allows sharing).
    #   Actually, let me compute: the 4x2x2 tensor T[a,b,c] where a in 0..3 (A entries),
    #   b in 0..1 (B entries), c in 0..1 (C entries).
    #   T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1, rest 0.
    #   This is 4 disjoint rank-1 tensors, so rank 4.
    # So M_{2,2,3} rank <= 7 + 4 = 11.
    # But the known rank of M_{2,2,3} is 10.
    
    # Let me look up: the rank of M_{2,2,3} is 10. This is a known result.
    # So M_{2,2,6} rank <= 2 * 10 = 20!
    
    # Now I need an explicit rank-10 decomposition for M_{2,2,3}.
    
    return None  # Fall through to other approaches

def _rank10_2x2x3():
    """
    Construct a rank-10 decomposition for M_{2,2,3}.
    A is 2x2, B is 2x3, C is 2x3.
    """
    n, m, p = 2, 2, 3
    r = 10
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # C indices (W convention): W[k*n+i, r] where i=row, k=col
    #   C[0,0] -> W[0], C[1,0] -> W[1], C[0,1] -> W[2], C[1,1] -> W[3],
    #   C[0,2] -> W[4], C[1,2] -> W[5]
    
    # We need to find U (4x10), V (6x10), W (6x10) such that
    # sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
    
    # Let's use a Strassen-like approach:
    # Split B into B_left (2x2, cols 0,1) and B_right (2x1, col 2).
    # M_{2,2,2} part: rank 7 (Strassen)
    # M_{2,2,1} part: rank 4
    # Total: rank 11. But we want rank 10, so we need to save 1.
    
    # The savings come from sharing between the two parts.
    
    # Let me try to construct it explicitly.
    # 
    # For the 2x2 part (cols 0,1 of B), use Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # For the 2x1 part (col 2 of B):
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # Schoolbook: 4 mults: A00*B02, A01*B12, A10*B02, A11*B12
    #
    # Total: 7 + 4 = 11. To get 10, we need to merge one mult.
    #
    # Idea: Can we modify one of the Strassen mults to also contribute to C02 or C12?
    # For example, modify M3 = A00(B01-B11) to be A00(B01-B11 + alpha*B02)
    # Then it contributes to C02 as well. But then we need to adjust other terms.
    
    # Let me try a different approach: use a known rank-10 formula.
    # 
    # Actually, let me just try to solve it computationally.
    # Set up the system and use a simple solver.
    
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(6)]
    W = [[0]*r for _ in range(6)]
    
    # Initialize with the rank-11 solution (7 Strassen + 4 schoolbook)
    # Strassen for 2x2 part (B cols 0,1):
    # B indices for 2x2 part: B00=0, B01=1, B10=3, B11=4
    # C indices for 2x2 part: C00=0, C01=2, C10=1, C11=3
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    # Schoolbook for 2x1 part (B col 2): B02=2, B12=5
    # C02 -> W[4], C12 -> W[5]
    # M8 = A00 * B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    # M9 = A01 * B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    # M10 = A10 * B02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    # M11 = A11 * B12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    # This is rank 11. Now try to reduce to rank 10.
    # 
    # Idea: Merge M8 (A00*B02) with one of the Strassen mults.
    # For example, modify M3: instead of A00*(B01-B11), use A00*(B01-B11+B02).
    # Then M3 contributes to C02 as well. We can remove M8.
    # But then C02 = M3's contribution + M9 + ... we need to check.
    
    # Let's try: modify M3 to include B02.
    # New M3' = A00 * (B01 - B11 + B02)
    # This contributes to:
    #   C01: from B01 part: +1 * A00 * B01 (same as before)
    #   C11: from -B11 part: -1 * A00 * B11 (same as before)
    #   C02: from +B02 part: +1 * A00 * B02 (new)
    # So we can remove M8 (A00*B02) and adjust.
    
    # Update:
    V[2][2] = 1  # Add B02 to M3
    W[4][2] = 1  # M3 now contributes to C02
    # Remove M8 (column 7):
    # Just zero it out and reduce rank.
    
    # But wait, we need to verify this gives an exact decomposition.
    # Let's check C02:
    # C02 = A00*B02 + A01*B12
    # With modified M3: M3' contributes A00*B02 to C02.
    # M9 contributes A01*B12 to C02.
    # So C02 = M3' + M9. Good.
    # 
    # Check C01:
    # C01 = A00*B01 + A01*B11
    # M3' contributes A00*B01 to C01.
    # M5 contributes A01*B11 to C01.
    # So C01 = M3' + M5. Same as before. Good.
    #
    # Check C11:
    # C11 = A00*B01 + A01*B11 + A10*B00 + A11*B11
    # Wait, C11 = A10*B01 + A11*B11? No.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = A10*B01 + A11*B11
    # M3' contributes -A00*B11 to C11? No, M3' = A00*(B01-B11+B02)
    #   The -B11 part: -A00*B11. But C11 doesn't involve A00.
    #   Wait, I'm confusing myself. Let me recheck.
    # 
    # In the Strassen decomposition for 2x2:
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # 
    # M3 = A00*(B01-B11)
    # M3 contributes to C01 (via +B01) and C11 (via -B11).
    # 
    # If I modify M3 to M3' = A00*(B01-B11+B02):
    # M3' contributes to C01 (via +B01), C11 (via -B11), and C02 (via +B02).
    # 
    # C01 = M3' + M5: 
    #   M3' gives A00*B01, M5 gives A01*B11. C01 = A00*B01 + A01*B11. Correct.
    # C11 = M1 - M2 + M3' + M6:
    #   M3' gives -A00*B11. But C11 should have A10*B01 + A11*B11.
    #   The -A00*B11 term is spurious! This breaks C11.
    # 
    # So this simple modification doesn't work.
    
    # Let me try a different merge. 
    # 
    # Alternative: Use a different rank-10 construction for M_{2,2,3}.
    # 
    # Known: M_{2,2,3} has rank 10. One construction uses the "Laser method"
    # or explicit formulas. Let me try to find one.
    #
    # Actually, let me just try to solve the rank-10 problem numerically.
    # I'll set up the bilinear system and use a simple iterative solver.
    
    # For now, let's just return the rank-11 solution and use it to get rank 22
    # for M_{2,2,6}, which is worse than 21. So this approach is not helpful
    # unless I can find the rank-10 solution.
    
    # Let me try a different strategy: directly search for rank 20 of M_{2,2,6}
    # using a numerical optimization approach.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Try rank 20 first
    result = _try_rank20()
    if result is not None:
        err = _verify(result["U"], result["V"], result["W"], n, m, p)
        if err < 1e-9:
            return result
    
    # Fall back to rank 21 (3-block Strassen)
    U21, V21, W21 = _block_strassen(n, m, p)
    err = _verify(U21, V21, W21, n, m, p)
    if err < 1e-9:
        return {"U": U21, "V": V21, "W": W21}
    
    # Fall back to schoolbook rank 24
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
