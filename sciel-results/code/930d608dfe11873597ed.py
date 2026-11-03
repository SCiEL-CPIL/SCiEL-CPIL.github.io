import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    n, m, p = 2, 4, 8

    # --- 4x4 matrix multiplication, rank 25 (Coppersmith-Winograd) ---
    # For A (4x4), B (4x4), C = A*B (4x4)
    # U: 16 x 25, V: 16 x 25, W: 16 x 25
    # Index convention: for 4x4 matrix M, index = row*4 + col
    # For W (output C), the task uses W[k*n+i, r] where k is col, i is row.
    # But here we're building a 4x4 decomposition, so W index = row*4 + col for the 4x4 block.
    # We'll handle the mapping to the 2x8 output separately.

    U4 = np.zeros((16, 25), dtype=int)
    V4 = np.zeros((16, 25), dtype=int)
    W4 = np.zeros((16, 25), dtype=int)

    # Coppersmith-Winograd 4x4 rank 25 decomposition
    # This is a known result. Let me use the standard formulation.
    # Actually, let me use a different approach: build the 2x4*4x8 directly
    # using a combination of Strassen blocks and direct multiplications.

    # Alternative approach: decompose the 2x4*4x8 multiplication
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # This is a tensor of size 2*4*8 = 64
    # The trivial rank is 2*4*8 = 64
    # Using Strassen on 2x2 blocks: 2 j-groups * 4 k-groups * 7 = 56

    # Let me try a different structural approach:
    # Treat A as a 2x4 matrix and B as a 4x8 matrix.
    # Split into 2x2 blocks of B: 4 blocks
    # For each 2x2 block of B, we have a 2x2*2x2 subproblem (rank 7 via Strassen)
    # But A is 2x4, so we can split A into two 2x2 blocks.
    # Total: 2 (A blocks) * 4 (B blocks) * 7 = 56

    # Let me try to find a rank-52 or better decomposition.
    # One approach: use the fact that the problem is equivalent to computing
    # 8 independent 2x2*2x2 products but with shared A entries.

    # Actually, let me try a completely different approach:
    # Use the 4x4 rank-25 decomposition and embed.
    # A is 2x4, embed into 4x4 by zero-padding rows 2,3
    # B is 4x8, we need to split into two 4x4 blocks
    # C is 2x8, extract from the 4x4 result

    # For B split into B1 (4x4) and B2 (4x4):
    # A4 = [[A], [0]] (4x4)
    # C4 = A4 * B4
    # C = C4[:2, :] (first 2 rows)

    # So we need 2 instances of 4x4*4x4, each rank 25
    # Total rank = 2 * 25 = 50

    # But wait, A4 is the same for both, so the U vectors are shared!
    # U4 is 16x25, but only rows 0-3 (first 2 rows of A) are non-zero.
    # Actually, A4 has A in rows 0-1, zeros in rows 2-3.
    # So U4 only has non-zero entries in rows 0-7 (first 2 rows of the 4x4).

    # For B1: V4 is 16x25, all rows potentially non-zero
    # For B2: V4' is 16x25, all rows potentially non-zero

    # The U vectors are the same for both B1 and B2 (since A4 is the same)
    # So total rank = 25 (shared U) + 25 (V for B1) + 25 (V for B2) = 75? No.

    # Actually, in tensor decomposition, each term has (U_r, V_r, W_r).
    # For the first 4x4 multiplication (A4 * B1):
    #   U_r from U4, V_r from V4(B1), W_r from W4
    # For the second 4x4 multiplication (A4 * B2):
    #   U_r from U4, V_r from V4(B2), W_r from W4

    # These are independent tensor terms, so total rank = 25 + 25 = 50
    # But the U vectors are duplicated. Can we merge them?
    # No, because the V and W vectors are different.

    # So rank = 50 with this approach.

    # Let me implement the 4x4 rank-25 decomposition.
    # I'll use a known rank-25 decomposition for 4x4 matrix multiplication.

    # Actually, let me use a simpler approach: just use the Strassen-based rank 56
    # but try to optimize it.

    # Let me try yet another approach: direct construction with rank < 56.
    # The tensor is T[i,k,j] = delta[i,k,j] where C[i,j] = sum_k A[i,k]*B[k,j]
    # Size: 2*4*8 = 64

    # Let me use a rank-52 decomposition by mixing Strassen and direct methods.
    # For the 2x4*4x8 problem:
    # Split into 2 independent 2x2*4x8 subproblems (by A rows)
    # Each 2x2*4x8: split B into two 2x4 blocks
    # Each 2x2*2x4: split into two 2x2*2x2 subproblems
    # Each 2x2*2x2: rank 7 (Strassen)
    # Total: 2 * 2 * 2 * 7 = 56

    # To improve, I need a better decomposition of at least one subproblem.

    # Let me try: 2x2*4x8 can be done with rank 14 (instead of 2*7=14, same)
    # Or 2x2*2x4 with rank 6 (instead of 2*7=14? No, 2x2*2x4 is different)

    # Actually, 2x2*2x4: A is 2x2, B is 2x4, C is 2x4
    # C[i,j] = sum_k A[i,k]*B[k,j] for k=0,1
    # This is 2*2*4 = 16 entries
    # Trivial rank: 16
    # Can we do better?
    # For each j, C[:,j] = A * B[:,j] (2x2 * 2x1)
    # 2x2*2x1 has rank 4 (trivial) or... actually it's just 4 scalar mults.
    # But we can share: A is fixed, B has 4 columns.
    # Rank of 2x2*2x4 tensor: this is the same as 2x2 matrix-vector mult, done 4 times.
    # If we do each independently: 4 * 4 = 16
    # Can we do better? The tensor is T[i,k,j] = delta[i,k,j]
    # This is a 2x2x4 tensor. Its rank is... let me think.
    # Actually, this is just 4 independent 2x2*2x1 multiplications.
    # Each 2x2*2x1 has rank 4 (since it's just 4 scalar products).
    # But can we share computation across the 4 columns?
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # For i=0: C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # For i=1: C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    # This is like: for each j, compute [A[0,0]*B[0,j] + A[0,1]*B[1,j], A[1,0]*B[0,j] + A[1,1]*B[1,j]]
    # We can precompute: S0 = A[0,0]*B[0,j] + A[0,1]*B[1,j], S1 = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    # But these depend on j, so no sharing across j.
    # Unless we use Strassen-like tricks on the 2x2 part.
    # Actually, 2x2*2x1 is just 4 scalar mults, rank 4.
    # So 2x2*2x4 has rank at least 4 (for each j, we need at least 4 mults).
    # With 4 j values, rank >= 4. Can we achieve rank 4?
    # No, because for each j, we need 4 independent products.
    # Actually, the rank of the 2x2x4 tensor is 4? Let me check.
    # T[i,k,j] = 1 if the indices match the multiplication pattern.
    # T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1, T[0,0,2]=1, T[0,1,2]=1, T[0,0,3]=1, T[0,1,3]=1
    # T[1,0,0]=1, T[1,1,0]=1, T[1,0,1]=1, T[1,1,1]=1, T[1,0,2]=1, T[1,1,2]=1, T[1,0,3]=1, T[1,1,3]=1
    # All other entries are 0.
    # This tensor has rank 4? Each "slice" j is a 2x2 identity-like matrix.
    # Actually, T[:, :, j] = [[1,1],[1,1]] for all j.
    # So T = sum_j e_j (outer) * [[1,1],[1,1]]
    # This can be written as: T[i,k,j] = 1 for all i,k,j in the support.
    # Rank of this tensor: it's a 2x2x4 tensor where every entry in the support is 1.
    # This is equivalent to 4 copies of the 2x2 all-ones tensor.
    # The 2x2 all-ones tensor has rank 1 (it's just the outer product of [1,1] and [1,1]).
    # Wait, T[i,k,j] = 1 means C[i,j] = sum_k A[i,k]*B[k,j] where the "coefficient" is 1 for all i,k.
    # That's not right. Let me re-examine.

    # I'm getting confused. Let me just implement the rank-56 solution cleanly
    # and see if I can find a rank-52 or better decomposition through search.

    # For now, let me implement a clean rank-56 solution and add a search step.

    # Actually, let me try the 4x4 embedding approach more carefully.
    # If I can get a rank-25 4x4 decomposition, then 2*25 = 50 < 56.

    # Let me hardcode a rank-25 4x4 decomposition.
    # I'll use the known Coppersmith-Winograd 4x4 rank-25 algorithm.

    # Here's a rank-25 decomposition for 4x4 matrix multiplication:
    # Reference: Coppersmith and Winograd (1987)

    # Let me use a different known decomposition.
    # Actually, the best known rank for 4x4 is 25, but I need the explicit decomposition.

    # Let me use a simpler approach: rank-56 with Strassen, but implement it more efficiently.

    # I'll just return the rank-56 solution for now, but with a cleaner implementation.

    total_rank = 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    # Strassen 2x2x2 base blocks
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)

    # M1: (A00+A11)*(B00+B11)
    U_local[0, 0] = 1; U_local[3, 0] = 1
    V_local[0, 0] = 1; V_local[3, 0] = 1
    W_local[0, 0] = 1
    W_local[3, 0] = 1

    # M2: (A10+A11)*B00
    U_local[2, 1] = 1; U_local[3, 1] = 1
    V_local[0, 1] = 1
    W_local[2, 1] = 1
    W_local[3, 1] = -1

    # M3: A00*(B01-B11)
    U_local[0, 2] = 1
    V_local[1, 2] = 1; V_local[3, 2] = -1
    W_local[1, 2] = 1
    W_local[3, 2] = 1

    # M4: A11*(B10-B00)
    U_local[3, 3] = 1
    V_local[2, 3] = 1; V_local[0, 3] = -1
    W_local[0, 3] = 1
    W_local[2, 3] = 1

    # M5: (A00+A01)*B11
    U_local[0, 4] = 1; U_local[1, 4] = 1
    V_local[3, 4] = 1
    W_local[0, 4] = -1
    W_local[1, 4] = 1

    # M6: (A10-A00)*(B00+B01)
    U_local[2, 5] = 1; U_local[0, 5] = -1
    V_local[0, 5] = 1; V_local[1, 5] = 1
    W_local[3, 5] = 1

    # M7: (A01-A11)*(B10+B11)
    U_local[1, 6] = 1; U_local[3, 6] = -1
    V_local[2, 6] = 1; V_local[3, 6] = 1
    W_local[0, 6] = 1

    for jg in range(2):
        a_map = [0*4 + jg*2 + 0, 0*4 + jg*2 + 1, 1*4 + jg*2 + 0, 1*4 + jg*2 + 1]
        for kg in range(4):
            b_map = [(jg*2+0)*8 + kg*2+0, (jg*2+0)*8 + kg*2+1,
                     (jg*2+1)*8 + kg*2+0, (jg*2+1)*8 + kg*2+1]
            c_map = [(kg*2+0)*2+0, (kg*2+1)*2+0, (kg*2+0)*2+1, (kg*2+1)*2+1]
            base = (jg*4 + kg)*7
            for l in range(7):
                r = base + l
                for la in range(4):
                    v = U_local[la, l]
                    if v: U[a_map[la], r] += v
                for lb in range(4):
                    v = V_local[lb, l]
                    if v: V[b_map[lb], r] += v
                for lc in range(4):
                    v = W_local[lc, l]
                    if v: W[c_map[lc], r] += v

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
