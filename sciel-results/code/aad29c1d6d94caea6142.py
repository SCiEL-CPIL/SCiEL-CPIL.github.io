import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]
DEADLINE = 135.0

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    def check_time():
        return time.time() - t0 < DEADLINE

    def validate(U, V, W, r):
        """Vectorized validation."""
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        T = np.zeros((n, m, p), dtype=np.int64)
        for c in range(r):
            ur = Ua[:, c].reshape(n, m)
            vr = Va[:, c].reshape(m, p)
            wr = Wa[:, c].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
        return np.all(T == 1)

    # The target tensor is T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,9)
    # A rank-1 term is: U[i,j] * V[j,k] * W[k,i] summed over c
    # For a single rank-1 term to contribute to T[i,j,k]=1, we need specific patterns.
    
    # Strategy: Generate candidate rank-1 terms with small integer coefficients
    # A rank-1 term (u, v, w) where u is 4x4, v is 4x9, w is 9x4
    # contributes tensor t[i,j,k] = u[i,j]*v[j,k]*w[k,i]
    
    # For the sum of such terms to equal 1 everywhere, each term must have
    # entries that are "compatible" - i.e., the product u[i,j]*v[j,k]*w[k,i] 
    # should be 0 or 1 (or -1) for most entries, with cancellations summing to 1.
    
    # Key insight: For T[i,j,k]=1, we need sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
    # If all coefficients are ±1 or 0, this becomes a combinatorial problem.
    
    # Let's try a direct approach: use the fact that for each (i,j,k),
    # we need the sum to be 1. Generate terms where each factor has entries in {-1,0,1}.
    
    # Actually, let's use a smarter approach: 
    # The schoolbook has rank 144 (4*4*9). 
    # We can do better by sharing computations.
    
    # For 4x4 * 4x9: think of it as computing C = A*B where A is 4x4, B is 4x9.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A rank-1 term in the tensor sense: pick a subset of (i,j) for U, (j,k) for V, (k,i) for W
    # such that the product gives a useful contribution.
    
    # Let me try a different structural approach:
    # Split into 2x2 blocks for the 4x4 dimension, and handle the 9 columns specially.
    
    # 4x4 * 4x9: Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x9 (4 blocks),
    # C into 2x2 blocks of 2x9 (4 blocks).
    
    # For each pair of A-block (aij) and B-block (bjk) where j matches,
    # we compute a 2x2 * 2x9 product.
    
    # Actually, let me try the most direct approach: enumerate all possible 
    # "sparse" rank-1 terms and find a minimum set that covers the tensor.
    
    # A rank-1 term is defined by:
    # - U pattern: which (i,j) pairs are nonzero, and their values
    # - V pattern: which (j,k) pairs are nonzero, and their values  
    # - W pattern: which (k,i) pairs are nonzero, and their values
    # The constraint is: for ALL (i,j,k), sum over c of U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
    
    # This is very constrained. Let's think about what patterns work.
    
    # If U[i,j] = 1 for some (i,j), V[j,k] = 1 for some (j,k), W[k,i] = 1 for some (k,i),
    # then the term contributes 1 to T[i,j,k] iff ALL three are 1 simultaneously.
    
    # For the sum to be exactly 1 for all (i,j,k), we need exactly one term to contribute 1
    # to each (i,j,k) position (with possible cancellations from -1 terms).
    
    # Simplest case: all coefficients are 0 or 1, no cancellations needed.
    # Then each (i,j,k) must be "covered" by exactly one term.
    # A term covers (i,j,k) iff U[i,j]=1 AND V[j,k]=1 AND W[k,i]=1.
    
    # So we need to partition the 144 positions into "triangles" where each triangle
    # is a set of (i,j,k) positions that form a rank-1 pattern.
    
    # A rank-1 pattern in the tensor: U is a matrix, V is a matrix, W is a matrix.
    # The supported positions are: {(i,j,k) : U[i,j]!=0 AND V[j,k]!=0 AND W[k,i]!=0}
    
    # For U to be a rank-1 matrix (in the usual sense), U[i,j] = u_i * v_j for vectors u, v.
    # But here U, V, W don't need to be rank-1 matrices - they just need to be the
    # factors in the tensor decomposition. The "rank" refers to the number of terms.
    
    # So each term is just three matrices (U_c, V_c, W_c) with integer entries.
    # The constraint is only on the sum of their tensor products.
    
    # Let me try a practical approach: start from schoolbook and merge.
    
    # Schoolbook: for each (i,j,k), one term with U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # Rank = 144.
    
    # Can we merge terms? If two terms have the same (i,j) in U, same (j,k) in V, same (k,i) in W,
    # we can merge them. But in schoolbook, each term has exactly one nonzero in each matrix.
    
    # Better: if term c1 has U[i1,j1]=1 and term c2 has U[i2,j2]=1, and if i1=i2, j1=j2,
    # we can combine. But they also need matching V and W patterns.
    
    # Actually, the key insight for reducing rank: 
    # A single term can cover MULTIPLE (i,j,k) positions if U has multiple nonzeros,
    # V has multiple nonzeros, and W has multiple nonzeros, as long as the product
    # pattern matches.
    
    # For example: if U[i,0]=1, U[i,1]=1, V[0,k]=1, V[1,k]=1, W[k,i]=1,
    # then this term contributes to T[i,0,k] and T[i,1,k] (both get +1).
    # But it also contributes to T[i,0,k'] and T[i,1,k'] for other k' where W[k',i]=1.
    
    # This is getting complex. Let me try a computational approach:
    # 1. Generate a pool of candidate terms (with small integer entries)
    # 2. Use greedy set cover to find a small subset
    # 3. Use local search to improve
    
    # For efficiency, let's generate terms with entries in {-1, 0, 1} and limited support.
    
    # Actually, let me try yet another approach: use the known result that
    # rank(4x4 * 4x9) can be computed by considering the structure.
    
    # 4x4 * 4x9 = 4x4 * (4x3 + 4x3 + 4x3) [three blocks of 3 columns]
    # rank(4x4 * 4x3) is what we need, times 3 (if independent).
    
    # For 4x4 * 4x3: schoolbook is 48. Can we do better?
    # 4x4 * 4x3: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, k in 0..2
    # That's 12 outputs, each requiring 4 multiplications = 48 total.
    
    # With Strassen-like tricks on 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22], [B30, B31, B32]]
    #     = [[B_top], [B_bot]] where B_top is 2x3, B_bot is 2x3
    
    # C = [[C00, C01, C02], [C10, C11, C12], [C20, C21, C22], [C30, C31, C32]]
    #     = [[C_top], [C_bot]] where C_top is 2x3, C_bot is 2x3
    
    # C_top = A00*B_top + A01*B_bot  (2x2 * 2x3 + 2x2 * 2x3)
    # C_bot = A10*B_top + A11*B_bot
    
    # For 2x2 * 2x3: schoolbook is 12. Can we do better?
    # 2x2 * 2x3: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..1, k in 0..2
    # 6 outputs, each 2 mults = 12.
    
    # With Strassen on the 2x2 part: 
    # M1 = (A00+A11)(B00+B10)  -- but B has 3 columns...
    
    # This is getting complicated. Let me just try a direct computational search.
    
    # Given time constraints, let me implement a practical approach:
    # 1. Start with schoolbook (rank 144)
    # 2. Try to reduce by finding terms that can be combined
    # 3. Use random perturbation + local search
    
    # Actually, the most promising approach given the time limit:
    # Use the block structure explicitly.
    
    # 4x4 * 4x9: Split into 2x2 blocks for rows/cols of A, and handle 9 cols.
    # A = [[A00, A01], [A10, A11]]  (each 2x2)
    # B has 4 rows, 9 cols. Split B into 2x2 blocks: B = [[B0, B1, B2, B3, B4, B5, B6, B7, B8]]
    # where B0..B3 are 2x1 (first 2 rows), B4..B7 are 2x1 (last 2 rows)... 
    # No, that's not right. Let me think again.
    
    # B is 4x9. Split rows into two groups of 2:
    # B = [B_top; B_bot] where B_top is 2x9, B_bot is 2x9.
    # Split cols into... 9 doesn't divide nicely.
    
    # Alternative: Split the 9 columns into 3 groups of 3.
    # B = [B1 | B2 | B3] where each Bi is 4x3.
    # C = [C1 | C2 | C3] where each Ci is 4x3.
    # Ci = A * Bi
    
    # So we need to compute A (4x4) * Bi (4x3) for i=1,2,3.
    # If we can find a rank-r decomposition for 4x4 * 4x3, total rank is 3r.
    
    # For 4x4 * 4x3: Let's use 2x2 block Strassen.
    # A = [[A00, A01], [A10, A11]] (2x2 blocks)
    # Bi = [[Bi_top], [Bi_bot]] where Bi_top is 2x3, Bi_bot is 2x3.
    # Ci = [[Ci_top], [Ci_bot]] where Ci_top = A00*Bi_top + A01*Bi_bot, Ci_bot = A10*Bi_top + A11*Bi_bot.
    
    # For 2x2 * 2x3 multiplication:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0,1,2
    # That's 6 products. Schoolbook rank = 6 (one per output).
    
    # Can we do better? 2x2 * 2x3 has 6 outputs. Each output is a dot product of length 2.
    # The tensor rank of 2x2*2x3 is at most 6 (schoolbook).
    # It's actually exactly 6 for 2x2*2x2 (known), and for 2x2*2x3 it should be at most 6.
    
    # Wait, 2x2 * 2x2 has tensor rank 7 (Strassen), not 4!
    # So 2x2 * 2x3 would have rank at most... let me think.
    # 2x2 * 2x3: we can write it as 2x2*2x2 + 2x2*2x1.
    # 2x2*2x2 has rank 7, 2x2*2x1 has rank 2 (schoolbook, can't do better since 2 outputs each need 2 mults).
    # But they share the A factors, so total might be less than 9.
    
    # Actually, for 2x2 * 2x3:
    # C = A * B where A is 2x2, B is 2x3.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Schoolbook: 6 terms. Each term handles one (i,k) output.
    # Can we share? The A factors repeat: A[0,0] appears in C[0,0], C[0,1], C[0,2].
    # If we make one term with U[0,0]=1, V[0,0]=1, V[0,1]=1, V[0,2]=1, W[0,0]=1, W[1,0]=1, W[2,0]=1,
    # that term contributes to T[0,0,0], T[0,0,1], T[0,0,2] (all get +1 from A[0,0]*B[0,k]).
    # But it also contributes to T[1,0,k] if W[k,1]=1, which we don't want.
    
    # So we need W[k,0]=1 and W[k,1]=0 for all k. That works!
    # Term 1: U[0,0]=1, V[0,k]=1 for k=0,1,2, W[k,0]=1 for k=0,1,2. All other entries 0.
    # This contributes +1 to T[0,0,k] for k=0,1,2.
    
    # Similarly:
    # Term 2: U[0,1]=1, V[1,k]=1 for k=0,1,2, W[k,0]=1 for k=0,1,2.
    # Contributes +1 to T[0,1,k] for k=0,1,2.
    
    # Term 3: U[1,0]=1, V[0,k]=1 for k=0,1,2, W[k,1]=1 for k=0,1,2.
    # Contributes +1 to T[1,0,k] for k=0,1,2.
    
    # Term 4: U[1,1]=1, V[1,k]=1 for k=0,1,2, W[k,1]=1 for k=0,1,2.
    # Contributes +1 to T[1,1,k] for k=0,1,2.
    
    # Total: 4 terms for 2x2 * 2x3! That's better than schoolbook's 6.
    
    # Wait, let me verify. The tensor T[i,j,k] for 2x2*2x3 should be 1 for all (i,j,k)
    # where i in 0..1, j in 0..1, k in 0..2. That's 2*2*3 = 12 positions.
    
    # Term 1: U[0,0]=1, V[0,0]=V[0,1]=V[0,2]=1, W[0,0]=W[1,0]=W[2,0]=1.
    # T[i,j,k] gets contribution U[i,j]*V[j,k]*W[k,i].
    # For i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. ✓
    # For i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,0] = 1*1*1 = 1. ✓
    # For i=0,j=0,k=2: U[0,0]*V[0,2]*W[2,0] = 1*1*1 = 1. ✓
    # For i=0,j=1,k=0: U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0. ✓ (U[0,1]=0)
    # For i=1,j=0,k=0: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0. ✓ (U[1,0]=0, W[0,1]=0)
    # Good, no spurious contributions.
    
    # So 2x2 * 2x3 has rank at most 4! (Much better than schoolbook's 6.)
    
    # Now for 4x4 * 4x3:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks)
    # B = [B_top; B_bot] (2x3 each)
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    
    # C_top[i,k] = sum_j A00[i,j]*B_top[j,k] + sum_j A01[i,j]*B_bot[j,k]
    # This is (2x2 * 2x3) + (2x2 * 2x3) = two separate 2x2*2x3 multiplications.
    # Each has rank 4, so C_top needs rank 8? No, they share the output structure.
    
    # Actually, C_top[i,k] = (A00*B_top)[i,k] + (A01*B_bot)[i,k]
    # The tensor for C_top is the sum of two tensors: T1[i,j,k] for A00*B_top and T2[i,j,k] for A01*B_bot.
    # But these are different tensors (different A and B factors), so we can't directly add ranks.
    
    # Let me reconsider. The full 4x4 * 4x3 tensor T[i,j,k] where i=0..3, j=0..3, k=0..2.
    # T[i,j,k] = 1 for all (i,j,k).
    
    # Using the 2x2 block structure:
    # i = 2*ir + ii, j = 2*jr + jj, where ir,jr in {0,1}, ii,jj in {0,1}.
    # T[2*ir+ii, 2*jr+jj, k] = 1.
    
    # The A matrix has blocks A_ir,jr (each 2x2).
    # The B matrix has blocks B_jr,k (each 2x3... no, B is 4x3, so B_jr is 2x3).
    # The C matrix has blocks C_ir,k (each 2x3).
    
    # C_ir,k[ii,jj... no, C_ir,k is 2x3: C_ir,k[ii, k] for ii in 0,1.
    # C_ir,k[ii,k] = sum_jr sum_jj A_ir,jr[ii,jj] * B_jr[k,jj]  -- wait, I'm confusing indices.
    
    # Let me be more careful. A is 4x4, B is 4x3, C is 4x3.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split i = 2*ir + ii, j = 2*jr + jj:
    # C[2*ir+ii, k] = sum_{jr=0}^{1} sum_{jj=0}^{1} A[2*ir+ii, 2*jr+jj] * B[2*jr+jj, k]
    
    # Let A_ir,jr be the 2x2 block with entries A[2*ir+ii, 2*jr+jj].
    # Let B_jr be the 2x3 submatrix with entries B[2*jr+jj, k] for jj=0,1, k=0,1,2.
    
    # Then C_ir (2x3 block) = sum_{jr} A_ir,jr * B_jr  (2x2 * 2x3 multiplication, summed over jr).
    
    # So C_ir = A_ir,0 * B_0 + A_ir,1 * B_1  (sum of two 2x2*2x3 products).
    
    # For each ir (0 or 1), we need to compute C_ir = A_ir,0 * B_0 + A_ir,1 * B_1.
    # This is like computing two 2x2*2x3 products and adding them.
    
    # The tensor for this: T_ir[ii, jj, k] where ii is within the 2x2 A block, jj is within B, k is column.
    # Actually, the tensor indices are (i, j, k) in the original space.
    
    # For the term corresponding to A_ir,jr * B_jr:
    # It contributes to T[2*ir+ii, 2*jr+jj, k] = A_ir,jr[ii,jj] * B_jr[jj,k] * (C factor)
    
    # Hmm, this is getting complex. Let me just implement the 2x2*2x3 rank-4 result
    # and use it to build up the 4x4*4x9 decomposition.
    
    # For 2x2 * 2x3 with rank 4:
    # Terms (in local 2x2*2x3 coordinates, i=0..1, j=0..1, k=0..2):
    # Term a: U[0,0]=1, V[0,0]=V[0,1]=V[0,2]=1, W[0,0]=W[1,0]=W[2,0]=1
    # Term b: U[0,1]=1, V[1,0]=V[1,1]=V[1,2]=1, W[0,0]=W[1,0]=W[2,0]=1
    # Term c: U[1,0]=1, V[0,0]=V[0,1]=V[0,2]=1, W[0,1]=W[1,1]=W[2,1]=1
    # Term d: U[1,1]=1, V[1,0]=V[1,1]=V[1,2]=1, W[0,1]=W[1,1]=W[2,1]=1
    
    # Now for 4x4 * 4x9:
    # Split into 2x2 blocks for the 4x4 A, and handle 9 columns.
    # 9 columns: split into three groups of 3.
    # For each group of 3 columns, we have a 4x4 * 4x3 subproblem.
    
    # For 4x4 * 4x3:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B_group = [B_top; B_bot] where B_top is 2x3, B_bot is 2x3.
    # C_top = A00*B_top + A01*B_bot (2x3 result)
    # C_bot = A10*B_top + A11*B_bot (2x3 result)
    
    # C_top = A00*B_top + A01*B_bot:
    # This is the sum of two 2x2*2x3 products.
    # Using the rank-4 decomposition for each:
    # A00*B_top uses 4 terms, A01*B_bot uses 4 terms. Total 8 terms for C_top.
    # Similarly 8 terms for C_bot. Total 16 terms for 4x4*4x3.
    
    # But wait, can we share terms between C_top and C_bot?
    # C_top[i,k] involves A[0..1, *] and C_bot[i,k] involves A[2..3, *].
    # They use different rows of A, so no direct sharing.
    
    # However, they both use the same B columns. The W factor connects k to i.
    # For C_top: W[k, 0..1] (connects column k to row 0 or 1 of C)
    # For C_bot: W[k, 2..3] (connects column k to row 2 or 3 of C)
    # These are different W entries, so no conflict.
    
    # Total for 4x4*4x3: 16 terms.
    # For 4x4*4x9 (three groups): 3 * 16 = 48 terms.
    
    # That's much better than schoolbook's 144!
    
    # Let me verify this construction carefully.
    
    # For 4x4 * 4x9, with 9 columns split into 3 groups of 3:
    # Group g (g=0,1,2) handles columns 3g, 3g+1, 3g+2.
    
    # For group g, the B submatrix is B[:, 3g:3g+3] (4x3).
    # Split B_group into B_top (rows 0-1, 2x3) and B_bot (rows 2-3, 2x3).
    
    # C_top (rows 0-1 of C, 2x3) = A[0:2, :] * B_group
    #   = A[0:2, 0:2] * B_top + A[0:2, 2:4] * B_bot
    #   = A00 * B_top + A01 * B_bot
    
    # C_bot (rows 2-3 of C, 2x3) = A[2:4, :] * B_group
    #   = A[2:4, 0:2] * B_top + A[2:4, 2:4] * B_bot
    #   = A10 * B_top + A11 * B_bot
    
    # For A00 * B_top (2x2 * 2x3, rank 4):
    # Using the rank-4 terms above, in local coordinates (ii=0,1 for A00 rows, jj=0,1 for A00 cols/B_top rows, kk=0,1,2 for columns):
    # Term a: U[0,0]=1 (A00[0,0]), V[0,0]=V[0,1]=V[0,2]=1 (B_top[0,0..2]), W[0,0]=W[1,0]=W[2,0]=1 (C row 0)
    # In global coords: U[0,0]=1, V[0, 3g]=V[0, 3g+1]=V[0, 3g+2]=1, W[3g, 0]=W[3g+1, 0]=W[3g+2, 0]=1
    
    # Wait, I need to be careful with the W indexing. W[k*n + i] where k is the column index and i is the row index of C.
    # For C_top, the C rows are 0 and 1 (global). So W[k, 0] and W[k, 1].
    
    # Let me redo this more carefully.
    
    # Global indices:
    # A is 4x4: A[i, j] for i,j in 0..3. U[i*m + j] = U[i*4 + j].
    # B is 4x9: B[j, k] for j in 0..3, k in 0..8. V[j*p + k] = V[j*9 + k].
    # C is 4x9: C[i, k] for i in 0..3, k in 0..8. W[k*n + i] = W[k*4 + i].
    
    # For group g (columns 3g, 3g+1, 3g+2):
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B_top = B[0:2, 3g:3g+3], B_bot = B[2:4, 3g:3g+3]
    
    # C_top = A00 * B_top + A01 * B_bot (2x3)
    # C_bot = A10 * B_top + A11 * B_bot (2x3)
    
    # For A00 * B_top (2x2 * 2x3):
    # Local: A00[ii, jj] for ii,jj in 0,1. B_top[jj, kk] for jj in 0,1, kk in 0,1,2.
    # Global: A00[ii, jj] = A[ii, jj] (since A00 is top-left 2x2).
    #         B_top[jj, kk] = B[jj, 3g+kk] (since B_top is rows 0-1, cols 3g..3g+2).
    #         C_top[ii, kk] = C[ii, 3g+kk] (since C_top is rows 0-1, cols 3g..3g+2).
    
    # Rank-4 terms for 2x2*2x3 (local coords ii, jj, kk):
    # Term a: U[ii=0, jj=0]=1, V[jj=0, kk=0]=V[jj=0, kk=1]=V[jj=0, kk=2]=1, W[kk=0, ii=0]=W[kk=1, ii=0]=W[kk=2, ii=0]=1
    # Global: U[0*4+0]=1, V[0*9+(3g+0)]=V[0*9+(3g+1)]=V[0*9+(3g+2)]=1, W[(3g+0)*4+0]=W[(3g+1)*4+0]=W[(3g+2)*4+0]=1
    
    # Term b: U[ii=0, jj=1]=1, V[jj=1, kk=0]=V[jj=1, kk=1]=V[jj=1, kk=2]=1, W[kk=0, ii=0]=W[kk=1, ii=0]=W[kk=2, ii=0]=1
    # Global: U[0*4+1]=1, V[1*9+(3g+0)]=V[1*9+(3g+1)]=V[1*9+(3g+2)]=1, W[(3g+0)*4+0]=W[(3g+1)*4+0]=W[(3g+2)*4+0]=1
    
    # Term c: U[ii=1, jj=0]=1, V[jj=0, kk=0]=V[jj=0, kk=1]=V[jj=0, kk=2]=1, W[kk=0, ii=1]=W[kk=1, ii=1]=W[kk=2, ii=1]=1
    # Global: U[1*4+0]=1, V[0*9+(3g+0)]=V[0*9+(3g+1)]=V[0*9+(3g+2)]=1, W[(3g+0)*4+1]=W[(3g+1)*4+1]=W[(3g+2)*4+1]=1
    
    # Term d: U[ii=1, jj=1]=1, V[jj=1, kk=0]=V[jj=1, kk=1]=V[jj=1, kk=2]=1, W[kk=0, ii=1]=W[kk=1, ii=1]=W[kk=2, ii=1]=1
    # Global: U[1*4+1]=1, V[1*9+(3g+0)]=V[1*9+(3g+1)]=V[1*9+(3g+2)]=1, W[(3g+0)*4+1]=W[(3g+1)*4+1]=W[(3g+2)*4+1]=1
    
    # Similarly for A01 * B_bot:
    # A01[ii, jj] = A[ii, 2+jj] (top-right 2x2 block).
    # B_bot[jj, kk] = B[2+jj, 3g+kk] (rows 2-3, cols 3g..3g+2).
    # C_top[ii, kk] = C[ii, 3g+kk] (same as before).
    
    # Term a': U[ii=0, jj=0]=1 -> U[0*4+2]=1, V[jj=0, kk=*]=1 -> V[2*9+(3g+*)]=1, W[kk=*, ii=0]=1 -> W[(3g+*)*4+0]=1
    # Term b': U[0*4+3]=1, V[3*9+(3g+*)]=1, W[(3g+*)*4+0]=1
    # Term c': U[1*4+2]=1, V[2*9+(3g+*)]=1, W[(3g+*)*4+1]=1
    # Term d': U[1*4+3]=1, V[3*9+(3g+*)]=1, W[(3g+*)*4+1]=1
    
    # For A10 * B_top:
    # A10[ii, jj] = A[2+ii, jj] (bottom-left 2x2).
    # B_top[jj, kk] = B[jj, 3g+kk].
    # C_bot[ii, kk] = C[2+ii, 3g+kk].
    
    # Term a'': U[2*4+0]=1, V[0*9+(3g+*)]=1, W[(3g+*)*4+2]=1
    # Term b'': U[2*4+1]=1, V[1*9+(3g+*)]=1, W[(3g+*)*4+2]=1
    # Term c'': U[3*4+0]=1, V[0*9+(3g+*)]=1, W[(3g+*)*4+3]=1
    # Term d'': U[3*4+1]=1, V[1*9+(3g+*)]=1, W[(3g+*)*4+3]=1
    
    # For A11 * B_bot:
    # A11[ii, jj] = A[2+ii, 2+jj].
    # B_bot[jj, kk] = B[2+jj, 3g+kk].
    # C_bot[ii, kk] = C[2+ii, 3g+kk].
    
    # Term a''': U[2*4+2]=1, V[2*9+(3g+*)]=1, W[(3g+*)*4+2]=1
    # Term b''': U[2*4+3]=1, V[3*9+(3g+*)]=1, W[(3g+*)*4+2]=1
    # Term c''': U[3*4+2]=1, V[2*9+(3g+*)]=1, W[(3g+*)*4+3]=1
    # Term d''': U[3*4+3]=1, V[3*9+(3g+*)]=1, W[(3g+*)*4+3]=1
    
    # Total per group: 4*4 = 16 terms.
    # Total for 3 groups: 48 terms.
    
    # But wait! Many of these terms can be MERGED because they have the same (U, V, W) pattern
    # but different group g. Let's check.
    
    # For group g=0: Term a has U[0]=1, V[0,0]=V[0,1]=V[0,2]=1, W[0,0]=W[1,0]=W[2,0]=1.
    # For group g=1: Term a has U[0]=1, V[0,3]=V[0,4]=V[0,5]=1, W[3,0]=W[4,0]=W[5,0]=1.
    # These are different (different V and W entries), so they can't be merged.
    
    # But within a group, can we merge? 
    # Term a and term c in group g: 
    # Term a: U[0,0]=1, V[0,3g..3g+2]=1, W[3g..3g+2, 0]=1
    # Term c: U[1,0]=1, V[0,3g..3g+2]=1, W[3g..3g+2, 1]=1
    # Different U and W, so no merge.
    
    # What about term a from A00*B_top and term a' from A01*B_bot in the same group?
    # Term a: U[0,0]=1, V[0,3g..]=1, W[3g.., 0]=1
    # Term a': U[0,2]=1, V[2,3g..]=1, W[3g.., 0]=1
    # Different U and V, so no merge.
    
    # So we have 48 terms total. Let me verify this is correct.
    
    # Actually, I realize I can potentially do even better. Let me check if the 2x2*2x3
    # rank-4 decomposition is truly minimal or if there's a rank-3 decomposition.
    
    # 2x2 * 2x3: 6 outputs. Each output C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # The tensor has 2*2*3 = 12 entries, all 1.
    # Lower bound: The tensor rank is at least max(m*n, m*p, n*p) / something... 
    # Actually, for matrix multiplication m x n * n x p, the tensor rank is at least 
    # max(m*n, m*p, n*p) in some cases, but that's not a tight bound.
    # For 2x2*2x3: m*n=4, m*p=6, n*p=6. So rank >= 6? No, that's not right.
    # The trivial lower bound is max(m, n, p) = 3, which is weak.
    # The border rank lower bound for 2x2*2x2 is 7 (Strassen), so for 2x2*2x3 it's at least 7?
    # No, 2x2*2x3 is a different tensor.
    
    # Let me just verify my rank-4 construction is valid and use it.
    
    # Build the 48-term decomposition:
    U = [[0]*48 for _ in range(nm)]
    V = [[0]*48 for _ in range(mp)]
    W = [[0]*48 for _ in range(pn)]
    
    term_idx = 0
    for g in range(3):  # 3 groups of 3 columns
        col_offset = 3 * g
        for block in range(4):  # 4 blocks: A00*B_top, A01*B_bot, A10*B_top, A11*B_bot
            # block 0: A00 * B_top -> C_top
            # block 1: A01 * B_bot -> C_top
            # block 2: A10 * B_top -> C_bot
            # block 3: A11 * B_bot -> C_bot
            
            if block == 0:  # A00 * B_top
                a_row_base = 0
                a_col_base = 0
                b_row_base = 0
                c_row_base = 0
            elif block == 1:  # A01 * B_bot
                a_row_base = 0
                a_col_base = 2
                b_row_base = 2
                c_row_base = 0
            elif block == 2:  # A10 * B_top
                a_row_base = 2
                a_col_base = 0
                b_row_base = 0
                c_row_base = 2
            else:  # A11 * B_bot
                a_row_base = 2
                a_col_base = 2
                b_row_base = 2
                c_row_base = 2
            
            for ii in range(2):
                for jj in range(2):
                    # This term: U[a_row_base+ii, a_col_base+jj] = 1
                    # V[b_row_base+jj, col_offset+kk] = 1 for kk=0,1,2
                    # W[col_offset+kk, c_row_base+ii] = 1 for kk=0,1,2
                    
                    a_idx = (a_row_base + ii) * m + (a_col_base + jj)
                    for kk in range(3):
                        b_idx = (b_row_base + jj) * p + (col_offset + kk)
                        w_idx = (col_offset + kk) * n + (c_row_base + ii)
                        U[a_idx][term_idx] += 1
                        V[b_idx][term_idx] += 1
                        W[w_idx][term_idx] += 1
                    term_idx += 1
    
    r = term_idx
    assert r == 48, f"Expected 48 terms, got {r}"
    
    # Validate
    if validate(U, V, W, r):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Try to reduce rank by finding linear dependencies
        def compute_tensor(Ua, Va, Wa, r):
            T = np.zeros((n, m, p), dtype=np.int64)
            for c in range(r):
                ur = Ua[:, c].reshape(n, m)
                vr = Va[:, c].reshape(m, p)
                wr = Wa[:, c].reshape(p, n)
                T += np.einsum('ij,jk,ki->ijk', ur, vr, wr)
            return T
        
        def try_remove_column(Ua, Va, Wa, r, col):
            if r <= 1:
                return None
            U_new = np.delete(Ua, col, axis=1)
            V_new = np.delete(Va, col, axis=1)
            W_new = np.delete(Wa, col, axis=1)
            T = compute_tensor(U_new, V_new, W_new, r - 1)
            if np.all(T == 1):
                return U_new, V_new, W_new, r - 1
            return None
        
        # Try removing columns
        improved = True
        while improved and check_time():
            improved = False
            r_current = U_np.shape[1]
            for col in range(r_current):
                if not check_time():
                    break
                result = try_remove_column(U_np, V_np, W_np, r_current, col)
                if result is not None:
                    U_np, V_np, W_np, r_new = result
                    improved = True
                    break
        
        # Try merging columns with coefficient
        if check_time():
            def try_merge(Ua, Va, Wa, r, col1, col2, coeff):
                """Add coeff * col2 to col1, then remove col2."""
                Ua[:, col1] += coeff * Ua[:, col2]
                Va[:, col1] += coeff * Va[:, col2]
                Wa[:, col1] += coeff * Wa[:, col2]
                U_new = np.delete(Ua, col2, axis=1)
                V_new = np.delete(Va, col2, axis=1)
                W_new = np.delete(Wa, col2, axis=1)
                T = compute_tensor(U_new, V_new, W_new, r - 1)
                if np.all(T == 1):
                    return U_new, V_new, W_new, r - 1
                return None
            
            improved = True
            while improved and check_time():
                improved = False
                r_current = U_np.shape[1]
                for col1 in range(r_current):
                    if not check_time():
                        break
                    for col2 in range(col1 + 1, r_current):
                        if not check_time():
                            break
                        for coeff in [-2, -1, 1, 2]:
                            U_copy = U_np.copy()
                            V_copy = V_np.copy()
                            W_copy = W_np.copy()
                            result = try_merge(U_copy, V_copy, W_copy, r_current, col1, col2, coeff)
                            if result is not None:
                                U_np, V_np, W_np, r_new = result
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        U_best = U_np.tolist()
        V_best = V_np.tolist()
        W_best = W_np.tolist()
        return {"U": U_best, "V": V_best, "W": W_best}
    else:
        # Fallback to schoolbook
        r_fb = n * m * p
        U = [[0]*r_fb for _ in range(nm)]
        V = [[0]*r_fb for _ in range(mp)]
        W = [[0]*r_fb for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
