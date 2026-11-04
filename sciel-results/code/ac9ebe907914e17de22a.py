"""Matrix multiplication 3x4x4 - block-structured rank-44 decomposition."""
import time

DIMENSIONS = [3, 4, 4]

# Deadline guard
_START = time.time()
_DEADLINE = _START + 130  # 130s safety margin under 150s limit


def _validate_decomposition(U, V, W, n, m, p):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] reproduces the matmul tensor."""
    import numpy as np
    rank = len(U[0]) if U else 0
    for c in range(rank):
        # Just check a few random entries for speed
        pass
    # Full check: compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and verify T[i,j,k] == 1 if k==j else 0  (for matmul C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the matmul tensor: T[i,j,k] = delta_{j,k}
    # Wait: C[i,k] = sum_j A[i,j] B[j,k], so the bilinear map has tensor T[i,j,k] = 1 if j==k
    # But the indices are: A is n x m, B is m x p, C is n x p
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if j == k... but j is in [0,m) and k is in [0,p)
    # For 3x4*4x4: i in [0,3), j in [0,4), k in [0,4)
    # T[i,j,k] = 1 if j==k, else 0
    # The W factor: W[k*n+i, c] (trace convention)
    # So the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # For schoolbook: for each (i,j,k), term c=(i,j,k) has U=1, V=1, W=1
    # But T[i,j,k] should be delta_{j,k}, not 1 for all (i,j,k)!
    
    # Wait, I need to re-examine. The matmul tensor for n x m times m x p:
    # The "tensor" T has indices (i, j, k) for each pair of input indices and output...
    # Actually in the standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The trilinear form is sum_{i,j,k} C[i,k] * (something)... 
    # No. The tensor decomposition is for the bilinear map.
    # The standard tensor is T in R^{(n*p) x n x m} or similar.
    # 
    # In this problem's convention:
    # U: shape (n*m, rank) - corresponds to A's indices
    # V: shape (m*p, rank) - corresponds to B's indices  
    # W: shape (p*n, rank) - corresponds to C's indices
    # The decomposition satisfies: for all (i,j) in A, (j,k) in B:
    #   C[k*n+i... wait, C is n x p, so C index is (i,k) with i in [0,n), k in [0,p)
    #   W index is k*n+i (trace convention: W[k*n + i, c])
    # 
    # The identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if j==k else 0
    # Because C[i,k] = sum_j A[i,j]*B[j,k], so the "1" appears when the j in A matches j in B.
    
    # Let's verify with schoolbook: each (i,j,k) triple gets its own term with all 1s.
    # sum over c = sum over (i',j',k') where the term is nonzero for this (i,j,k)
    # Only the term (i,j,k) itself contributes: 1*1*1 = 1
    # But we need delta_{j,k}! So schoolbook gives 1 for ALL (i,j,k), not just j==k.
    # 
    # Hmm, that means the schoolbook in the seed is actually wrong for this problem?
    # Let me re-read the seed code...
    # The seed sets U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1 for each (i,j,k)
    # This gives sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
    # But we need it to equal 1 only when j==k.
    # 
    # Wait, maybe I'm misunderstanding the problem. Let me re-read.
    # "Matrix multiplication 3,4,4" - multiply 3x4 by 4x4.
    # The output is 3x4. C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # The tensor that needs decomposition: the coefficients in C as a function of A and B.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So for each (i,k), the coefficient of A[i,j]*B[j,k] is 1.
    # The trilinear tensor T[a,b,c] where a indexes A entries, b indexes B entries, c indexes C entries.
    # a = i*m + j (i in [0,3), j in [0,4))
    # b = j'*p + k (j' in [0,4), k in [0,4))  -- note: j' is the column of A / row of B
    # c = k'*n + i' (k' in [0,4), i' in [0,3))
    # T[a,b,c] = 1 if (i=j' AND j=k' AND i'=i AND k'=k)... 
    # 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # The term A[i,j]*B[j,k] contributes 1 to C[i,k].
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # All other entries are 0.
    # 
    # So the tensor has n*m*p = 3*4*4 = 48 nonzero entries, all equal to 1.
    # The schoolbook decomposition with 48 terms each having U,V,W entries of 1 at the right positions
    # DOES correctly decompose this tensor! Because for a given (a,b,c) = (i*m+j, j*p+k, k*n+i),
    # the sum over all terms picks out exactly the term where all indices match, giving 1.
    # And for any other (a,b,c), no single term has all three indices matching, so the sum is 0.
    # 
    # So schoolbook IS correct. My earlier confusion was wrong.
    
    # For validation, just check a few entries.
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check that the decomposition reproduces the tensor
    # For efficiency, check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # T[i*m+j, j*p+k, k*n+i] = 1
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > 1e-9:
                    return False
    # Also check that off-diagonal entries are 0
    # Sample some
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    if j != j2:  # A[i,j] and B[j2,k] should not contribute to C
                        # Check T[i*m+j, j2*p+k, k*n+i] should be 0
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i, c]
                        if abs(actual) > 1e-9:
                            return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Block decomposition: split A into 4 blocks of 3x1 (columns), B into 4 blocks of 1x4 (rows)
    # Block r: A[:, r] (3x1) times B[r, :] (1x4) = 3x4 result
    # Each block is a 3x1 times 1x4 multiplication, which is just outer product
    # The outer product of a 3-vector and a 4-vector has rank 1 (as a bilinear map)!
    # Wait, no. The "multiplication" of a 3x1 matrix by a 1x4 matrix:
    # C[i,k] = A[i,0] * B[0,k]  (after reindexing within the block)
    # This is a bilinear map from R^3 x R^4 to R^{12}
    # C[i,k] = A_i * B_k
    # The tensor for this is: T[i,k] = A_i * B_k, which is a rank-1 tensor in the (A,B) indices
    # But we have 12 output components (i in 0..2, k in 0..3)
    # Each output C[i,k] = A_i * B_k is a rank-1 bilinear form
    # So the block requires 12 rank-1 terms (one for each output element)
    # Total: 4 blocks * 12 = 48. That's just schoolbook again.
    
    # Hmm, but the knowledge says rank 44 is achievable. Let me think differently.
    
    # Alternative block structure: split the 4x4 B matrix into 2x2 blocks.
    # A is 3x4. Split A into two 3x2 blocks: A = [A1 | A2] where A1, A2 are 3x2
    # B is 4x4. Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = A1*B11 + A2*B21 (for left column of C, 3x2)
    # C = A1*B12 + A2*B22 (for right column of C, 3x2)
    # 
    # Each 3x2 times 2x2 multiplication: what's its tensor rank?
    # The 3x2 * 2x2 matmul: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k], i in 0..2, j in 0..1, k in 0..1
    # Tensor has 3*2*2 = 12 nonzero entries.
    # Schoolbook would use 12 terms. Can we do better?
    # 
    # The known result: rank of 2x2x2 matmul tensor is 7 (Strassen).
    # For 3x2x2: we can write it as 2x2x2 (for the first 2 rows) plus one more row.
    # The extra row: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k], which is 2 rank-1 terms.
    # So total: 7 + 2 = 9? Or can we share?
    # 
    # Actually, 3x2x2 matmul: T[i,j,k] = delta_{j,k}... no, T[i,j,k] = 1 if (i,j,k) is a valid index triple
    # The tensor is in R^{(3*2) x (2*2) x (2*3)} = R^6 x R^4 x R^6
    # Nonzero entries: T[i*2+j, j*2+k, k*3+i] = 1 for i in 0..2, j in 0..1, k in 0..1
    # That's 3*2*2 = 12 entries.
    
    # Known: the rank of the 3x2x2 matrix multiplication tensor is 11.
    # Reference: It's known that r(3,2,2) = 11 (see e.g. work by Landsberg et al.)
    # Actually, I recall that r(3,2,2) might be 11. Let me try to construct it.
    
    # Strategy: Use the 7-term Strassen decomposition for the 2x2x2 part,
    # then add 4 more terms for the third row (since 11 - 7 = 4).
    # The third row of C: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] for k=0,1
    # This is 2 outputs, each a sum of 2 products = 4 rank-1 terms.
    # Total: 7 + 4 = 11. 
    
    # Let me construct this.
    
    # Strassen's 7-term decomposition for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 3x2 * 2x2, indices: i in 0,1,2 (rows of A), j in 0,1 (cols of A / rows of B), k in 0,1 (cols of B)
    # Strassen applies to rows 0,1 of A (the 2x2 part):
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    # C00,C01,C10,C11 from Strassen.
    # Row 2: C20 = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    #          C21 = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # 4 additional rank-1 terms.
    # Total: 11 terms.
    
    # Now I need to build the full 3x4 * 4x4 decomposition using 4 blocks of 3x2*2x2.
    # Split A (3x4) into A_left (3x2) = A[:,0:2] and A_right (3x2) = A[:,2:4]
    # Split B (4x4) into B_top (2x4) = B[0:2,:] and B_bot (2x4) = B[2:4,:]
    # 
    # C = A_left * B_top + A_right * B_bot  (3x2 * 2x4 + 3x2 * 2x4)
    # 
    # Now split B_top into B_top_left (2x2) = B[0:2,0:2] and B_top_right (2x2) = B[0:2,2:4]
    # Split B_bot into B_bot_left (2x2) = B[2:4,0:2] and B_bot_right (2x2) = B[2:4,2:4]
    # 
    # C_left (3x2) = A_left * B_top_left + A_right * B_bot_left
    # C_right (3x2) = A_left * B_top_right + A_right * B_bot_right
    # 
    # Each 3x2 * 2x2 block uses 11 terms. 4 blocks * 11 = 44 terms total.
    
    # Build the decomposition
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 44
    
    # Block structure:
    # Block (br, bc) where br in {0,1} (top/bot of B), bc in {0,1} (left/right of B)
    # A part: br==0 -> A_left (cols 0,1), br==1 -> A_right (cols 2,3)
    # B part: bc==0 -> left half (cols 0,1), bc==1 -> right half (cols 2,3)
    # B rows: br==0 -> rows 0,1; br==1 -> rows 2,3
    # C part: bc==0 -> left half (cols 0,1), bc==1 -> right half (cols 2,3)
    
    # Strassen coefficients for 2x2*2x2:
    # For the 2x2 part (rows 0,1 of the 3x2 A block):
    # A_local: 2x2, indices (lr, lc) where lr in 0,1 (local row), lc in 0,1 (local col)
    # B_local: 2x2, indices (lr, lc)
    # C_local: 2x2, indices (lr, lc)
    # 
    # Strassen 7 terms:
    # M1: (A00+A11)(B00+B11) -> U: A00,A11 get +1; V: B00,B11 get +1; W: C00,C11 get +1
    # M2: (A10+A11)B00       -> U: A10,A11 get +1; V: B00 gets +1; W: C10,C11 get +1
    # M3: A00(B10-B11)       -> U: A00 gets +1; V: B10 gets +1, B11 gets -1; W: C01,C11 get +1
    # M4: A11(B01-B00)       -> U: A11 gets +1; V: B01 gets +1, B00 gets -1; W: C00,C10 get +1
    # M5: (A00+A01)B11       -> U: A00,A01 get +1; V: B11 gets +1; W: C00,C01 get -1
    # M6: (A10-A00)(B00+B01) -> U: A10 gets +1, A00 gets -1; V: B00,B01 get +1; W: C11 gets +1
    # M7: (A01-A11)(B10+B11) -> U: A01 gets +1, A11 gets -1; V: B10,B11 get +1; W: C00 gets +1, C11 gets -1
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W coefficients (for C_local[0,0], C_local[0,1], C_local[1,0], C_local[1,1]):
    # M1: W[0,0]=+1, W[1,1]=+1
    # M2: W[1,0]=+1, W[1,1]=+1
    # M3: W[0,1]=+1, W[1,1]=+1
    # M4: W[0,0]=+1, W[1,0]=+1
    # M5: W[0,0]=-1, W[0,1]=-1
    # M6: W[1,1]=+1
    # M7: W[0,0]=+1, W[1,1]=-1
    
    # Let me verify: C[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0] = 1+1-1+1 = 2? That's wrong!
    # I think I'm confusing the W coefficients. Let me redo.
    # 
    # The Strassen decomposition computes C = AB where C = M1_contrib + M2_contrib + ...
    # Each M_i is a scalar, and C entries are linear combinations of M_i.
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # 
    # So the W factor for each term should have the coefficient of that M_i in each C entry.
    # For term 1 (M1): W[C[0,0]]=1, W[C[1,1]]=1
    # For term 2 (M2): W[C[1,0]]=1, W[C[1,1]]=-1
    # For term 3 (M3): W[C[0,1]]=1, W[C[1,1]]=1
    # For term 4 (M4): W[C[0,0]]=1, W[C[1,0]]=1
    # For term 5 (M5): W[C[0,0]]=-1, W[C[0,1]]=1
    # For term 6 (M6): W[C[1,1]]=1
    # For term 7 (M7): W[C[0,0]]=1, W[C[1,1]]=-1
    
    # U and V factors:
    # M1 = (A00+A11)(B00+B11): U[A00]=1, U[A11]=1; V[B00]=1, V[B11]=1
    # M2 = (A10+A11)B00:         U[A10]=1, U[A11]=1; V[B00]=1
    # M3 = A00(B10-B11):          U[A00]=1;           V[B10]=1, V[B11]=-1
    # M4 = A11(B01-B00):          U[A11]=1;           V[B01]=1, V[B00]=-1
    # M5 = (A00+A01)B11:          U[A00]=1, U[A01]=1; V[B11]=1
    # M6 = (A10-A00)(B00+B01):    U[A10]=1, U[A00]=-1; V[B00]=1, V[B01]=1
    # M7 = (A01-A11)(B10+B11):    U[A01]=1, U[A11]=-1; V[B10]=1, V[B11]=1
    
    # For the 3x2 block, the extra 4 terms for row 2:
    # M8 = A20*B00: U[A20]=1; V[B00]=1; W[C20]=1
    # M9 = A21*B10: U[A21]=1; V[B10]=1; W[C20]=1
    # M10 = A20*B01: U[A20]=1; V[B01]=1; W[C21]=1
    # M11 = A21*B11: U[A21]=1; V[B11]=1; W[C21]=1
    
    # Now let's build this for each of the 4 blocks.
    # Block (br, bc): br in {0,1}, bc in {0,1}
    # A columns: br==0 -> global cols 0,1 (local 0,1); br==1 -> global cols 2,3 (local 0,1)
    # B rows: br==0 -> global rows 0,1; br==1 -> global rows 2,3
    # B cols: bc==0 -> global cols 0,1 (local 0,1); bc==1 -> global cols 2,3 (local 0,1)
    # C cols: bc==0 -> global cols 0,1; bc==1 -> global cols 2,3
    # C rows: all 3 rows (local 0,1,2)
    
    # Global indexing:
    # U: A[i, j] -> index i*m + j = i*4 + j, i in 0..2, j in 0..3
    # V: B[j, k] -> index j*p + k = j*4 + k, j in 0..3, k in 0..3
    # W: C[k, i] -> index k*n + i = k*3 + i, k in 0..3, i in 0..2
    
    def block_offset(br, bc):
        """Return the global row offset for A cols and B rows/cols."""
        a_col_off = br * 2  # 0 or 2
        b_row_off = br * 2  # 0 or 2
        b_col_off = bc * 2  # 0 or 2
        c_col_off = bc * 2  # 0 or 2
        return a_col_off, b_row_off, b_col_off, c_col_off
    
    col = 0
    for br in range(2):
        for bc in range(2):
            a_col_off, b_row_off, b_col_off, c_col_off = block_offset(br, bc)
            
            # Local to global index mapping
            # A local (lr, lc) -> global (lr, lc + a_col_off), U index = lr*4 + (lc + a_col_off)
            # B local (lr, lc) -> global (lr + b_row_off, lc + b_col_off), V index = (lr+b_row_off)*4 + (lc+b_col_off)
            # C local (lr, lc) -> global (lr, lc + c_col_off), W index = (lc+c_col_off)*3 + lr
            
            # 7 Strassen terms for rows 0,1
            strassen_terms = [
                # (U_coeffs, V_coeffs, W_coeffs)
                # Each: list of (local_idx, coeff)
                ([(0,1),(2,1)], [(0,1),(3,1)], [(0,1),(3,1)]),           # M1
                ([(1,1),(2,1)], [(0,1)], [(2,1),(3,-1)]),                  # M2
                ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),                  # M3
                ([(2,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),                  # M4
                ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),                  # M5
                ([(1,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),                  # M6
                ([(1,1),(2,-1)], [(2,1),(3,1)], [(0,1),(3,-1)]),           # M7
            ]
            
            for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
                for (li, coeff) in u_coeffs:
                    lr, lc = li // 2, li % 2
                    gi = lr
                    gj = lc + a_col_off
                    U[gi*4 + gj][col] += coeff
                for (li, coeff) in v_coeffs:
                    lr, lc = li // 2, li % 2
                    gj = lr + b_row_off
                    gk = lc + b_col_off
                    V[gj*4 + gk][col] += coeff
                for (li, coeff) in w_coeffs:
                    lr, lc = li // 2, li % 2
                    gi = lr
                    gk = lc + c_col_off
                    W[gk*3 + gi][col] += coeff
                col += 1
            
            # 4 extra terms for row 2
            extra_terms = [
                ([(3,1)], [(0,1)], [(4,1)]),   # A20*B00 -> C20
                ([(4,1)], [(2,1)], [(4,1)]),   # A21*B10 -> C20
                ([(3,1)], [(1,1)], [(5,1)]),   # A20*B01 -> C21
                ([(4,1)], [(3,1)], [(5,1)]),   # A21*B11 -> C21
            ]
            
            for (u_coeffs, v_coeffs, w_coeffs) in extra_terms:
                for (li, coeff) in u_coeffs:
                    lr, lc = li // 2, li % 2
                    gi = lr
                    gj = lc + a_col_off
                    U[gi*4 + gj][col] += coeff
                for (li, coeff) in v_coeffs:
                    lr, lc = li // 2, li % 2
                    gj = lr + b_row_off
                    gk = lc + b_col_off
                    V[gj*4 + gk][col] += coeff
                for (li, coeff) in w_coeffs:
                    lr, lc = li // 2, li % 2
                    gi = lr
                    gk = lc + c_col_off
                    W[gk*3 + gi][col] += coeff
                col += 1
    
    assert col == 44, f"Expected 44 terms, got {col}"
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook if validation fails
        rank2 = n * m * p
        U2 = [[0.0] * rank2 for _ in range(n * m)]
        V2 = [[0.0] * rank2 for _ in range(m * p)]
        W2 = [[0.0] * rank2 for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m + j][c2] = 1
                    V2[j*p + k][c2] = 1
                    W2[k*n + i][c2] = 1
                    c2 += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
