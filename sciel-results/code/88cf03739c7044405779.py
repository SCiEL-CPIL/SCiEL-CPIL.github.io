"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 3x3 by 3x7 multiplication into:
    # Two 3x3 by 3x3 blocks (columns 0-2 and 3-5) and one 3x3 by 3x1 block (column 6)
    # For each 3x3x3 block, we use a block-decomposed rank-26 structure:
    # Partition A into 2x2, 2x1, 1x2, 1x1 blocks and B similarly.
    # The 2x2x2 core has rank 7 (Strassen).
    # Cross terms: a*e^T (4), A22*d (4), a*f (2), b^T*B22 (4), c*e^T (2), b^T*d (2), c*f (1)
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    
    # Strassen 2x2x2 decomposition (7 terms):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For a 3x3x3 multiplication, partition:
    # A = [[A11(2x2), a(2x1)], [b^T(1x2), c(1x1)]]
    # B = [[B11(2x2), e(2x1)], [f^T(1x2), d(2x1)]] -- wait, B is 3x3
    # B = [[B11(2x2), e(2x1)], [f(1x2), d(1x1)]] -- but d should be 2x1 for A22*d
    # Actually let me re-partition:
    # A = [[A11(2x2), a(2x1)], [b^T(1x2), c]]
    # B = [[B11(2x2), e(2x1)], [f^T(1x2), d(2x1)]] -- no, B is 3x3 so bottom-right is 1x1
    # Let me redo:
    # A is 3x3: top-left 2x2 = A11, top-right 2x1 = a, bottom-left 1x2 = b^T, bottom-right 1x1 = c
    # B is 3x3: top-left 2x2 = B11, top-right 2x1 = e, bottom-left 1x2 = f^T, bottom-right 1x1 = d
    # C = A*B, 3x3:
    # C11(2x2) = A11*B11 + a*f^T
    # C12(2x1) = A11*e + a*d
    # C21(1x2) = b^T*B11 + c*f^T
    # C22(1x1) = b^T*e + c*d
    
    # Rank-1 terms:
    # 1. A11*B11: rank 7 (Strassen)
    # 2. a*f^T: a is 2x1, f^T is 1x2, so 2x2. Rank 4 (schoolbook: a1*f1, a1*f2, a2*f1, a2*f2)
    # 3. A11*e: 2x2 by 2x1. Rank 4 (schoolbook)
    # 4. a*d: 2x1 by 1x1. Rank 2
    # 5. b^T*B11: 1x2 by 2x2. Rank 4 (schoolbook)
    # 6. c*f^T: 1x1 by 1x2. Rank 2
    # 7. b^T*e: 1x2 by 2x1. Rank 2 (dot product)
    # 8. c*d: 1x1 by 1x1. Rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    
    # For each 3x3x3 block, we need to map the 26 terms to U, V, W vectors.
    # U has dimension n*m = 9 (indices i*m+j)
    # V has dimension m*p = 21 (indices j*p+k)
    # W has dimension p*n = 21 (indices k*n+i)
    
    # Let's define the 26 terms for one 3x3x3 block with column offset col_offset for k indices.
    
    def build_3x3x3_terms(col_offset):
        """Build the 26 (U_index, V_index, W_index) tuples for a 3x3x3 block."""
        terms = []
        
        # Strassen 2x2x2 for A11*B11 (rows 0-1, cols 0-1 of A; rows 0-1, cols 0-1 of B)
        # A11 entries: (0,0),(0,1),(1,0),(1,1) -> U indices: 0,1,3,4
        # B11 entries: (0,0),(0,1),(1,0),(1,1) -> V indices: 0, p, p+1, 2p (with col_offset)
        # C11 entries: (0,0),(0,1),(1,0),(1,1) -> W indices: 0, n, n+1, 2n (with col_offset)
        
        # Strassen terms for 2x2:
        # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
        # In our notation, A11 top-left is (0,0),(0,1),(1,0),(1,1)
        # A22 bottom-right is... wait, in the 2x2 subproblem:
        # A_sub = [[A00, A01],[A10, A11]] corresponding to A[0:2,0:2]
        # B_sub = [[B00, B01],[B10, B11]] corresponding to B[0:2,0:2]
        # C_sub = A_sub * B_sub, 2x2
        
        # Strassen:
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B10-B11)
        # P4 = A11*(B01-B00)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # Mapping to our indices:
        # A00 = A[0,0] -> U[0], A01 = A[0,1] -> U[1], A10 = A[1,0] -> U[3], A11 = A[1,1] -> U[4]
        # B00 = B[0,0] -> V[0+col_offset*p], B01 = B[0,1] -> V[p+col_offset*p]
        # B10 = B[1,0] -> V[p+col_offset*p + 0]... wait, V index is j*p+k
        # For B[j,k], V index = j*p + k. With col_offset, k becomes k + col_offset.
        # So B[0,0] -> V[0*p + (0+col_offset)] = V[col_offset]
        # B[0,1] -> V[0*p + (1+col_offset)] = V[1+col_offset]
        # B[1,0] -> V[1*p + (0+col_offset)] = V[p+col_offset]
        # B[1,1] -> V[1*p + (1+col_offset)] = V[p+1+col_offset]
        # C[i,j] -> W[j*n + i]. With col_offset, j becomes j+col_offset.
        # C[0,0] -> W[(0+col_offset)*n + 0] = W[col_offset*n]
        # C[0,1] -> W[(1+col_offset)*n + 0] = W[(1+col_offset)*n]
        # C[1,0] -> W[(0+col_offset)*n + 1] = W[col_offset*n + 1]
        # C[1,1] -> W[(1+col_offset)*n + 1] = W[(1+col_offset)*n + 1]
        
        # P1: (A00+A11)*(B00+B11) -> contributes to C00 (+1) and C11 (+1)
        # This is 2 rank-1 terms? No, in tensor decomposition, each "multiplication" is a rank-1 term.
        # P1 involves computing (A00+A11) and (B00+B11) as intermediate products, then multiplying them.
        # In the tensor decomposition, each rank-1 term is: U_vector * V_vector * W_vector
        # where U_vector picks out certain A entries, V_vector picks out certain B entries,
        # and W_vector places the product into certain C entries.
        
        # P1: U has 1 at A00 and A11, V has 1 at B00 and B11, W has 1 at C00 and C11
        # But this is NOT a single rank-1 term because the product (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # which is a sum of 4 products. In tensor decomposition, a rank-1 term is a single product of one A entry, one B entry, placed in one C entry.
        
        # Wait, I'm confusing myself. In tensor decomposition of matrix multiplication:
        # The tensor T_{ijk} = 1 if C_{ij} = sum_k A_{ik} B_{kj}
        # A rank-1 term is: sum_i u_i * sum_j v_j * sum_k w_k where the product u_i * v_j * w_k contributes to T.
        # Actually, a rank-1 tensor is: T = sum_r (sum_i U_{ir} A_{i}) (sum_j V_{jr} B_{j}) (sum_k W_{kr} C_{k})
        # No, that's not right either.
        
        # The standard formulation: C = A*B means C_{ij} = sum_k A_{ik} B_{kj}
        # Tensor T has T_{ijk} = 1 for each (i,j,k) triple, and the multiplication is:
        # sum_{i,j,k} T_{ijk} A_{ik} B_{kj} = sum_{i,j} C_{ij}
        # A rank-r decomposition: T = sum_{r=1}^R sum_{i,j,k} U_{ir} V_{jr} W_{kr} e_i e_j e_k
        # which means: sum_{i,j,k} (sum_r U_{ir} V_{jr} W_{kr}) A_{ik} B_{kj} = sum_{i,j} C_{ij}
        # For this to equal sum_{i,j} sum_k A_{ik} B_{kj}, we need:
        # sum_r U_{ir} V_{jr} W_{kr} = 1 for each (i,j,k) where the term A_{ik}B_{kj} appears in C_{ij}
        # i.e., for each (i,j,k): sum_r U_{ir} V_{jr} W_{kr} = 1
        # And for (i,j,k) where it doesn't appear (which is never, since all k contribute), it's fine.
        
        # So each rank-1 term r contributes U_{ir} * V_{jr} * W_{kr} to the coefficient of A_{ik} B_{kj} in C_{ij}.
        # For schoolbook: for each (i,j,k), one term with U_{ir}=1, V_{jr}=1, W_{kr}=1.
        # For Strassen: the 7 terms have linear combinations.
        
        # Let me just enumerate the 26 terms explicitly with their U, V, W contributions.
        
        # Term indices within the 3x3x3 block:
        # Terms 0-6: Strassen for A11*B11 (2x2x2)
        # Terms 7-10: a*f^T (2x1 * 1x2 = 2x2), 4 terms
        # Terms 11-14: A11*e (2x2 * 2x1 = 2x1), 4 terms
        # Terms 15-16: a*d (2x1 * 1x1 = 2x1), 2 terms
        # Terms 17-20: b^T*B11 (1x2 * 2x2 = 1x2), 4 terms
        # Terms 21-22: c*f^T (1x1 * 1x2 = 1x2), 2 terms
        # Term 23-24: b^T*e (1x2 * 2x1 = 1x1), 2 terms
        # Term 25: c*d (1x1 * 1x1 = 1x1), 1 term
        
        # Strassen terms (0-6):
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
        # But this single "term" in Strassen's algorithm is actually the product of two sums,
        # which in the tensor decomposition is represented as ONE rank-1 term where:
        # U vector has 1 at positions corresponding to A00 and A11
        # V vector has 1 at positions corresponding to B00 and B11
        # W vector has 1 at positions corresponding to C00 and C11
        # The contribution to T is: U_i * V_j * W_k for all i,j,k
        # This gives: for i in {0,1} (A rows), j in {0,1} (B cols), k in {0,1} (C cols)... 
        # Hmm, I think I need to be more careful.
        
        # Actually, in the tensor decomposition, a rank-1 term is defined by vectors u (dim n*m), v (dim m*p), w (dim p*n).
        # The term contributes u_i * v_j * w_k to T_{(i,j,k)} where i indexes A, j indexes B, k indexes C.
        # Wait no. Let me re-read the problem.
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # The decomposition satisfies: for all (i,j,k) in the tensor:
        # T_{ijk} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
        # where T_{ijk} = 1 if the (i,k) entry of A and (k,j) entry of B multiply to contribute to (i,j) entry of C.
        # Actually, T_{ijk} = 1 for all valid (i,j,k) since C_{ij} = sum_k A_{ik} B_{kj}.
        
        # So for each rank r, we have vectors U_r (dim 9), V_r (dim 21), W_r (dim 21).
        # The constraint is: for each (i,j,k) with i in [0,3), j in [0,3), k in [0,7):
        # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1
        
        # For Strassen 2x2x2, the 7 rank-1 terms have:
        # Term 0 (P1): U has 1 at A[0,0] and A[1,1]; V has 1 at B[0,0] and B[1,1]; W has 1 at C[0,0] and C[1,1]
        # But wait, this doesn't work because U_{i,j} * V_{j,k} * W_{k,i} for i,j,k in the 2x2 subproblem.
        # If U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1):
        # For (i,j,k) = (0,0,0): U_{0,0}=1, V_{0,0}=1, W_{0,0}=1 -> contributes 1
        # For (i,j,k) = (0,0,1): U_{0,0}=1, V_{0,1}=0, W_{1,0}=0 -> contributes 0
        # This is wrong. Strassen terms don't work this way.
        
        # I think the issue is that Strassen's algorithm uses intermediate products, and each intermediate product is a rank-1 term in the decomposition where the U and V vectors can have multiple non-zero entries, but the W vector places the result.
        
        # Let me look at this differently. For 2x2x2 Strassen:
        # P1 = (A00+A11)(B00+B11)
        # This is a single multiplication of two linear combinations. In the tensor decomposition, this corresponds to ONE rank-1 term where:
        # U vector: 1 at A00, 1 at A11
        # V vector: 1 at B00, 1 at B11
        # W vector: 1 at C00, 1 at C11
        # The contribution to T_{ijk} for this term is U_i * V_j * W_k.
        # For the 2x2 subproblem with indices 0,1:
        # T_{0,0,0} gets U_0 * V_0 * W_0 = 1*1*1 = 1
        # T_{0,0,1} gets U_0 * V_1 * W_1 = 1*0*0 = 0 (V_1 is B01, which is 0 in this term)
        # Hmm, this still doesn't seem right.
        
        # I think I'm overcomplicating this. Let me just use schoolbook for the 3x3x3 blocks but with a better partitioning.
        # Actually, the simplest correct approach is to just use schoolbook for everything, which gives rank 63.
        # To improve, I can use the fact that the 3x1 block can be done more efficiently.
        # But the 3x1 block is already rank 9 with schoolbook, which is optimal (n*m*p = 3*3*1 = 9).
        
        # Let me just return the schoolbook decomposition with rank 63, which is what the current program does.
        # Since I can't easily implement Strassen correctly, I'll stick with schoolbook.
        
        return []
    
    # Given the complexity of implementing Strassen correctly in this tensor decomposition format,
    # and the risk of errors, I'll use the schoolbook decomposition which is known to be correct.
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
