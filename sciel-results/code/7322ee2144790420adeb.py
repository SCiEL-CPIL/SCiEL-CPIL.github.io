"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_block_decomposition(n, m, p):
    """
    Build a decomposition for n x m by m x p multiplication using block structure.
    We split the m=4 dimension into two blocks of size 2.
    We split the p=7 dimension into blocks: 2+2+2+1 (or similar).
    Actually, let's just do a general block decomposition.
    
    Let m = sum of m_blocks.
    Let p = sum of p_blocks.
    Let n = sum of n_blocks.
    
    For each block combination (i_block, j_block, k_block), we compute a sub-matrix.
    But the standard way is to treat the multiplication as a sum of outer products of blocks.
    
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    If we partition rows of A into I, cols of A (and rows of B) into J, and cols of B into K.
    Then C_{I,K} = sum_{J} A_{I,J} B_{J,K}.
    
    The rank of a block multiplication A_{I,J} * B_{J,K} is rank_min(I*J*K) if we treat it as a tensor?
    No, the rank of the bilinear map for a specific block (I,J,K) is the rank of the tensor for that block.
    If we just use schoolbook for each block, the rank is I*J*K.
    Sum of ranks is the total rank.
    
    To improve, we need to use a lower rank decomposition for the sub-blocks if possible.
    For 2x2x2, rank is 7 (Strassen).
    For 2x2x3, rank is ?
    For 2x2x4, rank is ?
    
    Let's try to use Strassen for 2x2x2 blocks.
    Dimensions: n=2, m=4, p=7.
    
    Split m=4 into [2, 2].
    Split p=7 into [2, 2, 3] or [3, 2, 2] or [2, 2, 2, 1].
    
    Let's try splitting m into [2, 2] and p into [2, 2, 2, 1].
    This gives 4 blocks in the m direction? No, 2 blocks in m.
    So we have blocks:
    (i=0..1, j=0..1, k=0..1) -> 2x2x2 -> rank 7
    (i=0..1, j=0..1, k=2..3) -> 2x2x2 -> rank 7
    (i=0..1, j=0..1, k=4..5) -> 2x2x2 -> rank 7
    (i=0..1, j=0..1, k=6..6) -> 2x2x1 -> rank 4 (schoolbook) or 3?
       2x2x1: C_ik = A_i0 B_0k + A_i1 B_1k.
       This is sum of two rank 1 tensors?
       T = e_i0 ⊗ e_0k ⊗ e_ki + e_i1 ⊗ e_1k ⊗ e_ki.
       Can we do better?
       Rank of 2x2x1 tensor.
       It is the identity map on the k dimension, tensored with the 2x2 multiplication?
       No, it's C_i,k = sum_j A_i,j B_j,k.
       Fixing k, it's a 2x2 multiplication.
       The tensor is sum_{k} (T_2x2x1_k).
       Since the k-dimension is size 1, the tensor is just the 2x2x1 tensor.
       Rank of 2x2x1 tensor:
       It is a 2x2 matrix (A) times a 2x1 matrix (B column).
       The bilinear map is A -> A * b.
       This is linear in A and b.
       The rank is the minimum number of rank 1 tensors.
       T = sum_{j} e_j (in A) ⊗ e_j (in B) ⊗ ...?
       Actually, T_{i,j,k} = 1 if k=0? No, k is the output index.
       T_{i,j,k} = delta_{k,0} * 1? No.
       The tensor is M_{i,j,k} = 1 for the multiplication.
       For 2x2x1, M_{i,j,0} = 1? No.
       C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}.
       So M_{i,j,0} = 1.
       This is a 2x2x1 tensor with all 1s.
       Rank of all-1 tensor of size IxJxK is I*J*K? No, it's 1 if all dims > 1?
       No, the all-1 tensor is rank 1?
       Let's check. T_{ijk} = 1.
       T = u ⊗ v ⊗ w where u_i=1, v_j=1, w_k=1.
       Yes, rank 1.
       Wait, is the matrix multiplication tensor all 1s?
       No!
       The schoolbook code sets U, V, W such that the sum of outer products equals the tensor.
       For the schoolbook, each term is a rank 1 tensor with 1s at specific positions.
       The tensor itself is defined by the bilinear map.
       The bilinear map is (A,B) -> AB.
       The tensor M is such that M(A,B) = sum_{i,j,k} M_{ijk} A_{ij} B_{jk} e_{ik}.
       So M_{ijk} is the coefficient of A_{ij} B_{jk} in the output (AB)_{ik}.
       (AB)_{ik} = sum_j A_{ij} B_{jk}.
       So the coefficient of A_{ij} B_{jk} is 1.
       So M_{ijk} = 1 for all i,j,k?
       Yes, the matrix multiplication tensor is the all-1 tensor of size n x m x p?
       No, the indices are i (0..n-1), j (0..m-1), k (0..p-1).
       The term A_{ij} B_{jk} contributes to C_{ik}.
       So M_{ijk} = 1.
       
       If M_{ijk} = 1, then the rank is 1?
       T = u ⊗ v ⊗ w with u_i=1, v_j=1, w_k=1.
       Then T_{ijk} = 1.
       So the rank of the matrix multiplication tensor is 1?
       That can't be right. Strassen's algorithm exists.
       
       Ah, the tensor is not just the values. The tensor represents the bilinear map.
       The bilinear map is a linear map from V* ⊗ V* to W*.
       The tensor is in V ⊗ V ⊗ W*.
       The components M_{ijk} are the coefficients.
       If M_{ijk} = 1, then T = (sum e_i) ⊗ (sum e_j) ⊗ (sum e_k).
       This tensor computes (sum A_{ij}) (sum B_{jk}) (sum e_k).
       This is NOT matrix multiplication.
       
       The correct tensor is:
       M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k^* ?
       No.
       The bilinear form is B(A,B) = sum_{i,k} (sum_j A_{ij} B_{jk}) e_{ik}.
       = sum_{i,j,k} A_{ij} B_{jk} e_{ik}.
       So the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
       Wait, e_i is in the output space?
       The tensor T is such that <T, A ⊗ B> = AB.
       T = sum_{i,j,k} e_{i}^{out} ⊗ e_{j}^{in1} ⊗ e_{k}^{in2}.
       So T_{ijk} = 1.
       
       Why is the rank not 1?
       Because the rank is the minimum number of rank-1 tensors that sum to T.
       A rank-1 tensor is u ⊗ v ⊗ w.
       (u ⊗ v ⊗ w)_{ijk} = u_i v_j w_k.
       If T_{ijk} = 1, then we need u_i v_j w_k = 1 for all i,j,k.
       This is possible with u_i=1, v_j=1, w_k=1.
       So the rank is 1?
       
       I am missing something fundamental.
       The matrix multiplication tensor is NOT the all-1 tensor.
       The indices i, j, k are coupled.
       i is the row of A, j is the col of A / row of B, k is the col of B.
       The output is C_{ik}.
       So the tensor is T = sum_{i,j,k} e_i (output) ⊗ e_j (input A) ⊗ e_k (input B).
       Wait, the input A is indexed by (i,j). The input B is indexed by (j,k).
       So the input space is not a simple product of 3 independent indices.
       The input space is V_A (size nm) and V_B (size mp).
       The tensor is in V_A^* ⊗ V_B^* ⊗ C^*.
       The component T_{(i,j), (j,k), (i,k)} = 1.
       But T is a 3-way tensor. The indices are:
       1. Index of A: a = i*m + j.
       2. Index of B: b = j*p + k.
       3. Index of C: c = k*n + i.
       
       So T_{a,b,c} is 1 if a, b, c are consistent with the multiplication rule.
       Specifically, a = i*m + j, b = j*p + k, c = k*n + i.
       Note that j is shared between a and b. i is shared between a and c. k is shared between b and c.
       
       This is NOT the all-1 tensor in the (a,b,c) space.
       It is a sparse tensor.
       For a fixed a (i,j), and b (j',k), and c (k',i'),
       T_{a,b,c} = 1 if j=j' and i=i' and k=k'.
       Otherwise 0.
       
       So the tensor is the "identity" in the sense of matching indices.
       This is why the rank is high.
       
       Okay, so we can't just use rank 1.
       
       Back to block decomposition.
       We can decompose the tensor into blocks based on the indices.
       If we partition the indices, we can use known decompositions for the blocks.
       
       Let's use the block decomposition with Strassen for 2x2x2.
       n=2, m=4, p=7.
       Split m=4 into [2,2].
       Split p=7 into [2,2,2,1].
       
       Blocks:
       1. i in [0,1], j in [0,1], k in [0,1] -> 2x2x2 block. Rank 7.
       2. i in [0,1], j in [0,1], k in [2,3] -> 2x2x2 block. Rank 7.
       3. i in [0,1], j in [0,1], k in [4,5] -> 2x2x2 block. Rank 7.
       4. i in [0,1], j in [0,1], k in [6,6] -> 2x2x1 block. Rank 4 (schoolbook).
       5. i in [0,1], j in [2,3], k in [0,1] -> 2x2x2 block. Rank 7.
       6. i in [0,1], j in [2,3], k in [2,3] -> 2x2x2 block. Rank 7.
       7. i in [0,1], j in [2,3], k in [4,5] -> 2x2x2 block. Rank 7.
       8. i in [0,1], j in [2,3], k in [6,6] -> 2x2x1 block. Rank 4.
       
       Total rank = 6*7 + 2*4 = 42 + 8 = 50.
       
       Can we do better?
       Maybe combine blocks?
       Or use a different decomposition for 2x2x1?
       2x2x1 tensor:
       T_{i,j,k} = 1 if k=0? No, k is the index in the block.
       In the block, the indices are local.
       The block tensor is the same structure as the full tensor, but smaller.
       So 2x2x1 tensor has rank 4?
       Let's check.
       T = sum_{i,j} e_i ⊗ e_j ⊗ e_0.
       This is (sum_i e_i) ⊗ (sum_j e_j) ⊗ e_0.
       This is rank 1!
       Wait.
       In the block, the output index is only k=0 (local).
       The input A is 2x2. Input B is 2x1.
       C = A * B.
       C_i = A_i0 B_0 + A_i1 B_1.
       T = sum_{i} (A_i0 B_0 + A_i1 B_1) e_i.
       = B_0 sum_i A_i0 e_i + B_1 sum_i A_i1 e_i.
       This is a sum of 2 terms?
       Term 1: (sum_i A_i0 e_i) * B_0.
       This is not a rank 1 tensor in the (A,B) space.
       The tensor is T(A,B) = C.
       T is a trilinear form.
       T = sum_{i,j} A_{ij} B_j e_i.
       = sum_{i,j} e_i ⊗ e_{ij} (in A) ⊗ e_j (in B).
       
       Let's use the standard basis.
       A has basis e_{ij}. B has basis e_{jk}. C has basis e_{ik}.
       T = sum_{i,j,k} e_{ik} ⊗ e_{ij} ⊗ e_{jk}.
       
       For 2x2x1:
       i=0,1. j=0,1. k=0.
       T = sum_{i,j} e_{i0} ⊗ e_{ij} ⊗ e_{j0}.
       = e_{00} ⊗ e_{00} ⊗ e_{00} + e_{00} ⊗ e_{01} ⊗ e_{10} + e_{10} ⊗ e_{10} ⊗ e_{00} + e_{10} ⊗ e_{11} ⊗ e_{10}.
       
       Can we factor this?
       Let U, V, W be matrices.
       T = sum_c u_c ⊗ v_c ⊗ w_c.
       
       If we use schoolbook, rank is 4.
       Is rank 3 possible?
       I suspect rank 4 is minimal for 2x2x1.
       
       So total rank 50.
       
       Can we reduce further?
       Maybe by combining the two 2x2x1 blocks?
       They are at k=6.
       One is for j=0,1. One is for j=2,3.
       They are disjoint in the j-index.
       So they can be treated independently.
       
       What if we split p=7 into [3,2,2]?
       Then we have 2x2x3 blocks.
       Rank of 2x2x3?
       Schoolbook is 12.
       Maybe 10?
       
       Let's stick with 50 for now.
       50 is better than 56.
       
       Implementation:
       We need to construct U, V, W for the full problem.
       We will build the decomposition for each block and then sum them up (block diagonal in the rank dimension).
       
       For each block (i_range, j_range, k_range):
       We have a local decomposition with rank r_block.
       We need to map the local indices to global indices.
       
       Local U: size (n_block * m_block) x r_block.
       Local V: size (m_block * p_block) x r_block.
       Local W: size (p_block * n_block) x r_block.
       
       Global U: size (n*m) x Rank_total.
       Global V: size (m*p) x Rank_total.
       Global W: size (p*n) x Rank_total.
       
       We will allocate columns in the global matrices for each block.
       
       We need a function to get the Strassen decomposition for 2x2x2.
       And a function for 2x2x1 (schoolbook).
       
       Let's write the Strassen 2x2x2 decomposition.
       Standard Strassen:
       A = [[a, b], [c, d]]
       B = [[e, f], [g, h]]
       
       M1 = (a+d)(e+h)
       M2 = (c+d)e
       M3 = a(f-h)
       M4 = d(g-e)
       M5 = (a+b)h
       M6 = (c-a)(e+f)
       M7 = (b-d)(g+h)
       
       C11 = M1 + M4 - M5 + M7
       C12 = M3 + M5
       C21 = M2 + M4
       C22 = M1 - M2 + M3 + M6
       
       The 7 multiplications are:
       1. (a+d) * (e+h)
       2. (c+d) * e
       3. a * (f-h)
       4. d * (g-e)
       5. (a+b) * h
       6. (c-a) * (e+f)
       7. (b-d) * (g+h)
       
       Each multiplication is a rank 1 tensor in the (A,B,C) space?
       No, the multiplication is of two linear combinations.
       So each term is a product of a linear form in A and a linear form in B.
       So it is a rank 1 tensor in the (A,B) space, mapped to a linear form in C.
       
       So for each Strassen term, we have:
       u = linear_form_in_A
       v = linear_form_in_B
       w = linear_form_in_C (which is just the output entry, e.g., C11)
       
       Actually, the term contributes to multiple C entries.
       So w is a vector in the C space.
       
       For term 1:
       u = a+d (coeff 1 for A00, 1 for A11)
       v = e+h (coeff 1 for B00, 1 for B11)
       w = C11 - C12 - C21 + C22?
       Let's check the contributions.
       M1 contributes to C11, C22.
       C11 += M1
       C22 += M1
       
       So w has 1 at C11, 1 at C22.
       
       So we can construct U, V, W for the 2x2x2 block.
       
       Let's define the indices for 2x2x2.
       A: 0,1,2,3 (00, 01, 10, 11)
       B: 0,1,2,3 (00, 01, 10, 11)
       C: 0,1,2,3 (00, 01, 10, 11)
       
       Strassen terms:
       1. u = [1, 0, 0, 1], v = [1, 0, 0, 1], w = [1, 0, 0, 1]
       2. u = [0, 0, 1, 1], v = [1, 0, 0, 0], w = [0, 0, 1, 0]
       3. u = [1, 0, 0, 0], v = [0, 1, 0, -1], w = [0, 1, 0, 1]
       4. u = [0, 0, 0, 1], v = [0, 0, 1, -1], w = [1, 0, 1, 0]
       5. u = [1, 1, 0, 0], v = [0, 0, 0, 1], w = [-1, 1, 0, 0]
       6. u = [0, 0, 1, -1], v = [1, 1, 0, 0], w = [0, 0, 0, 1]
       7. u = [0, 1, 0, -1], v = [0, 0, 1, 1], w = [1, 0, 0, -1]
       
       Let's verify term 1.
       M1 = (A00+A11)(B00+B11).
       Contributes to C00 and C11.
       C00 = ... + M1
       C11 = ... + M1
       So w = [1, 0, 0, 1]. Correct.
       
       Term 2: M2 = (A10+A11)B00.
       Contributes to C10.
       C10 = M2 + M4.
       So w = [0, 0, 1, 0]. Correct.
       
       Term 3: M3 = A00(B01-B11).
       Contributes to C01 and C11?
       C01 = M3 + M5.
       C11 = M1 - M2 + M3 + M6.
       So w = [0, 1, 0, 1]. Correct.
       
       Term 4: M4 = A11(B10-B00).
       Contributes to C00 and C10.
       C00 = M1 + M4 - M5 + M7.
       C10 = M2 + M4.
       So w = [1, 0, 1, 0]. Correct.
       
       Term 5: M5 = (A00+A01)B11.
       Contributes to C00 and C01.
       C00 = ... - M5 ...
       C01 = M3 + M5.
       So w = [-1, 1, 0, 0]. Correct.
       
       Term 6: M6 = (A10-A00)(B00+B01).
       Contributes to C11.
       C11 = ... + M6.
       So w = [0, 0, 0, 1]. Correct.
       
       Term 7: M7 = (A01-A11)(B10+B11).
       Contributes to C00 and C11.
       C00 = ... + M7.
       C11 = ... - M7.
       So w = [1, 0, 0, -1]. Correct.
       
       So we have the 2x2x2 decomposition.
       
       For 2x2x1:
       A: 0,1,2,3. B: 0,1 (since p=1, B is 2x1, indices 0,1). C: 0,1 (n=2, p=1, C is 2x1, indices 0,1).
       Wait, C index is k*n+i. k=0. i=0,1. So C indices 0,1.
       B index is j*p+k. k=0. j=0,1. So B indices 0,1.
       A index is i*m+j. m=2. i=0,1. j=0,1. A indices 0,1,2,3.
       
       Schoolbook:
       C0 = A00 B0 + A01 B1
       C1 = A10 B0 + A11 B1
       
       Terms:
       1. u = [1, 0, 0, 0], v = [1, 0], w = [1, 0]
       2. u = [0, 1, 0, 0], v = [0, 1], w = [1, 0]
       3. u = [0, 0, 1, 0], v = [1, 0], w = [0, 1]
       4. u = [0, 0, 0, 1], v = [0, 1], w = [0, 1]
       
       Rank 4.
       
       Now, we combine these blocks.
       
       Blocks:
       (i=0,1), (j=0,1), (k=0,1) -> 2x2x2. Rank 7.
       (i=0,1), (j=0,1), (k=2,3) -> 2x2x2. Rank 7.
       (i=0,1), (j=0,1), (k=4,5) -> 2x2x2. Rank 7.
       (i=0,1), (j=0,1), (k=6,6) -> 2x2x1. Rank 4.
       (i=0,1), (j=2,3), (k=0,1) -> 2x2x2. Rank 7.
       (i=0,1), (j=2,3), (k=2,3) -> 2x2x2. Rank 7.
       (i=0,1), (j=2,3), (k=4,5) -> 2x2x2. Rank 7.
       (i=0,1), (j=2,3), (k=6,6) -> 2x2x1. Rank 4.
       
       Total rank 50.
       
       We need to map local indices to global.
       
       Local A index a_loc = i_loc * m_loc + j_loc.
       Global A index a_glob = i_glob * m + j_glob.
       i_glob = i_offset + i_loc.
       j_glob = j_offset + j_loc.
       
       Local B index b_loc = j_loc * p_loc + k_loc.
       Global B index b_glob = j_glob * p + k_glob.
       j_glob = j_offset + j_loc.
       k_glob = k_offset + k_loc.
       
       Local C index c_loc = k_loc * n_loc + i_loc.
       Global C index c_glob = k_glob * n + i_glob.
       k_glob = k_offset + k_loc.
       i_glob = i_offset + i_loc.
       
       We will create the global U, V, W matrices and fill them in.
       
       """
    n, m, p = 2, 4, 7
    
    # Define blocks
    # i_blocks: [(0, 2)]
    # j_blocks: [(0, 2), (2, 4)]
    # k_blocks: [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    i_blocks = [(0, 2)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    total_rank = 0
    for i_off, i_size in [(b[0], b[1]-b[0]) for b in i_blocks]:
        for j_off, j_size in [(b[0], b[1]-b[0]) for b in j_blocks]:
            for k_off, k_size in [(b[0], b[1]-b[0]) for b in k_blocks]:
                if j_size == 2 and k_size == 2:
                    total_rank += 7
                elif j_size == 2 and k_size == 1:
                    total_rank += 4
                else:
                    total_rank += j_size * k_size # fallback
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    for i_off, i_size in [(b[0], b[1]-b[0]) for b in i_blocks]:
        for j_off, j_size in [(b[0], b[1]-b[0]) for b in j_blocks]:
            for k_off, k_size in [(b[0], b[1]-b[0]) for b in k_blocks]:
                # Get local decomposition
                if j_size == 2 and k_size == 2:
                    local_U, local_V, local_W, local_rank = _strassen_2x2x2()
                elif j_size == 2 and k_size == 1:
                    local_U, local_V, local_W, local_rank = _schoolbook_2x2x1()
                else:
                    local_U, local_V, local_W, local_rank = _schoolbook_block(i_size, j_size, k_size)
                
                # Map to global
                for c in range(local_rank):
                    global_c = rank_offset + c
                    for a_loc in range(i_size * j_size):
                        i_loc = a_loc // j_size
                        j_loc = a_loc % j_size
                        a_glob = (i_off + i_loc) * m + (j_off + j_loc)
                        U[a_glob][global_c] = local_U[a_loc][c]
                    
                    for b_loc in range(j_size * k_size):
                        j_loc = b_loc // k_size
                        k_loc = b_loc % k_size
                        b_glob = (j_off + j_loc) * p + (k_off + k_loc)
                        V[b_glob][global_c] = local_V[b_loc][c]
                    
                    for c_loc in range(k_size * i_size):
                        k_loc = c_loc // i_size
                        i_loc = c_loc % i_size
                        c_glob = (k_off + k_loc) * n + (i_off + i_loc)
                        W[c_glob][global_c] = local_W[c_loc][c]
                
                rank_offset += local_rank
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    # Returns U, V, W, rank
    # U: 4x7, V: 4x7, W: 4x7
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ]
    # Wait, let's re-verify the W matrix from the derivation.
    # Term 1: w = [1, 0, 0, 1]
    # Term 2: w = [0, 0, 1, 0]
    # Term 3: w = [0, 1, 0, 1]
    # Term 4: w = [1, 0, 1, 0]
    # Term 5: w = [-1, 1, 0, 0]
    # Term 6: w = [0, 0, 0, 1]
    # Term 7: w = [1, 0, 0, -1]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, -1]
    ]
    # This looks correct.
    return U, V, W, 7

def _schoolbook_2x2x1():
    # i=0,1, j=0,1, k=0
    # A: 4, B: 2, C: 2
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W, 4

def _schoolbook_block(n_block, m_block, p_block):
    rank = n_block * m_block * p_block
    U = [[0] * rank for _ in range(n_block * m_block)]
    V = [[0] * rank for _ in range(m_block * p_block)]
    W = [[0] * rank for _ in range(p_block * n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i * m_block + j][col] = 1
                V[j * p_block + k][col] = 1
                W[k * n_block + i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = 150
    margin = 5
    
    try:
        return _build_block_decomposition(2, 4, 7)
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
