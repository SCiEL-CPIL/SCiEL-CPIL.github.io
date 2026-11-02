"""Optimized tensor decomposition for 3x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# Strassen's algorithm for 2x2x2 matrix multiplication
def strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    Convention: W[k*n+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B11 - B01)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c]: A coefficient
    # V[j*p+k, c]: B coefficient
    # W[k*n+i, c]: C coefficient
    
    # M1: (A00 + A11) * (B00 + B11)
    # A00 = i=0,j=0 -> U[0, 0]
    # A11 = i=1,j=1 -> U[3, 0]
    # B00 = j=0,k=0 -> V[0, 0]
    # B11 = j=1,k=1 -> V[3, 0]
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    # C00 += M1 -> W[0*2+0, 0] = W[0, 0]
    # C11 += M1 -> W[1*2+1, 0] = W[3, 0]
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2: (A10 + A11) * B00
    # A10 = i=1,j=0 -> U[2, 1]
    # A11 = i=1,j=1 -> U[3, 1]
    # B00 = j=0,k=0 -> V[0, 1]
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    # C10 += M2 -> W[0*2+1, 1] = W[1, 1]
    # C11 -= M2 -> W[1*2+1, 1] = W[3, 1]
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3: A00 * (B10 - B00)
    # A00 = i=0,j=0 -> U[0, 2]
    # B10 = j=1,k=0 -> V[2, 2]
    # B00 = j=0,k=0 -> V[0, 2]
    U[0, 2] = 1
    V[2, 2] = 1
    V[0, 2] = -1
    # C01 += M3 -> W[0*2+0, 2] = W[0, 2]
    # C11 += M3 -> W[1*2+1, 2] = W[3, 2]
    W[0, 2] = 1
    W[3, 2] = 1
    
    # M4: A11 * (B11 - B01)
    # A11 = i=1,j=1 -> U[3, 3]
    # B11 = j=1,k=1 -> V[3, 3]
    # B01 = j=0,k=1 -> V[1, 3]
    U[3, 3] = 1
    V[3, 3] = 1
    V[1, 3] = -1
    # C00 += M4 -> W[0*2+0, 3] = W[0, 3]
    # C10 += M4 -> W[0*2+1, 3] = W[1, 3]
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5: (A00 + A01) * B11
    # A00 = i=0,j=0 -> U[0, 4]
    # A01 = i=0,j=1 -> U[1, 4]
    # B11 = j=1,k=1 -> V[3, 4]
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    # C00 -= M5 -> W[0*2+0, 4] = W[0, 4]
    # C01 += M5 -> W[1*2+0, 4] = W[2, 4]
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6: (A10 - A00) * (B00 + B01)
    # A10 = i=1,j=0 -> U[2, 5]
    # A00 = i=0,j=0 -> U[0, 5]
    # B00 = j=0,k=0 -> V[0, 5]
    # B01 = j=0,k=1 -> V[1, 5]
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    # C11 += M6 -> W[1*2+1, 5] = W[3, 5]
    W[3, 5] = 1
    
    # M7: (A01 - A11) * (B10 + B11)
    # A01 = i=0,j=1 -> U[1, 6]
    # A11 = i=1,j=1 -> U[3, 6]
    # B10 = j=1,k=0 -> V[2, 6]
    # B11 = j=1,k=1 -> V[3, 6]
    U[1, 6] = 1
    U[3, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    # C00 += M7 -> W[0*2+0, 6] = W[0, 6]
    W[0, 6] = 1
    
    return U, V, W

def schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p matrix multiplication."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def add_subproblem(global_U, global_V, global_W, sub_U, sub_V, sub_W,
                   i_off, j_off, k_off, n_sub, m_sub, p_sub, n_glob, m_glob, p_glob):
    """
    Add a sub-problem decomposition to global matrices.
    
    Local indices:
    U: i_loc * m_sub + j_loc
    V: j_loc * p_sub + k_loc
    W: k_loc * n_sub + i_loc
    
    Global indices:
    U: (i_loc + i_off) * m_glob + (j_loc + j_off)
    V: (j_loc + j_off) * p_glob + (k_loc + k_off)
    W: (k_loc + k_off) * n_glob + (i_loc + i_off)
    """
    n_sub_rows = n_sub * m_sub
    m_p_sub = m_sub * p_sub
    p_n_sub = p_sub * n_sub
    rank_sub = sub_U.shape[1]
    
    for r in range(rank_sub):
        for u_idx in range(n_sub_rows):
            val = sub_U[u_idx, r]
            if val != 0:
                i_loc = u_idx // m_sub
                j_loc = u_idx % m_sub
                glob_u = (i_loc + i_off) * m_glob + (j_loc + j_off)
                global_U[glob_u, r] += val
        
        for v_idx in range(m_p_sub):
            val = sub_V[v_idx, r]
            if val != 0:
                j_loc = v_idx // p_sub
                k_loc = v_idx % p_sub
                glob_v = (j_loc + j_off) * p_glob + (k_loc + k_off)
                global_V[glob_v, r] += val
        
        for w_idx in range(p_n_sub):
            val = sub_W[w_idx, r]
            if val != 0:
                k_loc = w_idx // n_sub
                i_loc = w_idx % n_sub
                glob_w = (k_loc + k_off) * n_glob + (i_loc + i_off)
                global_W[glob_w, r] += val

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4x7 matrix multiplication.
    Uses Strassen's algorithm where beneficial.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strategy: Split into blocks and use Strassen where 2x2 sub-blocks exist
    # A is 3x4, B is 4x7
    # We can split A into rows: top 2 rows (2x4) and bottom 1 row (1x4)
    # B into cols: first 6 cols (4x6) and last 1 col (4x1)
    # Or other splits...
    
    # Let's try a different approach:
    # Split A into 2x4 and 1x4
    # Split B into 4x2, 4x2, 4x2, 4x1
    # Actually, let me think about this more carefully.
    
    # For 3x4x7:
    # We can decompose as:
    # - 2x4x2 block (rows 0-1 of A, cols 0-1 of B)
    # - 2x4x2 block (rows 0-1 of A, cols 2-3 of B)
    # - 2x4x2 block (rows 0-1 of A, cols 4-5 of B)
    # - 2x4x1 block (rows 0-1 of A, col 6 of B)
    # - 1x4x2 block (row 2 of A, cols 0-1 of B)
    # - 1x4x2 block (row 2 of A, cols 2-3 of B)
    # - 1x4x2 block (row 2 of A, cols 4-5 of B)
    # - 1x4x1 block (row 2 of A, col 6 of B)
    
    # For the 2x4x2 blocks, we can potentially use Strassen-like techniques
    # But Strassen is for square matrices. Let me think...
    
    # Actually, a better approach: Use the known optimal or near-optimal decompositions
    # For 3x4x7, the schoolbook rank is 3*4*7 = 84
    # We can do better by grouping.
    
    # Let's use a hybrid approach:
    # Group columns of B into pairs where possible
    # B has 7 columns: cols 0,1 | 2,3 | 4,5 | 6
    
    # For rows 0-1 of A (2 rows) and cols 0-1 of B (2 cols): 2x4x2
    # For rows 0-1 of A (2 rows) and cols 2-3 of B (2 cols): 2x4x2
    # For rows 0-1 of A (2 rows) and cols 4-5 of B (2 cols): 2x4x2
    # For rows 0-1 of A (2 rows) and col 6 of B (1 col): 2x4x1
    # For row 2 of A (1 row) and cols 0-1 of B (2 cols): 1x4x2
    # For row 2 of A (1 row) and cols 2-3 of B (2 cols): 1x4x2
    # For row 2 of A (1 row) and cols 4-5 of B (2 cols): 1x4x2
    # For row 2 of A (1 row) and col 6 of B (1 col): 1x4x1
    
    # Schoolbook ranks:
    # 2x4x2: 16
    # 2x4x2: 16
    # 2x4x2: 16
    # 2x4x1: 8
    # 1x4x2: 8
    # 1x4x2: 8
    # 1x4x2: 8
    # 1x4x1: 4
    # Total: 16+16+16+8+8+8+8+4 = 84 (same as schoolbook)
    
    # To improve, we need to share computations. Let me try a different decomposition.
    
    # Approach: Use the fact that we can sometimes reduce rank by careful grouping.
    # Let's just start with schoolbook and see if we can improve iteratively.
    
    # For now, return schoolbook as a baseline
    return {"U": schoolbook(n, m, p)[0].tolist(),
            "V": schoolbook(n, m, p)[1].tolist(),
            "W": schoolbook(n, m, p)[2].tolist()}

# Let's try a better construction
def construct_v2():
    """
    Improved construction using block decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # We'll build up the decomposition piece by piece
    # Start with empty global matrices
    # We'll accumulate contributions
    
    # First, let's figure out the total rank we'll use
    # We'll try to be smarter than schoolbook
    
    # Strategy: 
    # Split the 4x7 matrix B into:
    #   B1: 4x2 (cols 0-1)
    #   B2: 4x2 (cols 2-3)
    #   B3: 4x2 (cols 4-5)
    #   B4: 4x1 (col 6)
    
    # Split A (3x4) into:
    #   A1: 2x4 (rows 0-1)
    #   A2: 1x4 (row 2)
    
    # For A1 * B1 (2x4 * 4x2 = 2x2):
    # We can use Strassen if we reshape appropriately
    # Actually, for 2x4 * 4x2, we can think of it as:
    # [A1_left A1_right] * [B1_top; B1_bottom]
    # where A1_left is 2x2, A1_right is 2x2, B1_top is 2x2, B1_bottom is 2x2
    # This is a 2x2 block matrix product, which can use Strassen!
    
    # Similarly for A1 * B2 and A1 * B3
    
    # For A2 * B1 (1x4 * 4x2 = 1x2):
    # This is just a row vector times a matrix, rank 8 by schoolbook
    # But we can do better: 1x4 * 4x2 has rank at most 4 (since one dim is 1)
    # Actually, the tensor rank for 1x4x2 is 4 (each output element needs one term)
    # Wait, no. For 1xm x p, the rank is m*p... let me think.
    # C[i,k] = sum_j A[i,j] * B[j,k] where i is always 0
    # So C[k] = sum_j A[j] * B[j,k]
    # Each C[k] for k=0,1 needs sum over j=0..3
    # So we need at least... actually the tensor T_{j,k} = A_j * B_k
    # The rank of this tensor is min(4, 2) = 2? No...
    # For fixed A and B, C is determined. The tensor multiplication T maps A,B to C.
    # T_{a,b,c} = 1 if a=b (for A index) and b=c (for B index)... 
    # Actually for 1xm x p: C_k = sum_j A_j B_{j,k}
    # The tensor has indices (j, j, k) -> delta_{j,j} delta_{j,k}... no
    # T_{j, j, k} = 1 (summing over j)
    # So T has rank... we need to express C_k as sum_r U_r A * V_r B
    # C_k = sum_j A_j B_{j,k} = sum_r (sum_j U_{r,j} A_j) (sum_j V_{r,j,k} B_{j,k})
    # This is tricky. The rank for 1xm x p is actually m (I think), achieved by:
    # For each j, term j: U_j = e_j (in A space), V_j = e_j^T in B space (row j of B)
    # Then C_k = sum_j A_j * B_{j,k}
    # So rank = m = 4 for 1x4x2.
    
    # Hmm, so for A2*B1, rank 4
    # For A2*B2, rank 4
    # For A2*B3, rank 4
    # For A2*B4, rank 4 (1x4x1 has rank 4? No, 1x4x1: C = sum_j A_j B_j, rank 4)
    # Wait, 1x4x1: C[0] = sum_j A_j B_j, this is an inner product. Rank is 4? 
    # Actually no, the rank of the 1x4x1 tensor is 4 (you need 4 terms in general).
    # Hmm, but actually for a single output, C = A dot B, the tensor T_{j,j,j}... 
    # Let me reconsider. The matrix multiplication tensor for (1,4,1) is T_{i,j,k} where i in {0}, j in {0,1,2,3}, k in {0}
    # T_{0,j,0} = delta_{j,j}... no, T_{i,j,k} = 1 if the multiplication uses A_{i,j} * B_{j,k}
    # So T_{0,j,0} = 1 for all j (since i=0, k=0 always)
    # This is just a vector of ones, rank 1!
    # Wait no. The tensor T has T_{i,j,k} = 1 if this term appears in the sum for C_{i,k}
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So for (1,4,1): C_{0,0} = sum_{j=0}^{3} A_{0,j} B_{j,0}
    # T_{0,j,0} = 1 for j=0,1,2,3
    # The rank of this tensor: we need to write it as sum_r U_r(i) V_r(j) W_r(k)
    # T_{0,j,0} = 1 = sum_r U_r(0) V_r(j) W_r(0)
    # This is possible with rank 1: U_0(0)=1, V_0(j)=1, W_0(0)=1
    # So rank is 1 for (1,4,1)!
    
    # I'm confusing myself. Let me just use schoolbook for the small blocks and Strassen for the 2x2 blocks.
    
    # For now, let's just return schoolbook. The task is to find rank-minimal, and we know
    # schoolbook is 84. Let's see if we can do better.
    
    # Actually, let me try a concrete improved decomposition.
    
    # I'll use the following strategy:
    # For the 2x4 x 4x2 sub-problems (A1 * B1, A1 * B2, A1 * B3):
    # Each is a 2x4 * 4x2 = 2x2 result
    # We can write A1 = [A1a A1b] where A1a is 2x2, A1b is 2x2
    # B1 = [B1a; B1b] where B1a is 2x2, B1b is 2x2
    # Then A1 * B1 = A1a*B1a + A1b*B1b
    # Each 2x2*2x2 can use Strassen (rank 7), so total rank 14 for 2x4x2
    # vs schoolbook 16. Saving of 2 per block.
    
    # For A2 * B1, A2 * B2, A2 * B3 (1x4 * 4x2 = 1x2):
    # Schoolbook rank 8 each
    # Can we do better? 1x4x2: C_k = sum_j A_j B_{j,k} for k=0,1
    # This is like computing two dot products with overlapping B rows
    # The tensor T_{j,k} for the multiplication: T_{j, j, k} = delta... 
    # Actually, I think the rank for 1x4x2 is 4, not 8.
    # Reason: C_0 = A_0 B_0 + A_1 B_1 + A_2 B_2 + A_3 B_3
    #         C_1 = A_0 B_4 + A_1 B_5 + A_2 B_6 + A_3 B_7
    # We can use 4 terms: term j uses A_j, and for B we need B_{j} for C_0 and B_{4+j} for C_1
    # But a single term can only have one B vector... 
    # Actually, in the tensor decomposition, each term is U_r(i) * V_r(j) * W_r(k)
    # For 1x4x2: i is fixed at 0, j in 0..3, k in 0..1
    # T_{0,j,k} = 1 if j and k are "compatible" in the multiplication
    # C_k = sum_j A_j B_{j,k}
    # So T_{0,j,k} = 1 for all j (since every j contributes to every k)
    # Wait no! C_0 = sum_j A_j B_{j,0}, C_1 = sum_j A_j B_{j,1}
    # So T_{0,j,0} = 1 and T_{0,j,1} = 1 for all j
    # This means T is just an all-ones tensor of shape (1,4,2)
    # The rank of an all-ones tensor of shape (1,m,p) is 1!
    # Because T_{0,j,k} = 1 = 1 * 1 * 1
    # So we can use rank 1: U_0(0)=1, V_0(j)=1, W_0(k)=1
    # But wait, that would give C_k = (sum_j A_j) * (sum... no
    # Let me re-examine. The decomposition is:
    # C_{i,k} = sum_r U_{r,i} * (V_r * B)_{...}
    # Actually, the standard formulation:
    # C_{i,k} = sum_r U_{r,i} * A_{i,j} ... no
    # 
    # Let me go back to the problem statement:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: C = sum_r (U[:,r] * A) * (B * V[:,r]) * W[:,r]... 
    # 
    # Actually, I think the convention is:
    # C_{i,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} ... no, that's not right either.
    # 
    # From the seed program:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This suggests: C_{i,k} = sum_j sum_col U[i*m+j, col] * A[i,j] * V[j*p+k, col] * B[j,k] * W[k*n+i, col]
    # But in the schoolbook, each col corresponds to a specific (i,j,k) triple, and
    # U[i*m+j, col]=1 only for that (i,j), V[j*p+k, col]=1 only for that (j,k), W[k*n+i, col]=1 only for that (k,i)
    # So effectively, each column picks out one term A[i,j]*B[j,k] and adds it to C[i,k]
    # 
    # So the formula is: C_{i,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # where the sum over r implicitly also sums over j (since U and V have j as part of their index)
    # 
    # More precisely: C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    # OK so with this understanding, for 1x4x2:
    # i is always 0, j in 0..3, k in 0..1
    # C_{0,k} = sum_j sum_r U[j, r] * V[j*2+k, r] * W[k, r] * A[0,j] * B[j,k]
    # 
    # For schoolbook, each (j,k) pair gets its own r, so rank = 4*2 = 8
    # 
    # Can we share? For example, for a fixed j, both k=0 and k=1 use the same A[0,j]
    # So we might be able to use fewer terms.
    # 
    # Let's try rank 4: one term per j
    # For term r=j: U[j, j] = 1, V[j*2+k, j] = 1 for both k=0,1, W[k, j] = 1 for both k=0,1
    # Then C_{0,k} = sum_j A[0,j] * B[j,k] * 1 * 1 * 1 = sum_j A[0,j] B[j,k] ✓
    # So yes, rank 4 works for 1x4x2!
    
    # Great, so for A2*B1, A2*B2, A2*B3: rank 4 each instead of 8. Saving 4 each, total saving 12.
    # For A2*B4 (1x4x1): rank 4 (one term per j, similar argument)
    
    # Now for A1*B1 (2x4x2):
    # Can we do better than 14 (Strassen on two 2x2 blocks)?
    # 2x4x2: C_{i,k} for i=0,1 and k=0,1
    # C_{i,k} = sum_j A[i,j] B[j,k]
    # Schoolbook: 2*4*2 = 16
    # Strassen approach: split into two 2x2 products, each rank 7, total 14
    # Can we do 13 or less?
    # 
    # Actually, for 2x4x2, we can view it as:
    # [A_i0 A_i1 A_i2 A_i3] * [[B_00 B_01], [B_10 B_11], [B_20 B_21], [B_30 B_31]]
    # 
    # Another approach: group j into pairs
    # C_{i,k} = (A[i,0] B[0,k] + A[i,1] B[1,k]) + (A[i,2] B[2,k] + A[i,3] B[3,k])
    # Each group is a 2x1 * 2x2 product... not quite.
    # 
    # Let's just use the Strassen-based decomposition for now.
    
    # Total rank estimate:
    # A1*B1: 14 (Strassen)
    # A1*B2: 14 (Strassen)
    # A1*B3: 14 (Strassen)
    # A1*B4 (2x4x1): ?
    #   C_{i,0} = sum_j A[i,j] B[j,0] for i=0,1
    #   Schoolbook: 8
    #   Can we do 4? One term per j: U[i*m+j, r]... for fixed j, both i=0,1 use A[i,j]
    #   Term r=j: U[0*4+j, j]=1, U[1*4+j, j]=1, V[j*1+0, j]=1, W[0*2+i, j]=1 for i=0,1
    #   Then C_{i,0} = sum_j A[i,j] B[j,0] ✓
    #   So rank 4 for 2x4x1!
    # A2*B1 (1x4x2): 4
    # A2*B2 (1x4x2): 4
    # A2*B3 (1x4x2): 4
    # A2*B4 (1x4x1): 4
    # Total: 14+14+14+4+4+4+4+4 = 62
    
    # That's a significant improvement over 84!
    
    # Let's implement this.
    
    n, m, p = 3, 4, 7
    
    # We'll build the decomposition in blocks
    # Block (i_off, i_end, j_off, j_end) for A rows i_off..i_end-1 and B cols j_off..j_end-1
    
    # Define helper to create Strassen-based 2x4x2 decomposition
    def make_2x4x2():
        """
        Decomposition for 2x4 * 4x2 = 2x2
        Split A (2x4) into A_left (2x2) and A_right (2x2)
        Split B (4x2) into B_top (2x2) and B_bottom (2x2)
        Result = A_left * B_top + A_right * B_bottom
        Use Strassen for each 2x2*2x2
        """
        # This is equivalent to two 2x2 matrix multiplications
        # Each has rank 7 by Strassen
        # Total rank: 14
        
        # For the left part (A_left * B_top):
        # A_left: i=0,1; j=0,1
        # B_top: j=0,1; k=0,1
        # This is a standard 2x2x2 Strassen
        U_left, V_left, W_left = strassen_2x2()
        
        # For the right part (A_right * B_bottom):
        # A_right: i=0,1; j=2,3
        # B_bottom: j=2,3; k=0,1
        U_right, V_right, W_right = strassen_2x2()
        
        # Combine: U has shape (2*4, 14), V has shape (4*2, 14), W has shape (2*2, 14)
        rank = 14
        U = np.zeros((8, rank))
        V = np.zeros((8, rank))
        W = np.zeros((4, rank))
        
        # Left part uses columns 0-6
        # A_left indices: i*4 + j for i=0,1 and j=0,1
        # In the sub-problem, local indices are i*2 + j for i=0,1 and j=0,1
        # Map: local_u = i*2 + j -> global_u = i*4 + j
        for r in range(7):
            for local_u in range(4):
                i = local_u // 2
                j = local_u % 2
                global_u = i * 4 + j
                U[global_u, r] = U_left[local_u, r]
            
            for local_v in range(4):
                j = local_v // 2
                k = local_v % 2
                global_v = j * 2 + k
                V[global_v, r] = V_left[local_v, r]
            
            for local_w in range(4):
                k = local_w // 2
                i = local_w % 2
                global_w = k * 2 + i
                W[global_w, r] = W_left[local_w, r]
        
        # Right part uses columns 7-13
        # A_right indices: i*4 + j for i=0,1 and j=2,3
        # In the sub-problem, local indices are i*2 + j for i=0,1 and j=0,1
        # Map: local_u = i*2 + j -> global_u = i*4 + (j+2)
        for r in range(7):
            for local_u in range(4):
                i = local_u // 2
                j = local_u % 2
                global_u = i * 4 + (j + 2)
                U[global_u, r + 7] = U_right[local_u, r]
            
            for local_v in range(4):
                j = local_v // 2
                k = local_v % 2
                global_v = (j + 2) * 2 + k
                V[global_v, r + 7] = V_right[local_v, r]
            
            for local_w in range(4):
                k = local_w // 2
                i = local_w % 2
                global_w = k * 2 + i
                W[global_w, r + 7] = W_right[local_w, r]
        
        return U, V, W
    
    def make_2x4x1():
        """
        Decomposition for 2x4 * 4x1 = 2x1
        Rank 4: one term per j
        """
        rank = 4
        U = np.zeros((8, rank))
        V = np.zeros((4, rank))
        W = np.zeros((2, rank))
        
        for j in range(4):
            # U[i*4 + j, j] = 1 for i=0,1
            U[0 * 4 + j, j] = 1
            U[1 * 4 + j, j] = 1
            # V[j*1 + 0, j] = 1
            V[j, j] = 1
            # W[0*2 + i, j] = 1 for i=0,1
            W[0, j] = 1
            W[1, j] = 1
        
        return U, V, W
    
    def make_1x4x2():
        """
        Decomposition for 1x4 * 4x2 = 1x2
        Rank 4: one term per j
        """
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((2, rank))
        
        for j in range(4):
            # U[0*4 + j, j] = 1
            U[j, j] = 1
            # V[j*2 + k, j] = 1 for k=0,1
            V[j * 2 + 0, j] = 1
            V[j * 2 + 1, j] = 1
            # W[k*1 + 0, j] = 1 for k=0,1
            W[0, j] = 1
            W[1, j] = 1
        
        return U, V, W
    
    def make_1x4x1():
        """
        Decomposition for 1x4 * 4x1 = 1x1
        Rank 4: one term per j
        """
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((1, rank))
        
        for j in range(4):
            U[j, j] = 1
            V[j, j] = 1
            W[0, j] = 1
        
        return U, V, W
    
    # Now assemble the full 3x4x7 decomposition
    # A is 3x4, B is 4x7
    # Blocks:
    # (0-1, 0-1): 2x4x2 -> rank 14
    # (0-1, 2-3): 2x4x2 -> rank 14
    # (0-1, 4-5): 2x4x2 -> rank 14
    # (0-1, 6-6): 2x4x1 -> rank 4
    # (2-2, 0-1): 1x4x2 -> rank 4
    # (2-2, 2-3): 1x4x2 -> rank 4
    # (2-2, 4-5): 1x4x2 -> rank 4
    # (2-2, 6-6): 1x4x1 -> rank 4
    # Total: 14+14+14+4+4+4+4+4 = 62
    
    total_rank = 62
    U = np.zeros((12, total_rank))  # 3*4 = 12
    V = np.zeros((28, total_rank))  # 4*7 = 28
    W = np.zeros((21, total_rank))  # 7*3 = 21
    
    col_offset = 0
    
    # Block (0-1, 0-1): 2x4x2
    U_blk, V_blk, W_blk = make_2x4x2()
    # Map to global:
    # A rows 0-1, cols 0-3 -> global U indices: (0-1)*4 + (0-3) = 0-7
    # B rows 0-3, cols 0-1 -> global V indices: (0-3)*7 + (0-1) = 0-8, 7-15, 14-22, 21-29... 
    # Wait, V index is j*p + k where j is row of B (0-3) and k is col of B (0-6)
    # For this block, k is 0-1, so V indices: j*7 + k for j=0-3, k=0-1
    # W index is k*n + i where k is col of B (0-1) and i is row of C (0-1)
    # For this block, i is 0-1, so W indices: k*3 + i for k=0-1, i=0-1
    
    for r in range(14):
        # U: local row i*4+j for i=0-1, j=0-3 -> global same (since i_off=0, j_off=0)
        for local_row in range(8):
            if U_blk[local_row, r] != 0:
                U[local_row, col_offset + r] = U_blk[local_row, r]
        
        # V: local row j*2+k for j=0-3, k=0-1 -> global row j*7+k
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + k
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        # W: local row k*2+i for k=0-1, i=0-1 -> global row k*3+i
        for local_row in range(4):
            k = local_row // 2
            i = local_row % 2
            global_row = k * 3 + i
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 14
    
    # Block (0-1, 2-3): 2x4x2
    U_blk, V_blk, W_blk = make_2x4x2()
    for r in range(14):
        for local_row in range(8):
            if U_blk[local_row, r] != 0:
                U[local_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + (k + 2)
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(4):
            k = local_row // 2
            i = local_row % 2
            global_row = (k + 2) * 3 + i
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 14
    
    # Block (0-1, 4-5): 2x4x2
    U_blk, V_blk, W_blk = make_2x4x2()
    for r in range(14):
        for local_row in range(8):
            if U_blk[local_row, r] != 0:
                U[local_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + (k + 4)
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(4):
            k = local_row // 2
            i = local_row % 2
            global_row = (k + 4) * 3 + i
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 14
    
    # Block (0-1, 6-6): 2x4x1
    U_blk, V_blk, W_blk = make_2x4x1()
    for r in range(4):
        for local_row in range(8):
            if U_blk[local_row, r] != 0:
                U[local_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(4):
            j = local_row
            k = 0
            global_row = j * 7 + 6
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(2):
            k = 0
            i = local_row
            global_row = 6 * 3 + i
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 4
    
    # Block (2-2, 0-1): 1x4x2
    U_blk, V_blk, W_blk = make_1x4x2()
    for r in range(4):
        for local_row in range(4):
            j = local_row
            i = 2
            global_row = 2 * 4 + j
            if U_blk[local_row, r] != 0:
                U[global_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + k
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(2):
            k = local_row
            i = 2
            global_row = k * 3 + 2
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 4
    
    # Block (2-2, 2-3): 1x4x2
    U_blk, V_blk, W_blk = make_1x4x2()
    for r in range(4):
        for local_row in range(4):
            j = local_row
            global_row = 2 * 4 + j
            if U_blk[local_row, r] != 0:
                U[global_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + (k + 2)
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(2):
            k = local_row
            global_row = (k + 2) * 3 + 2
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 4
    
    # Block (2-2, 4-5): 1x4x2
    U_blk, V_blk, W_blk = make_1x4x2()
    for r in range(4):
        for local_row in range(4):
            j = local_row
            global_row = 2 * 4 + j
            if U_blk[local_row, r] != 0:
                U[global_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(8):
            j = local_row // 2
            k = local_row % 2
            global_row = j * 7 + (k + 4)
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(2):
            k = local_row
            global_row = (k + 4) * 3 + 2
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 4
    
    # Block (2-2, 6-6): 1x4x1
    U_blk, V_blk, W_blk = make_1x4x1()
    for r in range(4):
        for local_row in range(4):
            j = local_row
            global_row = 2 * 4 + j
            if U_blk[local_row, r] != 0:
                U[global_row, col_offset + r] = U_blk[local_row, r]
        
        for local_row in range(4):
            j = local_row
            global_row = j * 7 + 6
            if V_blk[local_row, r] != 0:
                V[global_row, col_offset + r] = V_blk[local_row, r]
        
        for local_row in range(1):
            global_row = 6 * 3 + 2
            if W_blk[local_row, r] != 0:
                W[global_row, col_offset + r] = W_blk[local_row, r]
    
    col_offset += 4
    
    # Verify col_offset == total_rank
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Use the improved construction
def construct():
    return construct_v2()
