"""Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _build_rank1(n, m, p):
    """Build the rank-1 (schoolbook) decomposition for reference."""
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

def _verify(U, V, W, n, m, p):
    """Check that U ⊗ V ⊗ W equals the matmul tensor."""
    rank = len(W[0])
    for a in range(n * m):
        for b in range(m * p):
            for c_idx in range(p * n):
                s = 0
                for c in range(rank):
                    s += U[a][c] * V[b][c] * W[c_idx][c]
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c_idx // n
                i2 = c_idx % n
                valid = (i == i2 and j == j2 and k == k2)
                expected = 1 if valid else 0
                if s != expected:
                    return False
    return True

def _matmul_tensor_rank1(n, m, p):
    """
    Build rank-1 decomposition for n x m * m x p matmul.
    Returns (U, V, W) as lists of lists.
    """
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _add_decomposition(U, V, W, u_list, v_list, w_list, scale_u=1, scale_v=1, scale_w=1):
    """
    Add a set of rank-1 terms to existing decomposition.
    u_list, v_list, w_list are lists of (index, value) pairs for each rank-1 term.
    Returns new (U, V, W) with these terms added.
    """
    # Count how many new columns we need
    num_terms = len(u_list)
    old_rank = len(U[0]) if U else 0
    new_rank = old_rank + num_terms
    
    # Expand U, V, W
    U_new = [row[:] + [0] * num_terms for row in U] if U else [[0] * new_rank for _ in range(len(U_new))]
    V_new = [row[:] + [0] * num_terms for row in V] if V else [[0] * new_rank for _ in range(len(V_new))]
    W_new = [row[:] + [0] * num_terms for row in W] if W else [[0] * new_rank for _ in range(len(W_new))]
    
    for t in range(num_terms):
        u_idx, u_val = u_list[t]
        v_idx, v_val = v_list[t]
        w_idx, w_val = w_list[t]
        U_new[u_idx][old_rank + t] = u_val * scale_u
        V_new[v_idx][old_rank + t] = v_val * scale_v
        W_new[w_idx][old_rank + t] = w_val * scale_w
    
    return U_new, V_new, W_new

def _strassen_2x2_times_2x4(A_block, B_block, U, V, W, offset_i, offset_j, offset_k, m_size, p_size):
    """
    Apply Strassen to multiply a 2x2 block of A by a 2x4 block of B.
    A_block: 2x2 list of lists
    B_block: 2x4 list of lists
    U, V, W: current decomposition matrices
    offset_i: starting row index in A (for the 2x2 block)
    offset_j: starting col index in A / row index in B (for the 2x2/2x4 block)
    offset_k: starting col index in B (for the 2x4 block)
    m_size: the m dimension (should be 4)
    p_size: the p dimension (should be 8)
    
    Returns updated (U, V, W)
    """
    # Strassen intermediates for 2x2 * 2x2 (but here B is 2x4, so we treat it as 2x2 of 2x2 blocks? No...)
    # Actually, let's think of this as: we have A (2x2) and B (2x4).
    # C = A * B is 2x4.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # This is just 2x2*2x4 matmul. Rank is at most 16.
    # Can we do better? For 2x2*2x2 it's 7. For 2x2*2x4, let's see.
    # C has 8 elements, each is a sum of 2 products. Total 16 rank-1 terms.
    # I don't think we can reduce below 16 for generic 2x2*2x4.
    
    # So let's just use schoolbook for this sub-problem: rank 16.
    # But wait, we're calling this from a higher-level Strassen, so the "blocks" 
    # in the higher-level Strassen are these 2x2 and 2x4 matrices.
    
    # Actually, I think the confusion is that I'm trying to use Strassen at two levels.
    # Let me reconsider.
    
    # At the top level: 4x4 * 4x8.
    # Split into 2x2 blocks: A has 4 blocks of size 2x2, B has 2 blocks of size 2x4.
    # C has 2 blocks of size 2x4.
    # C_top = A_top * B_top where A_top is 2x2 "matrix" of 2x2 blocks, B_top is 2x2 "matrix" of 2x4 blocks.
    # Using Strassen: 7 "multiplications" of 2x2 by 2x4.
    # Each 2x2 * 2x4 matmul has rank at most 16.
    # Total rank: 7 * 16 = 112.
    
    # So for this function, I just need to do schoolbook 2x2 * 2x4.
    
    U_new = U
    V_new = V
    W_new = W
    
    # For each (i_local, j_local, k_local) in the 2x2*2x4 sub-problem:
    # A index: (offset_i + i_local) * m_size + (offset_j + j_local)
    # B index: (offset_j + j_local) * p_size + (offset_k + k_local)
    # C index: (offset_k + k_local) * n + (offset_i + i_local)  where n=4
    
    n = 4  # hardcode for now
    for i_local in range(2):
        for j_local in range(2):
            for k_local in range(4):
                a_idx = (offset_i + i_local) * m_size + (offset_j + j_local)
                b_idx = (offset_j + j_local) * p_size + (offset_k + k_local)
                c_idx = (offset_k + k_local) * n + (offset_i + i_local)
                
                # Add rank-1 term
                col = len(U_new[0])
                U_new = [row[:] + [0] for row in U_new]
                V_new = [row[:] + [0] for row in V_new]
                W_new = [row[:] + [0] for row in W_new]
                U_new[a_idx][col] = 1
                V_new[b_idx][col] = 1
                W_new[c_idx][col] = 1
    
    return U_new, V_new, W_new

def _build_strassen_decomposition():
    """
    Build a Strassen-based decomposition for 4x4 * 4x8 matmul.
    Rank should be 7 * 16 = 112.
    """
    n, m, p = 4, 4, 8
    
    # Initialize empty decomposition
    U = []
    V = []
    W = []
    
    # Strassen for 2x2 "matrix" multiplication:
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x4
    # C = [[C00, C01], [C10, C11]] each 2x4
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each "product" like (A00+A11)*(B00+B11) is a 2x2 * 2x4 matmul.
    # We need to decompose each into rank-1 terms.
    
    # Define helper: add a 2x2*2x4 schoolbook decomposition
    # with optional linear combinations of A and B blocks.
    
    def add_2x2_2x4_prod(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, 
                         A_signs, B_signs, C_sign, U, V, W):
        """
        Add decomposition for: (sum_{r,c} A_signs[r][c] * A[r, c]) * (sum_{r,c} B_signs[r][c] * B[r, c])
        = C_sign * C_block
        
        A_rows, A_cols: which 2x2 block of A (row offset, col offset in 2x2 grid)
        B_rows, B_cols: which 2x4 block of B
        C_rows, C_cols: which 2x4 block of C
        A_signs: 2x2 list of signs for A block
        B_signs: 2x2 list of signs for B block (but B is 2x4, so this applies to rows)
        C_sign: overall sign for C block
        
        Actually, this is getting complicated. Let me think differently.
        """
        pass
    
    # Let me take a simpler approach: just enumerate all rank-1 terms explicitly.
    
    # For each Strassen intermediate, I'll compute the rank-1 decomposition.
    
    # M1 = (A00+A11)*(B00+B11)
    # This means: for each i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    # Contribution to C: (A[0,j] + A[2,j]) * (B[0,k] + B[2,k]) for C[0,k]
    #                  (A[1,j] + A[3,j]) * (B[0,k] + B[2,k]) for C[1,k]
    # Wait, A00 is rows 0-1, cols 0-1. A11 is rows 2-3, cols 2-3.
    # So A00+A11 has entries: [A[0,0]+A[2,2], A[0,1]+A[2,3]; A[1,0]+A[3,2], A[1,1]+A[3,3]]
    # B00 is rows 0-1, cols 0-3. B11 is rows 2-3, cols 0-3.
    # B00+B11 has entries: [B[0,0]+B[2,0], ..., B[0,3]+B[2,3]; B[1,0]+B[3,0], ..., B[1,3]+B[3,3]]
    
    # The product (A00+A11)*(B00+B11) is a 2x4 matrix.
    # Entry (i,k) = sum_j (A00+A11)[i,j] * (B00+B11)[j,k]
    #             = sum_j (A[2*i, j] + A[2*i+2, j+2]) * (B[j, k] + B[j+2, k])
    
    # This expands to 4 terms for each (i,k):
    # A[2i,j]*B[j,k] + A[2i,j]*B[j+2,k] + A[2i+2,j+2]*B[j,k] + A[2i+2,j+2]*B[j+2,k]
    
    # So the rank-1 terms for M1 are:
    # For each i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    #   Term 1: U[A[2i,j]] * V[B[j,k]] * W[C[2i,k]]  (but with the right indices)
    #   Term 2: U[A[2i,j]] * V[B[j+2,k]] * W[C[2i,k]]
    #   Term 3: U[A[2i+2,j+2]] * V[B[j,k]] * W[C[2i,k]]
    #   Term 4: U[A[2i+2,j+2]] * V[B[j+2,k]] * W[C[2i,k]]
    
    # Wait, I need to be careful with the index mapping.
    
    # Let me define the index functions:
    # A[i,j] -> U index: i*m + j = i*4 + j
    # B[j,k] -> V index: j*p + k = j*8 + k
    # C[i,k] -> W index: k*n + i = k*4 + i
    
    # For M1 = (A00+A11)*(B00+B11):
    # A00: i in {0,1}, j in {0,1} -> A[0,0], A[0,1], A[1,0], A[1,1]
    # A11: i in {2,3}, j in {2,3} -> A[2,2], A[2,3], A[3,2], A[3,3]
    # B00: j in {0,1}, k in {0,1,2,3} -> B[0,0..3], B[1,0..3]
    # B11: j in {2,3}, k in {0,1,2,3} -> B[2,0..3], B[3,0..3]
    
    # (A00+A11)[i,j] for i in {0,1}, j in {0,1}:
    #   [0,0] = A[0,0] + A[2,2]
    #   [0,1] = A[0,1] + A[2,3]
    #   [1,0] = A[1,0] + A[3,2]
    #   [1,1] = A[1,1] + A[3,3]
    
    # (B00+B11)[j,k] for j in {0,1}, k in {0,1,2,3}:
    #   [0,k] = B[0,k] + B[2,k]
    #   [1,k] = B[1,k] + B[3,k]
    
    # Product[i,k] = sum_j (A00+A11)[i,j] * (B00+B11)[j,k]
    # = (A[0,0]+A[2,2])*(B[0,k]+B[2,k]) + (A[0,1]+A[2,3])*(B[1,k]+B[3,k])  for i=0
    # = (A[1,0]+A[3,2])*(B[0,k]+B[2,k]) + (A[1,1]+A[3,3])*(B[1,k]+B[3,k])  for i=1
    
    # Expanding i=0, k=0:
    # = A[0,0]*B[0,0] + A[0,0]*B[2,0] + A[2,2]*B[0,0] + A[2,2]*B[2,0]
    #   + A[0,1]*B[1,0] + A[0,1]*B[3,0] + A[2,3]*B[1,0] + A[2,3]*B[3,0]
    
    # So 8 rank-1 terms for each (i,k) pair, and there are 2*4=8 such pairs.
    # Total: 64 rank-1 terms for M1 alone!
    
    # That's way more than 16. I made an error.
    
    # The issue is that (A00+A11)*(B00+B11) is NOT a simple 2x2*2x4 matmul.
    # It's a 2x2 matrix (with entries being sums) times a 2x4 matrix (with entries being sums).
    # The result is a 2x4 matrix, but each entry is a sum of 4 products (not 2).
    
    # So the rank of each Strassen intermediate is 8*4 = 32? No wait...
    
    # Let me reconsider. In Strassen, the "multiplication" of two 2x2 matrices 
    # gives a 2x2 result. Here, we're "multiplying" a 2x2 (of 2x2 blocks) by a 2x2 (of 2x4 blocks).
    # The result should be a 2x2 (of 2x4 blocks).
    
    # But when we expand (A00+A11)*(B00+B11), we get a 2x4 matrix where each entry
    # is a sum of 4 scalar products. So the total number of scalar products is 2*4*4 = 32.
    
    # That means each Strassen intermediate has rank 32, and 7 of them gives 224 total.
    # That's worse than schoolbook (128)!
    
    # I see the problem: Strassen reduces the number of "matrix multiplications" from 4 to 7,
    # but each "matrix multiplication" here is more expensive because of the linear combinations.
    
    # For square matrices, Strassen works because (A00+A11)*(B00+B11) can be computed
    # with the same asymptotic cost as one matrix multiplication. But in our case,
    # the linear combinations increase the cost.
    
    # Let me abandon the Strassen approach and try something else.
    
    # Alternative: use the fact that 4x4*4x8 can be viewed as 4 instances of 1x4*4x8.
    # Each 1x4*4x8 has rank 4*8 = 32? No, it's just 8 dot products, each of length 4.
    # Rank is 8*4 = 32? Or is it just 32?
    
    # Actually, for 1xm * mxp, the rank is m*p (each output is a dot product).
    # So 1x4 * 4x8 has rank 4*8 = 32.
    # 4 such rows: 4*32 = 128. Same as schoolbook.
    
    # Let me try yet another approach: use the structure of the tensor more directly.
    
    # The tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T = 1 iff i=i', j=j', k=k'
    
    # This is a "permutation-like" tensor. The rank of such tensors can sometimes
    # be computed or bounded.
    
    # For now, let me just return schoolbook.
    return _build_rank1(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS
    
    best = _build_rank1(n, m, p)
    best_rank = len(best["W"][0])
    
    # Try Strassen-based approach
    if time.time() < deadline:
        strassen_result = _build_strassen_decomposition()
        if strassen_result:
            s_rank = len(strassen_result["W"][0])
            if s_rank < best_rank:
                if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                    best = strassen_result
                    best_rank = s_rank
    
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END
