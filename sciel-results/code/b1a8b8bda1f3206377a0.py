"""Program for 3x4x4 matrix multiplication rank minimization."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: sum over all rank terms should give 1 for (i,j,k) position
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook_rank(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _kronecker_decomposition(n, m, p):
    """
    Decompose the n×m×p matrix multiplication tensor using Kronecker product structure.
    
    For 3×4×4: decompose as sum over i of (e_i ⊗ I_4) structure.
    Each row i of A multiplies B (4×4) to produce row i of C (1×4).
    
    For each i, the 1×4×4 sub-tensor has rank 16 (schoolbook for 1×4×4).
    Total rank = 3 * 16 = 48.
    
    However, we can use the fact that for 1×m×p, the optimal rank is m*p
    (since each output element requires a separate product).
    
    Actually, let's think more carefully. The tensor T[i,j,k] = 1 if the 
    (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    
    For 3×4×4: T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    where the contribution is A[i,j]*B[j,k] -> C[i,k].
    
    The schoolbook gives rank 3*4*4 = 48.
    
    Using Kronecker: T = sum_i (e_i ⊗ I_4) ⊗ (I_4 ⊗ e_i^T) type structure...
    
    Let me think about this differently. The matrix multiplication tensor for 
    (n,m,p) can be written as:
    T = sum_{j=0}^{m-1} (I_n ⊗ e_j) ⊗ (e_j ⊗ I_p)
    
    where e_j is the j-th standard basis vector. This gives rank n*p for each j,
    total rank = m * n * p = 48 for 3×4×4.
    
    But we can potentially do better by exploiting structure. Let's try:
    For each column j of A (row j of B), we need to compute:
    C[:,k] += A[:,j] * B[j,k]
    
    This is: for each j, outer product of A[:,j] (3×1) and B[j,:] (1×4).
    The outer product tensor has rank 3*4 = 12 for each j.
    Total: 4 * 12 = 48.
    
    To beat 48, we need to share terms across different j values.
    
    Let's try a different approach: use the fact that 4×4 matrix multiplication
    can be done in rank 49 (or better with advanced algorithms).
    
    Actually, for 3×4×4, let's just use the structured decomposition and see
    if we can find a lower rank.
    
    For now, return schoolbook (48) and try to improve.
    """
    return _schoolbook_rank(n, m, p)

def _strassen_4x4():
    """
    Attempt to use Strassen-like decomposition for 4x4 matrix multiplication.
    
    Partition 4x4 into four 2x2 blocks.
    Use Strassen for each pair of 2x2 blocks.
    
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    
    C = A*B where C is 4x4.
    
    Using Strassen on the 2x2 block structure:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M1
    C10 = M2 + M4
    C11 = M1 - M2 + M6 + M3
    
    Each Mi is a 2x2 matrix product, which can be done in rank 7 (Strassen) 
    or rank 4 (schoolbook).
    
    If we use Strassen for each 2x2: 7 * 7 = 49 rank terms.
    If we use schoolbook for each 2x2: 7 * 4 = 28 rank terms.
    
    But wait, the Mi are themselves 2x2 matrix products. So:
    - Using schoolbook for inner 2x2: 7 * 4 = 28
    - Using Strassen for inner 2x2: 7 * 7 = 49
    
    So schoolbook for inner gives 28, which is better!
    
    For 3x4x4: 
    A is 3x4, B is 4x4.
    Partition A as [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4.
    B is 4x4.
    
    C_top = A_top * B (2x4 result)
    C_bottom = A_bottom * B (1x4 result)
    
    For C_top (2x4 from 2x4 * 4x4):
    This is like multiplying a 2x4 matrix by a 4x4 matrix.
    We can think of this as two 2x2 * 2x2 multiplications side by side.
    
    Let me just implement the block decomposition directly.
    """
    n, m, p = 3, 4, 4
    
    U = [[0.0] for _ in range(n * m)]
    V = [[0.0] for _ in range(m * p)]
    W = [[0.0] for _ in range(p * n)]
    
    # Decompose 3x4x4 as:
    # Rows 0,1 of A (2x4) times B (4x4) -> rows 0,1 of C (2x4)
    # Row 2 of A (1x4) times B (4x4) -> row 2 of C (1x4)
    
    # For the 2x4 * 4x4 part:
    # Partition into 2x2 blocks:
    # A_top = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C_top = [[C00, C01], [C10, C11]] where each is 2x2
    
    # Use Strassen for the 2x2 block multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each Mi is 2x2*2x2 -> use schoolbook (rank 4) for each
    # Total for 2x4*4x4: 7*4 = 28
    
    # For the 1x4 * 4x4 part:
    # A_bottom is 1x4, B is 4x4, C_bottom is 1x4
    # This is just 4 separate dot products: rank 4*4 = 16
    # Actually, it's: C_bottom[k] = sum_j A_bottom[j] * B[j,k]
    # Each C_bottom[k] requires 4 products, total 16 products.
    # But we can structure it: for each j, A_bottom[j] * B[j,:] is a 1x4 vector.
    # That's 4 outer products of 1x1 and 1x4, each rank 4, total 16.
    
    # Total rank: 28 + 16 = 44
    
    # Let's build this:
    col_offset = 0
    
    # Helper to add a schoolbook 2x2x2 block
    def add_schoolbook_2x2(i_off, j_off, k_off):
        nonlocal col_offset
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    U[u_idx].append(1.0)
                    V[v_idx].append(1.0)
                    W[w_idx].append(1.0)
                    col_offset += 1
    
    # Strassen terms for 2x2 blocks
    # i_off=0, j_off=0, k_off=0 for the 2x2x2 Strassen
    i_off, j_off, k_off = 0, 0, 0
    
    def u_row(i, j):
        return (i_off + i) * m + (j_off + j)
    def v_row(j, k):
        return (j_off + j) * p + (k_off + k)
    def w_col(k, i):
        return (k_off + k) * n + (i_off + i)
    
    # We'll add 7 Strassen terms, each using schoolbook for the inner 2x2
    # M1 = (A00+A11)(B00+B11)
    # This means: for the U side, A00 and A11 have coefficient 1
    # For the V side, B00 and B11 have coefficient 1
    # For the W side, we need to combine: C00 += M1, C01 += M1, C10 += M1, C11 += M1
    
    # But each Mi is itself a 2x2 matrix product. So we need to expand each Mi
    # into 4 scalar multiplications.
    
    # Let's define the 7 Strassen combinations and then expand each into schoolbook terms.
    
    # For each Strassen term, we have:
    # U coefficients: which A(i,j) entries participate and with what sign
    # V coefficients: which B(j,k) entries participate and with what sign  
    # W coefficients: which C(k,i) entries get the result and with what sign
    
    # M1: U={A00:1, A11:1}, V={B00:1, B11:1}, W={C00:1, C01:1, C10:1, C11:1}
    # M2: U={A10:1, A11:1}, V={B00:1}, W={C10:1, C11:1}
    # M3: U={A00:1}, V={B10:1, B11:-1}, W={C00:1, C01:-1}
    # M4: U={A11:1}, V={B01:1, B10:-1}, W={C10:1, C11:-1}
    # M5: U={A00:1, A01:1}, V={B11:1}, W={C00:1, C01:1}
    # M6: U={A10:1, A00:-1}, V={B00:1, B10:1}, W={C10:1, C11:1}
    # M7: U={A01:1, A11:-1}, V={B01:1, B11:1}, W={C00:1, C01:1}
    
    strassen_terms = [
        # (u_dict, v_dict, w_dict)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1}),  # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (0,1):-1}),  # M3
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(1,0):1, (1,1):-1}),  # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (0,1):1}),  # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,0):1, (1,1):1}),  # M6
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (0,1):1}),  # M7
    ]
    
    for u_dict, v_dict, w_dict in strassen_terms:
        # Expand this Strassen term into schoolbook 2x2x2 multiplications
        # For each combination of u_entry, v_entry, w_entry:
        # The product is: sum over (i,j) in u_dict, (j',k) in v_dict, (k',i') in w_dict
        # of u_coeff * v_coeff * w_coeff * A[i,j] * B[j',k] * C[k',i']
        # But we need j == j' for the product to be valid (contraction over j)
        
        # Actually, the Strassen decomposition works at the block level.
        # Each Mi is a 2x2 matrix = (linear combo of A blocks) * (linear combo of B blocks)
        # So Mi[i',k'] = sum_j (sum_i u_coeff[i,j] * A[i,j]) * (sum_j' v_coeff[j',k'] * B[j',k'])
        # where the j index is contracted.
        
        # For the scalar level, we need:
        # For each (i,j) in u_dict with coeff uc, and (j',k') in v_dict with coeff vc:
        #   if j == j' (same contraction index), then we have a term uc*vc * A[i,j]*B[j,k']
        #   This contributes to w entries: for each (k'',i'') in w_dict with coeff wc:
        #     if k' == k'' (output index matches), add wc to W[k'',i'']
        
        # Wait, I'm overcomplicating this. Let me think again.
        
        # The Strassen algorithm says:
        # M1 = (A00+A11) * (B00+B11)
        # where A00, A11, B00, B11 are 2x2 matrices.
        # M1 is a 2x2 matrix.
        # Then C00 += M1, C01 += M1, C10 += M1, C11 += M1
        
        # So M1[i',k'] = sum_j (A00[i',j] + A11[i',j]) * (B00[j,k'] + B11[j,k'])
        # = sum_j sum_j2 (A00[i',j] * B00[j2,k'] + A00[i',j] * B11[j2,k'] + A11[i',j] * B00[j2,k'] + A11[i',j] * B11[j2,k'])
        # where j == j2 for the contraction.
        
        # So M1[i',k'] = sum_j (A00[i',j]*B00[j,k'] + A00[i',j]*B11[j,k'] + A11[i',j]*B00[j,k'] + A11[i',j]*B11[j,k'])
        
        # Each term A[ia, ja] * B[jb, kb] is a scalar product.
        # In the tensor decomposition, each such product gets its own rank term.
        
        # For M1, the scalar products are:
        # A00[i,j]*B00[j,k] for i,j,k in {0,1}
        # A00[i,j]*B11[j,k] for i,j,k in {0,1}
        # A11[i,j]*B00[j,k] for i,j,k in {0,1}
        # A11[i,j]*B11[j,k] for i,j,k in {0,1}
        # Total: 4*4 = 16 scalar products for M1 alone!
        
        # That's not right. Strassen reduces the number of 2x2 multiplications from 8 to 7,
        # but each 2x2 multiplication still requires 4 scalar multiplications (schoolbook).
        # So 7 * 4 = 28 scalar multiplications for the 2x2 block Strassen.
        
        # Let me just enumerate all the scalar products directly.
        
        # For each Strassen term, expand into individual A[i,j]*B[j,k] products
        # that contribute to C[k,i] with appropriate coefficients.
        
        products = []  # list of (u_idx, v_idx, w_idx, coeff)
        
        for (ui, uj), uc in u_dict.items():
            for (vj, vk), vc in v_dict.items():
                if uj != vj:
                    continue  # contraction index must match
                for (wk, wi), wc in w_dict.items():
                    if vk != wk:
                        continue  # output k index must match
                    # We have: uc * A[ui, uj] * vc * B[vj, vk] contributes wc to C[wk, wi]
                    # But we need ui == wi for the output i index to match
                    # Actually, in the tensor, C[k,i] gets A[i,j]*B[j,k]
                    # So the output is C[k, i] where i is from A and k is from B.
                    # In w_dict, (wk, wi) means k'=wk, i'=wi
                    # We need i' == ui (the row index from A)
                    # And k' == vk (the col index from B)
                    if wi != ui:
                        continue
                    products.append((ui, uj, vj, vk, wk, wi, uc * vc * wc))
        
        # Now, for each unique (ui, uj, vj, vk, wk, wi) combination, we have a coefficient.
        # But actually, each (ui, uj, vk) determines the product A[ui,uj]*B[vj,vk] where uj==vj.
        # And it contributes to C[vk, ui] with coefficient uc*vc*wc.
        
        # Let's group by (ui, uj, vk) since uj==vj:
        from collections import defaultdict
        coeff_map = defaultdict(float)
        for (ui, uj, vj, vk, wk, wi, coeff) in products:
            # The product is A[ui, uj] * B[uj, vk] contributing to C[vk, ui]
            key = (ui, uj, vk)
            coeff_map[key] += coeff
        
        # Now add rank terms for each nonzero coefficient
        for (ui, uj, vk), coeff in coeff_map.items():
            if abs(coeff) < 1e-12:
                continue
            u_idx = (i_off + ui) * m + (j_off + uj)
            v_idx = (j_off + uj) * p + (k_off + vk)
            w_idx = (k_off + vk) * n + (i_off + ui)
            U[u_idx].append(coeff)
            V[v_idx].append(coeff)
            W[w_idx].append(coeff)
            col_offset += 1
    
    # Now add the 1x4 * 4x4 part (row 2 of A)
    # A[2,j] * B[j,k] -> C[k,2] for j,k in 0..3
    for j in range(4):
        for k in range(4):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx].append(1.0)
            V[v_idx].append(1.0)
            W[w_idx].append(1.0)
            col_offset += 1
    
    return U, V, W, col_offset

def _strassen_2x2x2(U, V, W, i_off, j_off, k_off, n, m, p):
    """Add a Strassen decomposition for a 2x2x2 block. Returns new rank offset."""
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # U rows: (i,j) -> i*m+j, but within a 2x2 block starting at i_off, j_off
    # V rows: (j,k) -> j*p+k, within block at j_off, k_off
    # W cols: (k,i) -> k*n+i, within block at k_off, i_off
    
    # We'll add 7 new columns
    rank_start = len(U[0]) if U else 0
    new_rank = 7
    
    # U indices: (i,j) for i in {0,1}, j in {0,1} -> row i*m + j within block
    # Actually, U row index = (i_off + i) * m + (j_off + j)
    # V row index = (j_off + j) * p + (k_off + k)
    # W col index = (k_off + k) * n + (i_off + i)
    
    def u_row(i, j):
        return (i_off + i) * m + (j_off + j)
    def v_row(j, k):
        return (j_off + j) * p + (k_off + k)
    def w_col(k, i):
        return (k_off + k) * n + (i_off + i)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For each term, we need entries in U, V, W
    # U entries are coefficients on A(i,j), V on B(j,k), W on C(k,i)
    
    terms = []
    
    # M1: U has A00=1, A11=1; V has B00=1, B11=1; W has C00=1, C01=1, C10=1, C11=1
    # C = M1 contributes to: C00 += 1, C01 += 1, C10 += 1, C11 += 1
    terms.append({
        'u': {0: 1, 3: 1},  # A00, A11
        'v': {0: 1, 3: 1},  # B00, B11
        'w': {0: 1, 1: 1, 2: 1, 3: 1}  # C00, C01, C10, C11
    })
    # M2: U has A10=1, A11=1; V has B00=1; W has C10=1, C11=1
    terms.append({
        'u': {2: 1, 3: 1},  # A10, A11
        'v': {0: 1},  # B00
        'w': {2: 1, 3: 1}  # C10, C11
    })
    # M3: U has A00=1; V has B10=1, B11=-1; W has C00=1, C01=-1
    terms.append({
        'u': {0: 1},  # A00
        'v': {2: 1, 3: -1},  # B10, B11
        'w': {0: 1, 1: -1}  # C00, C01
    })
    # M4: U has A11=1; V has B01=1, B10=-1; W has C10=1, C11=-1
    terms.append({
        'u': {3: 1},  # A11
        'v': {1: 1, 2: -1},  # B01, B10
        'w': {2: 1, 3: -1}  # C10, C11
    })
    # M5: U has A00=1, A01=1; V has B11=1; W has C00=1, C01=1
    terms.append({
        'u': {0: 1, 1: 1},  # A00, A01
        'v': {3: 1},  # B11
        'w': {0: 1, 1: 1}  # C00, C01
    })
    # M6: U has A10=1, A00=-1; V has B00=1, B10=1; W has C10=1, C11=1
    terms.append({
        'u': {2: 1, 0: -1},  # A10, A00
        'v': {0: 1, 2: 1},  # B00, B10
        'w': {2: 1, 3: 1}  # C10, C11
    })
    # M7: U has A01=1, A11=-1; V has B01=1, B11=1; W has C00=1, C01=1
    terms.append({
        'u': {1: 1, 3: -1},  # A01, A11
        'v': {1: 1, 3: 1},  # B01, B11
        'w': {0: 1, 1: 1}  # C00, C01
    })
    
    # Extend U, V, W with 7 new columns
    for row in range(len(U)):
        U[row] = U[row] + [0.0] * new_rank
    for row in range(len(V)):
        V[row] = V[row] + [0.0] * new_rank
    for row in range(len(W)):
        W[row] = W[row] + [0.0] * new_rank
    
    for r_idx, term in enumerate(terms):
        col = rank_start + r_idx
        for u_idx, coeff in term['u'].items():
            U[u_row(u_idx // 2, u_idx % 2)][col] = coeff
        for v_idx, coeff in term['v'].items():
            V[v_row(v_idx // 2, v_idx % 2)][col] = coeff
        for w_idx, coeff in term['w'].items():
            W[w_col(w_idx // 2, w_idx % 2)][col] = coeff
    
    return U, V, W

def _apply_strassen_blocks(n, m, p):
    """Apply Strassen to 2x2x2 blocks where possible, schoolbook for rest."""
    # For 3x4x4, we can partition as:
    # i: 3 rows -> use Strassen for first 2, schoolbook for 3rd
    # j: 4 cols -> use Strassen for first 2, Strassen for next 2
    # k: 4 rows -> use Strassen for first 2, Strassen for next 2
    
    # Start with empty
    U = []
    V = []
    W = []
    
    # We'll build it incrementally
    U = [[0.0]]  # placeholder, will rebuild
    V = [[0.0]]
    W = [[0.0]]
    
    # Actually, let's build properly
    U = [[0.0] for _ in range(n * m)]
    V = [[0.0] for _ in range(m * p)]
    W = [[0.0] for _ in range(p * n)]
    
    # Define which blocks to apply Strassen to
    # i blocks: [0,1] (Strassen), [2] (schoolbook)
    # j blocks: [0,1] (Strassen), [2,3] (Strassen)
    # k blocks: [0,1] (Strassen), [2,3] (Strassen)
    
    i_blocks = [(0, 1, 'S'), (2, 2, 'B')]  # (start, end, type)
    j_blocks = [(0, 1, 'S'), (2, 3, 'S')]
    k_blocks = [(0, 1, 'S'), (2, 3, 'S')]
    
    col_offset = 0
    
    for i_start, i_end, i_type in i_blocks:
        i_size = i_end - i_start
        for j_start, j_end, j_type in j_blocks:
            j_size = j_end - j_start
            for k_start, k_end, k_type in k_blocks:
                k_size = k_end - k_start
                
                if i_type == 'S' and j_type == 'S' and k_type == 'S' and i_size == 2 and j_size == 2 and k_size == 2:
                    # Apply Strassen
                    U, V, W = _strassen_2x2x2(U, V, W, i_start, j_start, k_start, n, m, p)
                    col_offset += 7
                else:
                    # Schoolbook for this block
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                U[i*m+j].append(1.0)
                                V[j*p+k].append(1.0)
                                W[k*n+i].append(1.0)
                                col_offset += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return None
    
    return U, V, W, col_offset

def _find_better_decomposition(n, m, p, time_limit):
    """Search for a decomposition with rank less than n*m*p."""
    start_time = time.time()
    best_rank = n * m * p
    best_result = None
    
    # Try Kronecker/Strassen block decomposition
    try:
        U, V, W, rank = _strassen_4x4()
        if _verify(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    except Exception:
        pass
    
    # Also try the older Strassen blocks approach
    try:
        result = _apply_strassen_blocks(n, m, p)
        if result is not None:
            U2, V2, W2, rank2 = result
            if rank2 < best_rank:
                best_rank = rank2
                best_result = (U2, V2, W2)
    except Exception:
        pass
    
    if best_result:
        return best_result
    else:
        U, V, W = _schoolbook_rank(n, m, p)
        return (U, V, W)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = 3, 4, 4
    
    time_limit = 120  # seconds
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + time_limit - 5  # 5 second safety margin
    
    # Try to find a better decomposition
    try:
        U, V, W = _find_better_decomposition(n, m, p, time_limit)
    except Exception:
        U, V, W = _schoolbook_rank(n, m, p)
    
    # Check if we're still within time
    if time.time() > deadline:
        U, V, W = _schoolbook_rank(n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook_rank(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
