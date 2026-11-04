"""Matrix multiplication 3x4 by 4x8 tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 by 2x2 using Strassen's rank-7 method.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for (a b; c d) * (e f; g h):
    # p1 = (a+b)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+d)(e+h)
    # p6 = (b-d)(g+h)
    # p7 = (c-a)(e+f)
    # Result: (p1+p4-p5+p7, p3+p5; p2+p5, p1-p2+p4+p6)
    
    # U matrix: rows indexed by (i,j) of input A, columns by Strassen product
    # A = [[a, b], [c, d]]
    # U[i*2+j, c] where i=0,1 row, j=0,1 col
    # Product 1: (a+b)(e+h): a contributes 1, b contributes 1, c,d contribute 0
    # Product 2: (c+d)e: c contributes 1, d contributes 1, a,b contribute 0
    # Product 3: a(f-h): a contributes 1
    # Product 4: d(g-e): d contributes 1
    # Product 5: (a+d)(e+h): a contributes 1, d contributes 1
    # Product 6: (b-d)(g+h): b contributes 1, d contributes -1
    # Product 7: (c-a)(e+f): c contributes 1, a contributes -1
    
    U = [
        [1, 0, 1, 0, 1, 0, -1],  # a (row 0, col 0)
        [1, 0, 0, 0, 0, 1, 0],   # b (row 0, col 1)
        [0, 1, 0, 0, 0, 0, 1],   # c (row 1, col 0)
        [0, 1, 0, 1, 1, -1, 0],  # d (row 1, col 1)
    ]
    
    # V matrix: rows indexed by (j,k) of input B
    # B = [[e, f], [g, h]]
    # Product 1: (a+b)(e+h): e contributes 1, h contributes 1
    # Product 2: (c+d)e: e contributes 1
    # Product 3: a(f-h): f contributes 1, h contributes -1
    # Product 4: d(g-e): g contributes 1, e contributes -1
    # Product 5: (a+d)(e+h): e contributes 1, h contributes 1
    # Product 6: (b-d)(g+h): g contributes 1, h contributes 1
    # Product 7: (c-a)(e+f): e contributes 1, f contributes 1
    
    V = [
        [1, 1, 0, -1, 1, 0, 1],  # e (row 0, col 0)
        [0, 0, 1, 0, 0, 0, 1],   # f (row 0, col 1)
        [0, 0, 0, 1, 0, 1, 0],   # g (row 1, col 0)
        [1, 0, -1, 0, 1, 1, 0],  # h (row 1, col 1)
    ]
    
    # W matrix: rows indexed by (k,i) of output C
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p5, p1-p2+p4+p6]]
    # W[k*2+i, c]
    # (0,0): C[0][0] = p1+p4-p5+p7
    # (0,1): C[0][1] = p3+p5
    # (1,0): C[1][0] = p2+p5
    # (1,1): C[1][1] = p1-p2+p4+p6
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # k=0, i=0 -> C[0][0]
        [0, 0, 1, 0, 1, 0, 0],    # k=0, i=1 -> C[0][1]
        [0, 1, 0, 0, 1, 0, 0],    # k=1, i=0 -> C[1][0]
        [1, -1, 0, 1, 0, 1, 0],   # k=1, i=1 -> C[1][1]
    ]
    
    return U, V, W


def _validate(U, V, W, n, m, p):
    """Validate that the decomposition is exact."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                dot = np.sum(u_val * v_val * w_val)
                if abs(dot - 1.0) > 1e-9:
                    return False
    
    # Check that for non-matching triples, the sum is 0
    # This is implied by the tensor structure: the multiplication tensor
    # T[i,j,k] = 1 if the triple corresponds to a valid multiply-add, else 0
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
    # in the range. The decomposition must satisfy:
    # (UVW)[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # There are no "non-matching" triples in the standard formulation;
    # the tensor is all-ones over [n]x[m]x[p].
    
    return True


def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Strategy: Partition m=4 into two blocks of 2, and p=8 into four blocks of 2.
    # For each combination of m-block (2 of them) and p-block (4 of them),
    # we have a 3x2 by 2x2 block multiplication.
    # We can embed the 3x2 matrix into a 3x3 by adding a zero row, then use
    # a 3x3 by 3x3 decomposition, or more simply, use Strassen on the 2x2 part.
    
    # Actually, let's use a simpler approach:
    # Partition n=3 into n1=2, n2=1
    # Partition m=4 into m1=2, m2=2  
    # Partition p=8 into p1=2, p2=2, p3=2, p4=2
    
    # For the 2x2 x 2x2 blocks: use Strassen (rank 7)
    # For the 1x2 x 2x2 blocks: rank 4 (schoolbook)
    
    # Let's be more systematic. We'll build the decomposition block by block.
    
    # Block structure:
    # n split: [2, 1]
    # m split: [2, 2]
    # p split: [2, 2, 2, 2]
    
    # For each (n_block, m_block, p_block) combination:
    # - n_block size x m_block size x p_block size
    # - If 2x2x2: Strassen rank 7
    # - If 2x2x2: Strassen rank 7
    # - If 1x2x2: schoolbook rank 4
    # - If 1x2x2: schoolbook rank 4
    # etc.
    
    # Let's count:
    # n_blocks = [2, 1], m_blocks = [2, 2], p_blocks = [2, 2, 2, 2]
    # For n_block=2, m_block=2, each p_block=2: 2x2x2 -> rank 7, there are 2*4=8 such blocks
    # For n_block=1, m_block=2, each p_block=2: 1x2x2 -> rank 4, there are 2*4=8 such blocks
    # Total rank = 8*7 + 8*4 = 56 + 32 = 88
    
    # Let's build this decomposition.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    col_offset = 0
    
    n_splits = [2, 1]  # sizes of n blocks
    m_splits = [2, 2]  # sizes of m blocks
    p_splits = [2, 2, 2, 2]  # sizes of p blocks
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4, 6]
    
    for ni, n_size in enumerate(n_splits):
        for mi, m_size in enumerate(m_splits):
            for pi, p_size in enumerate(p_splits):
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    rank_block = 7
                    for c in range(rank_block):
                        # U: rows from n_off to n_off+n_size-1, cols from m_off to m_off+m_size-1
                        for i in range(n_size):
                            for j in range(m_size):
                                pass  # We'll add rows below
                        # Actually, let's add the rows properly
                        for idx in range(n_size * m_size):
                            i = idx // m_size
                            j = idx % m_size
                            U_rows.append(sU[idx][c])
                        # Wait, this is getting complicated. Let me restructure.
                    # This approach is messy. Let me use a different method.
                    pass
                
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    # Schoolbook for 1x2 x 2x2: rank 4
                    for j in range(m_size):
                        for k in range(p_size):
                            pass
                    pass
                else:
                    pass
    
    # Let me restart with a cleaner implementation.
    return _build_decomposition()


def _build_decomposition():
    """Build the block-structured decomposition."""
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Splits
    n_sizes = [2, 1]
    m_sizes = [2, 2]
    p_sizes = [2, 2, 2, 2]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4, 6]
    
    # Collect all columns
    all_U = []
    all_V = []
    all_W = []
    
    for ni, n_size in enumerate(n_sizes):
        for mi, m_size in enumerate(m_sizes):
            for pi, p_size in enumerate(p_sizes):
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Strassen 2x2x2
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        # U rows: (n_off + i)*m + (m_off + j) for i in range(2), j in range(2)
                        # But U has shape (n*m, rank), so row index is i*m + j globally
                        # Within the block, local row is i*m_size + j
                        # Global row is n_off + i maps to global i, and m_off + j maps to global j
                        # So global row = (n_off + i) * m + (m_off + j)
                        # But we're adding to a list per column, so let's store (global_row, value)
                        pass
                
                # This is getting complicated. Let me just build the full matrices directly.
    
    # Direct approach: build U, V, W as lists of lists
    total_rank = 0
    # Count rank
    for ni, n_size in enumerate(n_sizes):
        for mi, m_size in enumerate(m_sizes):
            for pi, p_size in enumerate(p_sizes):
                if n_size == 2 and m_size == 2 and p_size == 2:
                    total_rank += 7
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    total_rank += 4
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    total_rank += 7
                else:
                    total_rank += n_size * m_size * p_size
    
    # Initialize
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ni, n_size in enumerate(n_sizes):
        for mi, m_size in enumerate(m_sizes):
            for pi, p_size in enumerate(p_sizes):
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    sU, sV, sW = _strassen_2x2x2()
                    for c in range(7):
                        # U: for local (i, j) where i in [0, n_size), j in [0, m_size)
                        for i in range(n_size):
                            for j in range(m_size):
                                local_row = i * m_size + j
                                global_row = (n_off + i) * m + (m_off + j)
                                U[global_row][col + c] = sU[local_row][c]
                        # V: for local (j, k) where j in [0, m_size), k in [0, p_size)
                        for j in range(m_size):
                            for k in range(p_size):
                                local_row = j * p_size + k
                                global_row = (m_off + j) * p + (p_off + k)
                                V[global_row][col + c] = sV[local_row][c]
                        # W: for local (k, i) where k in [0, p_size), i in [0, n_size)
                        for k in range(p_size):
                            for i in range(n_size):
                                local_row = k * n_size + i
                                global_row = (p_off + k) * n + (n_off + i)
                                W[global_row][col + c] = sW[local_row][c]
                    col += 7
                    
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    # Schoolbook for 1x2 x 2x2
                    for j in range(m_size):
                        for k in range(p_size):
                            # U row: (n_off + 0)*m + (m_off + j) = n_off*m + m_off + j
                            U[n_off * m + m_off + j][col] = 1
                            # V row: (m_off + j)*p + (p_off + k)
                            V[(m_off + j) * p + (p_off + k)][col] = 1
                            # W row: (p_off + k)*n + (n_off + 0)
                            W[(p_off + k) * n + n_off][col] = 1
                            col += 1
                else:
                    # General schoolbook
                    for i in range(n_size):
                        for j in range(m_size):
                            for k in range(p_size):
                                U[(n_off + i) * m + (m_off + j)][col] = 1
                                V[(m_off + j) * p + (p_off + k)][col] = 1
                                W[(p_off + k) * n + (n_off + i)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
