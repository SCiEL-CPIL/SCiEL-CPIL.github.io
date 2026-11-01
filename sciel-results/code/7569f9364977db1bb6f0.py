import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for 2x4 by 4x5 matrix multiplication.
    Strategy: Search for optimal block tilings of the input matrices.
    For each tiling, estimate the rank using a cost model that uses Strassen-like 
    decompositions for square blocks and schoolbook for rectangular ones.
    Construct the explicit decomposition for the best tiling found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Search for the best tiling
    # A tiling is defined by:
    # - Partitions of n (rows of A)
    # - Partitions of m (cols of A / rows of B)
    # - Partitions of p (cols of B)
    
    def get_partitions(dim):
        """Generate all possible integer partitions of 'dim' into parts >= 1."""
        if dim == 1:
            return [[1]]
        parts = []
        for i in range(1, dim + 1):
            for rest in get_partitions(dim - i):
                parts.append([i] + rest)
        return parts

    # Precompute partitions for small dimensions (2, 4, 5 are small)
    parts_n = get_partitions(n)
    parts_m = get_partitions(m)
    parts_p = get_partitions(p)
    
    best_rank = n * m * p
    best_tiling = (list(range(n)), list(range(m)), list(range(p))) # Fallback to schoolbook (1x1 blocks)
    
    # Cost model for rank of multiplying (r1 x c1) by (r2 x c2)
    # We assume we can use Strassen for square blocks if size >= 2.
    # For rectangular, we use a heuristic: min(r1*c1*c2, strassen_cost_if_square)
    # Actually, for small sizes, schoolbook is often best or close.
    # Let's use a simple model:
    # If r1==c1==r2==c2 (square), rank = 7 * (size/2)^omega ? No, Strassen is for 2x2.
    # For 2x2x2, rank 7.
    # For 1x1x1, rank 1.
    # For rectangular, schoolbook is usually optimal or near optimal for small sizes.
    # Let's just use schoolbook rank r1*r2*c2 for the cost estimation, 
    # but allow "Strassen" reduction if the block is exactly 2x2x2.
    
    def estimate_block_rank(r1, c1, r2, c2):
        # r1 x c1 times r2 x c2
        # Note: c1 must equal r2 for multiplication to be valid.
        if c1 != r2:
            return float('inf')
        
        # If it's a 2x2x2 block, use Strassen rank 7
        if r1 == 2 and c1 == 2 and r2 == 2 and c2 == 2:
            return 7
        
        # Otherwise, use schoolbook
        return r1 * c1 * c2

    # Iterate over all tilings
    for pn in parts_n:
        for pm in parts_m:
            for pp in parts_p:
                # Check if time is running out
                if time.time() > deadline - 5.0:
                    break
                
                # Calculate total rank for this tiling
                total_rank = 0
                valid = True
                for i in range(len(pn)):
                    for j in range(len(pm)):
                        for k in range(len(pp)):
                            r1 = pn[i]
                            c1 = pm[j]
                            r2 = pm[j] # Rows of B block corresponding to cols of A block
                            c2 = pp[k]
                            
                            # The block multiplication is (r1 x c1) * (r2 x c2)
                            # This requires c1 == r2, which is guaranteed by pm[j]
                            rank = estimate_block_rank(r1, c1, r2, c2)
                            total_rank += rank
                            
                            if total_rank >= best_rank:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid and total_rank < best_rank:
                    best_rank = total_rank
                    best_tiling = (pn, pm, pp)
                    
    # 2. Construct the decomposition for the best tiling
    pn, pm, pp = best_tiling
    
    # We need to map the "block indices" to "global indices"
    # pn[i] is the height of block i in A (and C)
    # pm[j] is the width of block j in A (and height in B)
    # pp[k] is the width of block k in B (and C)
    
    # Precompute start indices
    n_starts = [0] * (len(pn) + 1)
    for i in range(len(pn)):
        n_starts[i+1] = n_starts[i] + pn[i]
        
    m_starts = [0] * (len(pm) + 1)
    for j in range(len(pm)):
        m_starts[j+1] = m_starts[j] + pm[j]
        
    p_starts = [0] * (len(pp) + 1)
    for k in range(len(pp)):
        p_starts[k+1] = p_starts[k] + pp[k]

    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    
    rank_idx = 0
    
    for i in range(len(pn)):
        for j in range(len(pm)):
            for k in range(len(pp)):
                r1 = pn[i]
                c1 = pm[j]
                r2 = pm[j]
                c2 = pp[k]
                
                # Determine the specific decomposition for this block
                if r1 == 2 and c1 == 2 and r2 == 2 and c2 == 2:
                    # Strassen decomposition for 2x2 * 2x2
                    # A = [[a11, a12], [a21, a22]]
                    # B = [[b11, b12], [b21, b22]]
                    # C = A*B
                    #
                    # M1 = (a11 + a22)(b11 + b22)
                    # M2 = (a21 + a22)b11
                    # M3 = a11(b12 - b22)
                    # M4 = a22(b21 - b11)
                    # M5 = (a11 + a12)b22
                    # M6 = (a21 - a11)(b11 + b12)
                    # M7 = (a12 - a22)(b21 + b22)
                    #
                    # c11 = M1 + M4 - M5 + M7
                    # c12 = M3 + M5
                    # c21 = M2 + M4
                    # c22 = M1 - M2 + M3 + M6
                    
                    # We need to add 7 rank-1 terms to U, V, W.
                    # The indices in the global matrices are:
                    # A block is at rows n_starts[i] to n_starts[i+1]-1, cols m_starts[j] to m_starts[j+1]-1
                    # B block is at rows m_starts[j] to m_starts[j+1]-1, cols p_starts[k] to p_starts[k+1]-1
                    # C block is at rows p_starts[k] to p_starts[k+1]-1, cols n_starts[i] to n_starts[i+1]-1
                    
                    # Local indices within the 2x2 block:
                    # a_00, a_01, a_10, a_11
                    # b_00, b_01, b_10, b_11
                    # c_00, c_01, c_10, c_11
                    
                    # Strassen coefficients
                    # M1: (a00+a11)(b00+b11)
                    # M2: (a10+a11)b00
                    # M3: a00(b01-b11)
                    # M4: a11(b10-b00)
                    # M5: (a00+a01)b11
                    # M6: (a10-a00)(b00+b01)
                    # M7: (a01-a11)(b10+b11)
                    
                    strassen_ops = [
                        # (U_coeff, V_coeff, W_coeff) for each of the 7 products
                        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
                        
                        # M1 = (a00+a11) * (b00+b11)
                        # U part: a00 and a11 contribute to this term
                        # V part: b00 and b11 contribute
                        # W part: c00, c01, c10, c11 contribute? No, W is the output.
                        # The tensor decomposition is sum of u_t v_t w_t^T
                        # For each term t, U[:, t] is a vector in A-space, V[:, t] in B-space, W[:, t] in C-space.
                        
                        # Let's define the 7 terms explicitly.
                        # Term 1: M1
                        # U: a00=1, a11=1
                        # V: b00=1, b11=1
                        # W: c00=1, c11=1 (from c11=M1+..., c22=M1-...)
                        # Wait, c11 (local index 0,0) = M1 + M4 - M5 + M7
                        # c12 (local index 0,1) = M3 + M5
                        # c21 (local index 1,0) = M2 + M4
                        # c22 (local index 1,1) = M1 - M2 + M3 + M6
                        
                        # So for M1, W has 1 at c00 and 1 at c11.
                        {
                            "u": [(0, 1), (1, 1)], # (local_row, coeff)
                            "v": [(0, 1), (1, 1)],
                            "w": [(0, 1), (1, 1)]  # c00, c11
                        },
                        # Term 2: M2 = (a10+a11)b00
                        {
                            "u": [(1, 1), (0, 1)], # a10, a11? No, a10 is row 1 col 0. a11 is row 1 col 1.
                            # Correction: a10 is (1,0), a11 is (1,1).
                            "u": [(1, 1), (1, 1)], # a10, a11
                            "v": [(0, 1)],          # b00
                            "w": [(1, 1), (1, -1)]  # c21 (M2+...), c22 (M1-M2+...) -> M2 has +1 in c21, -1 in c22
                        },
                        # Term 3: M3 = a00(b01-b11)
                        {
                            "u": [(0, 1)],          # a00
                            "v": [(0, 1), (1, -1)], # b01, b11
                            "w": [(1, 1), (1, 1)]   # c12 (M3+M5), c22 (M1-M2+M3+M6) -> M3 has +1 in c12, +1 in c22
                        },
                        # Term 4: M4 = a11(b10-b00)
                        {
                            "u": [(1, 1)],          # a11
                            "v": [(1, 1), (0, -1)], # b10, b00
                            "w": [(0, 1), (1, 1)]   # c11 (M1+M4...), c21 (M2+M4) -> M4 has +1 in c11, +1 in c21
                        },
                        # Term 5: M5 = (a00+a01)b11
                        {
                            "u": [(0, 1), (0, 1)],  # a00, a01
                            "v": [(1, 1)],          # b11
                            "w": [(0, -1), (1, 1)]  # c11 (M1+M4-M5+M7), c12 (M3+M5) -> M5 has -1 in c11, +1 in c12
                        },
                        # Term 6: M6 = (a10-a00)(b00+b01)
                        {
                            "u": [(1, 1), (0, -1)], # a10, a00
                            "v": [(0, 1), (0, 1)],  # b00, b01
                            "w": [(1, 1)]           # c22 (M1-M2+M3+M6) -> M6 has +1 in c22
                        },
                        # Term 7: M7 = (a01-a11)(b10+b11)
                        {
                            "u": [(0, 1), (1, -1)], # a01, a11
                            "v": [(1, 1), (1, 1)],  # b10, b11
                            "w": [(0, 1)]           # c11 (M1+M4-M5+M7) -> M7 has +1 in c11
                        }
                    ]
                    
                    for op in strassen_ops:
                        # Add to U
                        for (lr, coeff) in op["u"]:
                            # Global index in U: row = n_starts[i] + lr, col = m_starts[j] + lc?
                            # No, U is (n*m, rank). The index is i_global * m + j_global.
                            # Local row lr, local col lc.
                            # In my op["u"], I only provided (row, coeff). I need col too.
                            # Let's refine the op structure to include (row, col, coeff).
                            pass
                    
                    # Let's redo the Strassen addition with correct indices.
                    # I will define a helper to add a rank-1 term.
                    def add_term(u_indices, v_indices, w_indices):
                        nonlocal rank_idx
                        for (lr, lc, coeff) in u_indices:
                            global_row = n_starts[i] + lr
                            global_col = m_starts[j] + lc
                            U[global_row * m + global_col, rank_idx] = coeff
                        for (lr, lc, coeff) in v_indices:
                            global_row = m_starts[j] + lr
                            global_col = p_starts[k] + lc
                            V[global_row * p + global_col, rank_idx] = coeff
                        for (lr, lc, coeff) in w_indices:
                            global_row = p_starts[k] + lr
                            global_col = n_starts[i] + lc
                            W[global_row * n + global_col, rank_idx] = coeff
                        rank_idx += 1

                    # M1
                    add_term(
                        [(0,0,1), (1,1,1)],
                        [(0,0,1), (1,1,1)],
                        [(0,0,1), (1,1,1)]
                    )
                    # M2
                    add_term(
                        [(1,0,1), (1,1,1)],
                        [(0,0,1)],
                        [(1,0,1), (1,1,-1)]
                    )
                    # M3
                    add_term(
                        [(0,0,1)],
                        [(0,1,1), (1,1,-1)],
                        [(0,1,1), (1,1,1)]
                    )
                    # M4
                    add_term(
                        [(1,1,1)],
                        [(1,0,1), (0,0,-1)],
                        [(0,0,1), (1,0,1)]
                    )
                    # M5
                    add_term(
                        [(0,0,1), (0,1,1)],
                        [(1,1,1)],
                        [(0,0,-1), (0,1,1)]
                    )
                    # M6
                    add_term(
                        [(1,0,1), (0,0,-1)],
                        [(0,0,1), (0,1,1)],
                        [(1,1,1)]
                    )
                    # M7
                    add_term(
                        [(0,1,1), (1,1,-1)],
                        [(1,0,1), (1,1,1)],
                        [(0,0,1)]
                    )

                else:
                    # Schoolbook for this block
                    for lr in range(r1):
                        for lc in range(c1):
                            for lc2 in range(c2):
                                add_term(
                                    [(lr, lc, 1)],
                                    [(lc, lc2, 1)], # Note: B block row is lc (which corresponds to A col), B block col is lc2
                                    [(lc2, lr, 1)]  # C block row is lc2, C block col is lr
                                )
                                # Wait, the helper `add_term` increments rank_idx.
                                # So this loop adds r1*c1*c2 terms.
                                # But `add_term` is defined inside the `if` block? No, I defined it inside the `if` block in the draft.
                                # I need to move `add_term` out or redefine it.
                                # Let's restructure to avoid nested function definition issues in the loop.
                                # Actually, `add_term` uses `rank_idx` from the outer scope.
                                # I will define `add_term` once outside the loops.

    # The above code structure is flawed because `add_term` was defined inside the if block.
    # Let's rewrite the construction part cleanly.
    
    # Reset U, V, W and rank_idx
    U = np.zeros((n * m, best_rank))
    V = np.zeros((m * p, best_rank))
    W = np.zeros((p * n, best_rank))
    rank_idx = 0

    def add_term(u_indices, v_indices, w_indices):
        nonlocal rank_idx
        for (lr, lc, coeff) in u_indices:
            global_row = n_starts[i] + lr
            global_col = m_starts[j] + lc
            U[global_row * m + global_col, rank_idx] = coeff
        for (lr, lc, coeff) in v_indices:
            global_row = m_starts[j] + lr
            global_col = p_starts[k] + lc
            V[global_row * p + global_col, rank_idx] = coeff
        for (lr, lc, coeff) in w_indices:
            global_row = p_starts[k] + lr
            global_col = n_starts[i] + lc
            W[global_row * n + global_col, rank_idx] = coeff
        rank_idx += 1

    for i in range(len(pn)):
        for j in range(len(pm)):
            for k in range(len(pp)):
                r1 = pn[i]
                c1 = pm[j]
                r2 = pm[j]
                c2 = pp[k]
                
                if r1 == 2 and c1 == 2 and r2 == 2 and c2 == 2:
                    # Strassen
                    add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
                    add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
                    add_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
                    add_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
                    add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
                    add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
                    add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
                else:
                    # Schoolbook
                    for lr in range(r1):
                        for lc in range(c1):
                            for lc2 in range(c2):
                                add_term(
                                    [(lr, lc, 1)],
                                    [(lc, lc2, 1)],
                                    [(lc2, lr, 1)]
                                )
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
