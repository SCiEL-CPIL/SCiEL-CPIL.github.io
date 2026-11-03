"""Matrix multiplication 3x4x8 with rank-reduced decomposition using Strassen blocks and column merging."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_decomposition_3x2x2():
    """Build a 3x2x2 decomposition using Strassen + schoolbook (rank 11)."""
    n, m, p = 3, 2, 2
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    rank = 7 + 4  # 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for rows 0,1 of A
    for i in range(4):
        for c in range(7):
            U[i][c] += U2[i][c]
            V[i][c] += V2[i][c]
            W[i][c] += W2[i][c]
    
    # Schoolbook for row 2 of A
    for i in range(2):
        for c in range(4):
            U[4 + i][7 + c] += U1[i][c]
    for i in range(4):
        for c in range(4):
            V[i][7 + c] += V1[i][c]
    for i in range(2):
        for c in range(4):
            W[4 + i][7 + c] += W1[i][c]
    
    return U, V, W, rank


def _build_full_decomposition(block_decomp):
    """Build the full 3x4x8 decomposition from a 3x2x2 block decomposition."""
    n, m, p = 3, 4, 8
    U_block, V_block, W_block, r = block_decomp
    
    total_rank = 8 * r  # 8 blocks (2 jp * 4 kp)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for jp in range(2):
        for kp in range(4):
            col_off = (jp * 4 + kp) * r
            j0 = 2 * jp
            k0 = 2 * kp
            
            # Map block indices to full indices
            # Block A: rows 0..5 (i*2+j for i in 0..2, j in 0..1)
            # Full A: i*m+j for i in 0..2, j in 0..3
            for bi in range(6):
                bi_i = bi // 2
                bi_j = bi % 2
                full_i = bi_i
                full_j = j0 + bi_j
                for c in range(r):
                    U[full_i * m + full_j][col_off + c] += U_block[bi][c]
            
            # Block B: rows 0..3 (j*2+k for j in 0..1, k in 0..1)
            # Full B: j*p+k for j in 0..3, k in 0..7
            for bi in range(4):
                bi_j = bi // 2
                bi_k = bi % 2
                full_j = j0 + bi_j
                full_k = k0 + bi_k
                for c in range(r):
                    V[full_j * p + full_k][col_off + c] += V_block[bi][c]
            
            # Block W: rows 0..5 (k*3+i for k in 0..1, i in 0..2)
            # Full W: k*n+i for k in 0..7, i in 0..2
            for bi in range(6):
                bi_k = bi // 3
                bi_i = bi % 3
                full_k = k0 + bi_k
                full_i = bi_i
                for c in range(r):
                    W[full_k * n + full_i][col_off + c] += W_block[bi][c]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1) > tol:
                    return False
    return True


def _try_merge_columns(U, V, W, n, m, p, time_limit=None):
    """Try to find and remove redundant columns from the decomposition.
    
    Two columns c1, c2 can be merged (remove c1) if:
    U[:,c1] = a * U[:,c2], V[:,c1] = b * V[:,c2], W[:,c1] = c * W[:,c2]
    and a * b * c = 1
    """
    rank = len(U[0])
    n_keep = list(range(rank))
    
    if time_limit is None:
        time_limit = 100
    
    start = time.time()
    removed = 0
    
    # For each pair of columns, check if one can be expressed as a linear combination
    # that doesn't affect the tensor product
    # Simpler: try removing each column and see if the decomposition still works
    # (this only works if a column is exactly zero in the tensor product sense)
    
    # Better: for each column c, check if there exist scalars alpha, beta for another column c'
    # such that the tensor contribution of c equals alpha * tensor contribution of c'
    
    new_U = [row[:] for row in U]
    new_V = [row[:] for row in V]
    new_W = [row[:] for row in W]
    
    # Try removing columns that are linearly dependent in a specific way
    # For column c, its contribution is: sum over (i,j,k) of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Two columns c1, c2 can be merged if U[:,c1] = a*U[:,c2], V[:,c1]=b*V[:,c2], W[:,c1]=c*W[:,c2]
    # and a*b*c = 1 (for the product to match)
    
    for c1 in range(rank):
        if time.time() > time_limit:
            break
        if c1 not in n_keep:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 not in n_keep:
                continue
            # Check if column c1 is proportional to column c2 in all three matrices
            # with product of ratios = 1
            try:
                # Find ratio for U
                ratio_u = None
                for idx in range(len(U)):
                    if new_U[idx][c2] != 0:
                        if new_U[idx][c1] == 0:
                            ratio_u = 0
                            break
                        r = new_U[idx][c1] / new_U[idx][c2]
                        if ratio_u is None:
                            ratio_u = r
                        elif abs(r - ratio_u) > 1e-9:
                            ratio_u = None
                            break
                    else:
                        if new_U[idx][c1] != 0:
                            ratio_u = None
                            break
                
                if ratio_u is None:
                    continue
                
                # Find ratio for V
                ratio_v = None
                for idx in range(len(V)):
                    if new_V[idx][c2] != 0:
                        if new_V[idx][c1] == 0:
                            ratio_v = 0
                            break
                        r = new_V[idx][c1] / new_V[idx][c2]
                        if ratio_v is None:
                            ratio_v = r
                        elif abs(r - ratio_v) > 1e-9:
                            ratio_v = None
                            break
                    else:
                        if new_V[idx][c1] != 0:
                            ratio_v = None
                            break
                
                if ratio_v is None:
                    continue
                
                # Find ratio for W
                ratio_w = None
                for idx in range(len(W)):
                    if new_W[idx][c2] != 0:
                        if new_W[idx][c1] == 0:
                            ratio_w = 0
                            break
                        r = new_W[idx][c1] / new_W[idx][c2]
                        if ratio_w is None:
                            ratio_w = r
                        elif abs(r - ratio_w) > 1e-9:
                            ratio_w = None
                            break
                    else:
                        if new_W[idx][c1] != 0:
                            ratio_w = None
                            break
                
                if ratio_w is None:
                    continue
                
                # Check product: ratio_u * ratio_v * ratio_w should equal 1
                if abs(ratio_u * ratio_v * ratio_w - 1.0) > 1e-9:
                    continue
                
                # Columns are mergeable! Remove c1
                n_keep.remove(c1)
                removed += 1
                break  # Restart c2 loop after removal
                
            except (ZeroDivisionError, IndexError):
                continue
    
    if removed == 0:
        return U, V, W, rank
    
    # Build new matrices
    new_rank = len(n_keep)
    final_U = [[new_U[i][c] for c in n_keep] for i in range(len(U))]
    final_V = [[new_V[i][c] for c in n_keep] for i in range(len(V))]
    final_W = [[new_W[i][c] for c in n_keep] for i in range(len(W))]
    
    return final_U, final_V, final_W, new_rank


def _try_reduce_rank(U, V, W, n, m, p, max_iter=50, time_limit=100):
    """Iteratively try to reduce rank by finding redundant columns."""
    start = time.time()
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break
        U, V, W, new_rank = _try_merge_columns(U, V, W, n, m, p, time_limit)
        if new_rank == len(U[0]):
            break
        # Verify after reduction
        if not _verify_decomposition(U, V, W, n, m, p):
            break
    return U, V, W


def _build_decomposition_fallback():
    """Original fallback: 8 Strassen + 8 schoolbook blocks."""
    n, m, p = 3, 4, 8
    total_rank = 8 * 7 + 8 * 4  # 88
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    col_offset = 0
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            # Strassen for i=0,1
            u_rows = [0 * m + j0, 0 * m + j0 + 1, 1 * m + j0, 1 * m + j0 + 1]
            v_rows = [j0 * p + k0, j0 * p + k0 + 1, (j0+1) * p + k0, (j0+1) * p + k0 + 1]
            w_rows = [k0 * n + 0, (k0+1) * n + 0, k0 * n + 1, (k0+1) * n + 1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            k0 = 2 * kp
            # Schoolbook for i=2
            u_rows = [2 * m + j0, 2 * m + j0 + 1]
            v_rows = [j0 * p + k0, (j0+1) * p + k0, j0 * p + k0 + 1, (j0+1) * p + k0 + 1]
            w_rows = [k0 * n + 2, (k0+1) * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
    
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the base 3x2x2 block decomposition (rank 11)
    U_block, V_block, W_block, r_block = _build_decomposition_3x2x2()
    
    # Build full decomposition
    result = _build_full_decomposition((U_block, V_block, W_block, r_block))
    
    # Verify
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to original construction
        U, V, W = _build_decomposition_fallback()
        result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() < deadline - 20:
        result["U"], result["V"], result["W"] = _try_reduce_rank(
            result["U"], result["V"], result["W"], n, m, p,
            max_iter=100, time_limit=deadline - 20
        )
    
    # Final verification
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Ultimate fallback: schoolbook
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
        result = {"U": U, "V": V, "W": W}
    
    return result
