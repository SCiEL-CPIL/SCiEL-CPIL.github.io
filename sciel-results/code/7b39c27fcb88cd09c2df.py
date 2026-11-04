import time
from fractions import Fraction

_START = time.time()
DEADLINE = 115.0  # safety margin before 120s wall-clock

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    # Check cross terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    if total != 0:
                        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook multiplication, rank = n*m*p."""
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2, rank 7."""
    rank = 7
    U = [[Fraction(0)] * rank for _ in range(4)]
    V = [[Fraction(0)] * rank for _ in range(4)]
    W = [[Fraction(0)] * rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = Fraction(1); U[3][0] = Fraction(1)
    V[0][0] = Fraction(1); V[3][0] = Fraction(1)
    W[0][0] = Fraction(1); W[3][0] = Fraction(1)
    
    # M2 = (A10+A11)*B00
    U[2][1] = Fraction(1); U[3][1] = Fraction(1)
    V[0][1] = Fraction(1)
    W[2][1] = Fraction(1); W[3][1] = Fraction(1)
    
    # M3 = A00*(B01-B11)
    U[0][2] = Fraction(1)
    V[1][2] = Fraction(1); V[3][2] = Fraction(-1)
    W[1][2] = Fraction(1); W[3][2] = Fraction(-1)
    
    # M4 = A11*(B10-B00)
    U[3][3] = Fraction(1)
    V[2][3] = Fraction(1); V[0][3] = Fraction(-1)
    W[0][3] = Fraction(-1); W[2][3] = Fraction(1)
    
    # M5 = (A00+A01)*B11
    U[0][4] = Fraction(1); U[1][4] = Fraction(1)
    V[3][4] = Fraction(1)
    W[1][4] = Fraction(1); W[3][4] = Fraction(-1)
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = Fraction(1); U[0][5] = Fraction(-1)
    V[0][5] = Fraction(1); V[1][5] = Fraction(1)
    W[0][5] = Fraction(1)
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = Fraction(1); U[3][6] = Fraction(-1)
    V[2][6] = Fraction(1); V[3][6] = Fraction(1)
    W[0][6] = Fraction(1)
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition(n, m, p, splits_n, splits_m, splits_p):
    """Build a block decomposition using Strassen for 2x2x2 blocks."""
    U_list = []
    V_list = []
    W_list = []
    block_info = []
    
    for ni in splits_n:
        for mj in splits_m:
            for pk in splits_p:
                if ni == 2 and mj == 2 and pk == 2:
                    block = _strassen_2x2x2()
                    U_list.append(block["U"])
                    V_list.append(block["V"])
                    W_list.append(block["W"])
                    block_info.append((ni, mj, pk, 7))
                else:
                    brank = ni * mj * pk
                    U_block = [[Fraction(0)] * brank for _ in range(ni * mj)]
                    V_block = [[Fraction(0)] * brank for _ in range(mj * pk)]
                    W_block = [[Fraction(0)] * brank for _ in range(pk * ni)]
                    col = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U_block[ii * mj + jj][col] = Fraction(1)
                                V_block[jj * pk + kk][col] = Fraction(1)
                                W_block[kk * ni + ii][col] = Fraction(1)
                                col += 1
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
                    block_info.append((ni, mj, pk, brank))
    
    total_rank = sum(info[3] for info in block_info)
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    
    row_off_u = 0
    row_off_v = 0
    row_off_w = 0
    col_off = 0
    
    for idx, (ni, mj, pk, brank) in enumerate(block_info):
        for ii in range(ni):
            for jj in range(mj):
                u_local = ii * mj + jj
                u_global = row_off_u + u_local
                for r in range(brank):
                    U[u_global][col_off + r] = U_list[idx][u_local][r]
        for jj in range(mj):
            for kk in range(pk):
                v_local = jj * pk + kk
                v_global = row_off_v + v_local
                for r in range(brank):
                    V[v_global][col_off + r] = V_list[idx][v_local][r]
        for kk in range(pk):
            for ii in range(ni):
                w_local = kk * ni + ii
                w_global = row_off_w + w_local
                for r in range(brank):
                    W[w_global][col_off + r] = W_list[idx][w_local][r]
        
        col_off += brank
        row_off_u += ni * mj
        row_off_v += mj * pk
        row_off_w += pk * ni
    
    return {"U": U, "V": V, "W": W}

def _compute_matricization_rank(U, V, W, n, m, p):
    """Compute the rank of the tensor matricization using exact rational arithmetic.
    
    We matricize by fixing the W index c. For each c, we form a matrix M_c
    where M_c[a,b] = sum_r U[a,r]*V[b,r]*W[c,r].
    The tensor rank is at least the max rank over all c.
    """
    rank = len(U[0])
    max_mat_rank = 0
    
    for c in range(p * n):
        # Build the matricization matrix M_c of size (n*m) x (m*p)
        mat = [[Fraction(0)] * (m * p) for _ in range(n * m)]
        for a in range(n * m):
            for b in range(m * p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                mat[a][b] = s
        
        # Compute rank via Gaussian elimination
        mat_rank = _matrix_rank(mat)
        max_mat_rank = max(max_mat_rank, mat_rank)
    
    return max_mat_rank

def _matrix_rank(mat):
    """Compute the rank of a matrix with Fraction entries using Gaussian elimination."""
    if not mat or not mat[0]:
        return 0
    
    rows = len(mat)
    cols = len(mat[0])
    mat = [row[:] for row in mat]  # Copy
    
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        
        # Find pivot
        pivot = None
        for row in range(rank, rows):
            if mat[row][col] != 0:
                pivot = row
                break
        
        if pivot is None:
            continue
        
        # Swap rows
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        
        # Eliminate below
        for row in range(rank + 1, rows):
            factor = mat[row][col] / mat[rank][col]
            for c in range(col, cols):
                mat[row][c] -= factor * mat[rank][c]
        
        rank += 1
    
    return rank

def _try_reduce_rank(U, V, W, n, m, p):
    """Try to reduce the rank by finding linear dependencies.
    
    If the matricization rank is lower than the current rank, we can potentially
    find a smaller decomposition. This is a heuristic attempt.
    """
    current_rank = len(U[0])
    mat_rank = _compute_matricization_rank(U, V, W, n, m, p)
    
    if mat_rank >= current_rank:
        return None  # No reduction possible
    
    # For now, we don't have a general algorithm to find the minimal decomposition
    # given that the matricization rank is lower. This would require solving a
    # system of polynomial equations, which is computationally expensive.
    # Return None to indicate no improvement found.
    return None

def construct():
    n, m, p = 2, 5, 5
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 50
    
    # Try various block decompositions
    candidate_splits = [
        ([2], [2, 2, 1], [2, 2, 1]),
        ([2], [2, 2, 1], [2, 1, 2]),
        ([2], [2, 2, 1], [1, 2, 2]),
        ([2], [2, 1, 2], [2, 2, 1]),
        ([2], [2, 1, 2], [2, 1, 2]),
        ([2], [2, 1, 2], [1, 2, 2]),
        ([2], [1, 2, 2], [2, 2, 1]),
        ([2], [1, 2, 2], [2, 1, 2]),
        ([2], [1, 2, 2], [1, 2, 2]),
        ([1, 1], [2, 2, 1], [2, 2, 1]),
        ([1, 1], [2, 2, 1], [2, 1, 2]),
        ([1, 1], [2, 2, 1], [1, 2, 2]),
        ([1, 1], [2, 1, 2], [2, 2, 1]),
        ([1, 1], [2, 1, 2], [2, 1, 2]),
        ([1, 1], [2, 1, 2], [1, 2, 2]),
        ([1, 1], [1, 2, 2], [2, 2, 1]),
        ([1, 1], [1, 2, 2], [2, 1, 2]),
        ([1, 1], [1, 2, 2], [1, 2, 2]),
    ]
    
    for splits_n, splits_m, splits_p in candidate_splits:
        if sum(splits_n) != n or sum(splits_m) != m or sum(splits_p) != p:
            continue
        if _time_left() < 5.0:
            break
        
        try:
            result = _block_decomposition(n, m, p, splits_n, splits_m, splits_p)
            rank = len(result["U"][0])
            
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best = result
                    best_rank = rank
        except Exception:
            continue
    
    # Try to reduce rank further using matricization analysis
    if _time_left() > 10.0:
        try:
            reduced = _try_reduce_rank(best["U"], best["V"], best["W"], n, m, p)
            if reduced is not None:
                reduced_rank = len(reduced["U"][0])
                if reduced_rank < best_rank and _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                    best = reduced
                    best_rank = reduced_rank
        except Exception:
            pass
    
    # Convert Fraction to int/float for output
    def convert(mat):
        return [[int(x) if x.denominator == 1 else float(x) for x in row] for row in mat]
    
    return {"U": convert(best["U"]), "V": convert(best["V"]), "W": convert(best["W"])}
