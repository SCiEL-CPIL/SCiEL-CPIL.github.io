"""3x5x6 matrix multiplication with maximal 2x2x2 Strassen tiling.
Uses rank-7 Strassen for 2x2x2 sub-blocks where possible, schoolbook elsewhere.
Maximizes the number of 2x2x2 blocks to minimize total rank."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    terms = [
        ([1,0, 0,1], [1,0, 0,1], [1,0, 0,1]),
        ([0,0, 1,1], [1,0, 0,0], [0,0, 0,1]),
        ([1,0, 0,0], [0,1, 0,0], [1,0, 0,0]),
        ([0,0, 0,1], [1,0, 0,0], [0,1, 0,0]),
        ([0,0, 1,0], [0,1, 0,0], [1,0, 0,0]),
        ([1,0, 0,0], [0,0, 1,1], [0,0, 0,1]),
        ([0,1, 0,0], [0,0, 1,0], [0,1, 0,0]),
    ]
    U = [list(t[0]) for t in terms]
    V = [list(t[1]) for t in terms]
    W = [list(t[2]) for t in terms]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U_list = []
    V_list = []
    W_list = []
    U2, V2, W2 = _strassen_2x2x2()
    
    def add_strassen(i0, j0, k0):
        for t in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for idx in range(4):
                r, c = divmod(idx, 2)
                u_col[(i0+r)*m + (j0+c)] = U2[t][idx]
                v_col[(j0+r)*p + (k0+c)] = V2[t][idx]
                w_col[(k0+r)*n + (i0+c)] = W2[t][idx]
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # Place 2x2x2 Strassen blocks greedily to maximize coverage.
    # i: 0..2 (3 values), j: 0..4 (5 values), k: 0..5 (6 values)
    # A 2x2x2 block at (i0,j0,k0) covers i0..i0+1, j0..j0+1, k0..k0+1.
    # Valid if i0+1 < n, j0+1 < m, k0+1 < p.
    
    blocks = []
    for i0 in range(n - 1):  # 0, 1
        for j0 in range(m - 1):  # 0, 1, 2, 3
            for k0 in range(p - 1):  # 0, 1, 2, 3, 4
                blocks.append((i0, j0, k0))
    
    # Greedy: pick blocks that cover the most uncovered cells.
    covered = set()
    total_cells = n * m * p
    selected = []
    for _ in range(total_cells):
        best_block = None
        best_gain = 0
        for b in blocks:
            i0, j0, k0 = b
            gain = 0
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        cell = (i0+di, j0+dj, k0+dk)
                        if cell not in covered:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        if best_gain == 0:
            break
        selected.append(best_block)
        i0, j0, k0 = best_block
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
        blocks.remove(best_block)
    
    for b in selected:
        add_strassen(*b)
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                u_col = [0] * (n * m)
                v_col = [0] * (m * p)
                w_col = [0] * (p * n)
                u_col[i*m + j] = 1
                v_col[j*p + k] = 1
                w_col[k*n + i] = 1
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
    
    rank = len(U_list)
    U = [[U_list[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_list[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_list[c][r] for c in range(rank)] for r in range(p*n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
