import time
import numpy as np
from itertools import combinations, product
from collections import defaultdict
from scipy.optimize import lsq_linear

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Build the target tensor
    target = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0

    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        # T_decomp has shape (n*m, m*p, p*n)
        # Target T[i,j,k] = 1 corresponds to T_decomp[i*m+j, j*p+k, k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = T_decomp[i*m + j, j*p + k, k*n + i]
                    if not np.isclose(val, 1.0, atol=1e-10):
                        return False
        return True

    def build_trivial():
        """Trivial rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return U, V, W

    def add_strassen_block(U, V, W, col, i0, j0, k0, scale=1.0):
        """Add 7 Strassen terms to U, V, W starting at column col. Returns new col."""
        s = scale
        # Term 0
        c = col + 0
        U[i0*m + j0][c] += s
        U[(i0+1)*m + (j0+1)][c] += s
        V[j0*p + k0][c] += s
        V[(j0+1)*p + (k0+1)][c] += s
        W[k0*n + i0][c] += s
        W[(k0+1)*n + (i0+1)][c] += s
        # Term 1
        c = col + 1
        U[(i0+1)*m + j0][c] += s
        U[(i0+1)*m + (j0+1)][c] += s
        V[j0*p + k0][c] += s
        W[k0*n + (i0+1)][c] += s
        W[(k0+1)*n + (i0+1)][c] += -s
        # Term 2
        c = col + 2
        U[i0*m + j0][c] += s
        V[j0*p + (k0+1)][c] += s
        V[(j0+1)*p + (k0+1)][c] += -s
        W[(k0+1)*n + i0][c] += s
        W[(k0+1)*n + (i0+1)][c] += s
        # Term 3
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += s
        V[(j0+1)*p + k0][c] += s
        V[j0*p + k0][c] += -s
        W[k0*n + i0][c] += s
        W[k0*n + (i0+1)][c] += s
        # Term 4
        c = col + 4
        U[i0*m + j0][c] += s
        U[i0*m + (j0+1)][c] += s
        V[(j0+1)*p + (k0+1)][c] += s
        W[k0*n + i0][c] += -s
        W[(k0+1)*n + i0][c] += s
        # Term 5
        c = col + 5
        U[(i0+1)*m + j0][c] += s
        U[i0*m + j0][c] += -s
        V[j0*p + k0][c] += s
        V[j0*p + (k0+1)][c] += s
        W[(k0+1)*n + (i0+1)][c] += s
        # Term 6
        c = col + 6
        U[i0*m + (j0+1)][c] += s
        U[(i0+1)*m + (j0+1)][c] += -s
        V[(j0+1)*p + k0][c] += s
        V[(j0+1)*p + (k0+1)][c] += s
        W[k0*n + i0][c] += s
        return col + 7

    def build_from_blocks(blocks, scales):
        """Build decomposition from given Strassen block positions and scales."""
        U = [[0] * (len(blocks) * 7) for _ in range(n * m)]
        V = [[0] * (len(blocks) * 7) for _ in range(m * p)]
        W = [[0] * (len(blocks) * 7) for _ in range(p * n)]
        col = 0
        for idx, (i0, j0, k0) in enumerate(blocks):
            col = add_strassen_block(U, V, W, col, i0, j0, k0, scales[idx])
        return U, V, W

    # Trivial fallback
    U_f, V_f, W_f = build_trivial()
    best_rank = n * m * p
    best_result = (U_f, V_f, W_f)

    # Known good 4-block placement
    default_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    default_scales = [1.0, 1.0, 1.0, 1.0]
    U_d, V_d, W_d = build_from_blocks(default_blocks, default_scales)
    r_d = len(U_d[0])
    if verify(U_d, V_d, W_d) and r_d < best_rank:
        best_rank = r_d
        best_result = (U_d, V_d, W_d)

    # Search for 10 blocks that cover all entries and have a consistent linear system
    valid_positions = []
    for i0 in range(2):
        for j0 in range(4):
            for k0 in range(4):
                if i0 + 1 < n and j0 + 1 < m and k0 + 1 < p:
                    valid_positions.append((i0, j0, k0))
    
    # Precompute coverage mask for each position
    pos_masks = []
    for pos in valid_positions:
        i0, j0, k0 = pos
        mask = np.zeros((n, m, p), dtype=bool)
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ni, nj, nk = i0+di, j0+dj, k0+dk
                    if ni < n and nj < m and nk < p:
                        mask[ni, nj, nk] = True
        pos_masks.append(mask)

    # Greedy set cover to find 10 blocks
    # We want to select 10 blocks that cover all 75 entries.
    # Then check if the system is consistent.
    
    # Heuristic: Start with the 4 good blocks, then add blocks that cover the most uncovered entries.
    current_blocks = [0, 1, 16, 17] # Indices in valid_positions corresponding to default_blocks
    # (0,0,0) is index 0?
    # valid_positions generation:
    # i0=0: j0=0..3, k0=0..3 -> 16 positions
    # i0=1: j0=0..3, k0=0..3 -> 16 positions
    # Total 32? No.
    # i0 in range(2) -> 0, 1.
    # j0 in range(4) -> 0, 1, 2, 3.
    # k0 in range(4) -> 0, 1, 2, 3.
    # Total 2*4*4 = 32 positions.
    
    # Let's map default_blocks to indices
    def get_pos_index(pos):
        return valid_positions.index(pos)
        
    current_indices = [get_pos_index(pos) for pos in default_blocks]
    
    # Greedy expansion
    selected = set(current_indices)
    covered = np.zeros((n, m, p), dtype=bool)
    for idx in selected:
        covered |= pos_masks[idx]
        
    remaining = 75 - np.sum(covered)
    
    # Add blocks greedily
    while len(selected) < 10 and remaining > 0:
        best_idx = -1
        best_gain = 0
        for idx in range(len(valid_positions)):
            if idx in selected:
                continue
            gain = np.sum(pos_masks[idx] & ~covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1 or best_gain == 0:
            break
        selected.add(best_idx)
        covered |= pos_masks[best_idx]
        remaining = 75 - np.sum(covered)
        
    if len(selected) == 10 and remaining == 0:
        # Check linear system
        # M is 75x10
        M = np.zeros((75, 10))
        for col_idx, b_idx in enumerate(selected):
            mask = pos_masks[b_idx]
            # Flatten mask
            M[:, col_idx] = mask.flatten()
            
        # Solve M s = 1
        # Check consistency
        # Use least squares to find solution, then check residual
        sol, residuals, rank, sv = np.linalg.lstsq(M, np.ones(75), rcond=None)
        if np.allclose(M @ sol, np.ones(75), atol=1e-8):
            # System is consistent
            scales = sol
            blocks = [valid_positions[idx] for idx in selected]
            U_t, V_t, W_t = build_from_blocks(blocks, scales)
            r_t = len(U_t[0])
            if r_t < best_rank:
                if verify(U_t, V_t, W_t):
                    best_rank = r_t
                    best_result = (U_t, V_t, W_t)

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
