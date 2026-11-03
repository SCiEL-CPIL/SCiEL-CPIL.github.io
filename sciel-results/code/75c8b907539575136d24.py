"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    total_elements = n * m * p  # 75
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 3x3x3 Laderman decomposition starting at (i0, j0, k0)."""
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + (j0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[(j0+2)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        V[j0*p + k0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += -1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[k0*n + (i0+1)][c] += 1
    
    def build_decomposition(blocks):
        """Build decomposition from block placements + schoolbook for rest."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        total_rank = 0
        
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 7
            elif btype == 'larderman':
                for di in range(3):
                    for dj in range(3):
                        for dk in range(3):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 7
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        
        total_rank += remaining
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                add_strassen_block(U, V, W, i0, j0, k0, col)
                col += 7
            elif btype == 'larderman':
                add_laderman_block(U, V, W, i0, j0, k0, col)
                col += 7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def is_valid_placement(blocks):
        """Check if all blocks are within bounds and non-overlapping."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                size = 2
            elif btype == 'larderman':
                size = 3
            else:
                return False
            
            if i0 < 0 or i0+size > n or j0 < 0 or j0+size > m or k0 < 0 or k0+size > p:
                return False
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        if covered[i0+di][j0+dj][k0+dk]:
                            return False
                        covered[i0+di][j0+dj][k0+dk] = True
        return True
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    
    # Generate all possible block positions
    strassen_positions = [('strassen', i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    laderman_positions = [('larderman', i0, j0, k0) for i0 in range(n-2) for j0 in range(m-2) for k0 in range(p-2)]
    
    best_rank = n * m * p
    best_result = None
    
    # First, try the known good configuration: 4 Strassen blocks
    base_config = [
        ('strassen', 0, 0, 0),
        ('strassen', 0, 2, 0),
        ('strassen', 0, 0, 2),
        ('strassen', 0, 2, 2),
    ]
    U, V, W, rank = build_decomposition(base_config)
    if verify(U, V, W):
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Try configurations with 1 Laderman block
    if time.time() < deadline - 20:
        for lpos in laderman_positions:
            if time.time() > deadline - 10:
                break
            # Try adding Strassen blocks around the Laderman block
            for s_combo in combinations(strassen_positions, 1):
                if time.time() > deadline - 10:
                    break
                config = [lpos] + list(s_combo)
                if not is_valid_placement(config):
                    continue
                rank_est = 2 * 7 + (n*m*p - 27 - 8)
                if rank_est >= best_rank:
                    continue
                U, V, W, rank = build_decomposition(config)
                if verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # Try 2 Laderman blocks
    if time.time() < deadline - 20:
        for l_combo in combinations(laderman_positions, 2):
            if time.time() > deadline - 10:
                break
            if not is_valid_placement(list(l_combo)):
                continue
            rank_est = 2 * 7 + (n*m*p - 2*27)
            if rank_est >= best_rank:
                continue
            U, V, W, rank = build_decomposition(list(l_combo))
            if verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    # Try various combinations of Strassen blocks (up to 5)
    if time.time() < deadline - 20:
        for size in [3, 4, 5]:
            if time.time() > deadline - 10:
                break
            for combo in combinations(strassen_positions, size):
                if time.time() > deadline - 10:
                    break
                if not is_valid_placement(list(combo)):
                    continue
                rank_est = size * 7 + (n*m*p - size*8)
                if rank_est >= best_rank:
                    continue
                U, V, W, rank = build_decomposition(list(combo))
                if verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # Try mixed: 1 Laderman + multiple Strassen
    if time.time() < deadline - 20:
        for lpos in laderman_positions:
            if time.time() > deadline - 10:
                break
            for s_size in [2, 3]:
                if time.time() > deadline - 10:
                    break
                for s_combo in combinations(strassen_positions, s_size):
                    if time.time() > deadline - 10:
                        break
                    config = [lpos] + list(s_combo)
                    if not is_valid_placement(config):
                        continue
                    rank_est = (1 + s_size) * 7 + (n*m*p - 27 - s_size*8)
                    if rank_est >= best_rank:
                        continue
                    U, V, W, rank = build_decomposition(config)
                    if verify(U, V, W):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
    
    # Try more aggressive combinations: 1 Laderman + 4 Strassen
    if time.time() < deadline - 20:
        for lpos in laderman_positions:
            if time.time() > deadline - 10:
                break
            s_size = 4
            for s_combo in combinations(strassen_positions, s_size):
                if time.time() > deadline - 10:
                    break
                config = [lpos] + list(s_combo)
                if not is_valid_placement(config):
                    continue
                rank_est = (1 + s_size) * 7 + (n*m*p - 27 - s_size*8)
                if rank_est >= best_rank:
                    continue
                U, V, W, rank = build_decomposition(config)
                if verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # Try 2 Laderman + Strassen combinations
    if time.time() < deadline - 20:
        for l_combo in combinations(laderman_positions, 2):
            if time.time() > deadline - 10:
                break
            if not is_valid_placement(list(l_combo)):
                continue
            for s_size in [1, 2]:
                if time.time() > deadline - 10:
                    break
                for s_combo in combinations(strassen_positions, s_size):
                    if time.time() > deadline - 10:
                        break
                    config = list(l_combo) + list(s_combo)
                    if not is_valid_placement(config):
                        continue
                    rank_est = (2 + s_size) * 7 + (n*m*p - 2*27 - s_size*8)
                    if rank_est >= best_rank:
                        continue
                    U, V, W, rank = build_decomposition(config)
                    if verify(U, V, W):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
    
    # NEW: Try a more exhaustive search with better pruning
    # Use a bitmask approach for faster overlap checking
    if time.time() < deadline - 15:
        # Precompute bitmasks for each block
        def block_to_mask(btype, i0, j0, k0):
            if btype == 'strassen':
                size = 2
            else:
                size = 3
            mask = 0
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        bit = (i0+di) * m * p + (j0+dj) * p + (k0+dk)
                        mask |= (1 << bit)
            return mask
        
        strassen_masks = [(pos, block_to_mask('strassen', pos[1], pos[2], pos[3])) for pos in strassen_positions]
        laderman_masks = [(pos, block_to_mask('larderman', pos[1], pos[2], pos[3])) for pos in laderman_positions]
        
        # Try all subsets of Strassen blocks up to size 6
        for s_size in range(3, 7):
            if time.time() > deadline - 10:
                break
            for combo in combinations(strassen_masks, s_size):
                if time.time() > deadline - 10:
                    break
                # Check non-overlap using bitmasks
                combined_mask = 0
                valid = True
                for _, mask in combo:
                    if combined_mask & mask:
                        valid = False
                        break
                    combined_mask |= mask
                if not valid:
                    continue
                rank_est = s_size * 7 + (n*m*p - s_size*8)
                if rank_est >= best_rank:
                    continue
                blocks = [pos for pos, _ in combo]
                U, V, W, rank = build_decomposition(blocks)
                if verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
        
        # Try 1 Laderman + up to 5 Strassen
        for lpos, lmask in laderman_masks:
            if time.time() > deadline - 10:
                break
            for s_size in range(2, 6):
                if time.time() > deadline - 10:
                    break
                for s_combo in combinations(strassen_masks, s_size):
                    if time.time() > deadline - 10:
                        break
                    combined_mask = lmask
                    valid = True
                    for _, mask in s_combo:
                        if combined_mask & mask:
                            valid = False
                            break
                        combined_mask |= mask
                    if not valid:
                        continue
                    rank_est = (1 + s_size) * 7 + (n*m*p - 27 - s_size*8)
                    if rank_est >= best_rank:
                        continue
                    blocks = [lpos] + [pos for pos, _ in s_combo]
                    U, V, W, rank = build_decomposition(blocks)
                    if verify(U, V, W):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c2] = 1
                    V_s[j*p+k][c2] = 1
                    W_s[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return best_result
# EVOLVE-BLOCK-END
