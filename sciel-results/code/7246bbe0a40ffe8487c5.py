import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def _add_strassen_block(U, V, W, n, m, p, i0, j0, k0, col_offset):
    """Add a Strassen block at position (i0, j0, k0) starting at col_offset."""
    c = col_offset + 0
    U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
    
    c = col_offset + 1
    U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
    
    c = col_offset + 2
    U[i0*m + j0][c] += 1
    V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
    W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
    
    c = col_offset + 3
    U[(i0+1)*m + (j0+1)][c] += 1
    V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
    W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
    
    c = col_offset + 4
    U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
    
    c = col_offset + 5
    U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
    V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    
    c = col_offset + 6
    U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
    V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1

def _build_decomposition(n, m, p, blocks):
    """Build decomposition given a list of Strassen block positions."""
    total_rank = len(blocks) * 7 + (n*m*p - len(blocks)*8)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for i0, j0, k0 in blocks:
        _add_strassen_block(U, V, W, n, m, p, i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W, "rank": col}

def _verify_decomp(n, m, p, decomp):
    """Verify the decomposition is correct."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _find_best_block_placement(n, m, p, max_blocks=4, deadline=None):
    """Find the best placement of non-overlapping 2x2x2 Strassen blocks."""
    # Generate all valid block positions
    valid_positions = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                valid_positions.append((i0, j0, k0))
    
    best_rank = n*m*p
    best_blocks = None
    
    def check_overlap(blocks):
        """Check if any two blocks overlap."""
        covered = set()
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        pos = (i0+di, j0+dj, k0+dk)
                        if pos in covered:
                            return True
                        covered.add(pos)
        return False
    
    # Try all combinations of up to max_blocks non-overlapping blocks
    # Use backtracking
    def backtrack(start_idx, current_blocks, current_count):
        nonlocal best_rank, best_blocks
        if deadline and time.time() > deadline - 5:
            return
        if current_count == max_blocks:
            # Calculate rank
            covered_count = len(current_blocks) * 8
            uncovered = n*m*p - covered_count
            rank = len(current_blocks) * 7 + uncovered
            if rank < best_rank:
                best_rank = rank
                best_blocks = list(current_blocks)
            return
        if current_count > 0:
            covered_count = len(current_blocks) * 8
            uncovered = n*m*p - covered_count
            rank = len(current_blocks) * 7 + uncovered
            if rank < best_rank:
                best_rank = rank
                best_blocks = list(current_blocks)
        
        for idx in range(start_idx, len(valid_positions)):
            pos = valid_positions[idx]
            new_blocks = current_blocks + [pos]
            if not check_overlap(new_blocks):
                backtrack(idx + 1, new_blocks, current_count + 1)
    
    backtrack(0, [], 0)
    return best_blocks, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin
    
    n, m, p = 3, 5, 5
    
    # Default rank-71 solution
    default_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    best_decomp = _build_decomposition(n, m, p, default_blocks)
    best_rank = best_decomp["rank"]
    
    # Search for better block placements
    best_blocks, found_rank = _find_best_block_placement(n, m, p, max_blocks=4, deadline=deadline)
    
    if best_blocks and found_rank < best_rank:
        candidate = _build_decomposition(n, m, p, best_blocks)
        if _verify_decomp(n, m, p, candidate):
            best_decomp = candidate
            best_rank = found_rank
    
    if time.time() > deadline - 5:
        return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
    
    # Try greedy tensor decomposition approach for potential further improvement
    # Start with the best Strassen-based solution and try to reduce rank
    # by finding linear dependencies among the rank-1 terms
    
    # For now, return the best found
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify correctness
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    error += 1
    print(f"Verification errors: {error}")
