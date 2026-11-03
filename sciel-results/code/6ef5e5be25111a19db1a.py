import time
import numpy as np
import itertools
from collections import defaultdict

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = U.shape[1]
        for a in range(N * M):
            for b in range(M * P):
                acc = np.zeros(P * N)
                for c in range(rank):
                    u = U[a, c]
                    if u == 0:
                        continue
                    v = V[b, c]
                    if v == 0:
                        continue
                    acc += u * v * W[:, c]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                expected_idx = k2 * N + i
                if abs(acc[expected_idx] - 1.0) > tol:
                    return False
                mask = np.ones(P * N, dtype=bool)
                mask[expected_idx] = False
                if np.any(np.abs(acc[mask]) > tol):
                    return False
        return True
    except Exception:
        return False

def _build_schoolbook():
    """Build the 72-rank schoolbook decomposition."""
    U = np.zeros((N * M, SCHOOLBOOK), dtype=np.float64)
    V = np.zeros((M * P, SCHOOLBOOK), dtype=np.float64)
    W = np.zeros((P * N, SCHOOLBOOK), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def _generate_strassen_blocks():
    """
    Generate all possible 2x2x2 Strassen blocks.
    A block is defined by (i0, j0, k0) where i0 in {0,1}, j0 in {0,1}, k0 in {0..6}.
    Each block covers A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2].
    It saves 1 rank compared to schoolbook (7 terms instead of 8).
    """
    blocks = []
    for i0 in range(N - 1):
        for j0 in range(M - 1):
            for k0 in range(P - 1):
                blocks.append((i0, j0, k0))
    return blocks

def _get_block_cells(block):
    """Get the set of (i, j, k) indices covered by a Strassen block."""
    i0, j0, k0 = block
    cells = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                cells.add((i0 + di, j0 + dj, k0 + dk))
    return cells

def _build_decomposition_with_blocks(selected_blocks):
    """
    Build a decomposition using the selected non-overlapping Strassen blocks
    and schoolbook for the rest.
    Returns (U, V, W) if valid, else None.
    """
    # Identify all cells covered by Strassen blocks
    covered_cells = set()
    for block in selected_blocks:
        cells = _get_block_cells(block)
        if covered_cells.intersection(cells):
            return None  # Overlap detected, shouldn't happen if pre-filtered
        covered_cells.update(cells)
        
    # Initialize arrays with max possible rank (72)
    U = np.zeros((N * M, SCHOOLBOOK), dtype=np.float64)
    V = np.zeros((M * P, SCHOOLBOOK), dtype=np.float64)
    W = np.zeros((P * N, SCHOOLBOOK), dtype=np.float64)
    
    col = 0
    used_cells = set()
    
    # First, add all Strassen blocks
    for block in selected_blocks:
        i0, j0, k0 = block
        # Indices for A
        a00 = i0 * M + j0
        a01 = i0 * M + (j0 + 1)
        a10 = (i0 + 1) * M + j0
        a11 = (i0 + 1) * M + (j0 + 1)
        
        # Indices for B
        b00 = j0 * P + k0
        b01 = j0 * P + (k0 + 1)
        b10 = (j0 + 1) * P + k0
        b11 = (j0 + 1) * P + (k0 + 1)
        
        # Indices for C (W layout: W[k*N + i, r])
        c00 = k0 * N + i0
        c01 = k0 * N + (i0 + 1)
        c10 = (k0 + 1) * N + i0
        c11 = (k0 + 1) * N + (i0 + 1)
        
        # Strassen 7 terms
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        W[c01, col] -= 1; W[c10, col] -= 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c10, col] -= 1
        col += 1
        
        # Mark cells as used
        for cell in _get_block_cells(block):
            used_cells.add(cell)

    # Second, add schoolbook terms for remaining cells
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) in used_cells:
                    continue
                U[i * M + j, col] += 1
                V[j * P + k, col] += 1
                W[k * N + i, col] += 1
                col += 1

    # Trim to actual rank
    return U[:, :col], V[:, :col], W[:, :col]

def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin
    
    # 1. Generate all possible Strassen blocks
    all_blocks = _generate_strassen_blocks()
    
    # 2. Pre-compute cell sets for each block
    block_cells = {b: _get_block_cells(b) for b in all_blocks}
    
    # 3. Find all maximal sets of non-overlapping blocks
    # We want to maximize the number of Strassen blocks (each saves 1 rank).
    # Total rank = 72 - (# Strassen blocks).
    # We search for the maximum independent set in the conflict graph of blocks.
    
    # Build conflict graph
    n_blocks = len(all_blocks)
    conflicts = [set() for _ in range(n_blocks)]
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            if block_cells[all_blocks[i]].intersection(block_cells[all_blocks[j]]):
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Backtracking search for maximum independent set
    # Since the graph is small (18 nodes), this is fast.
    best_independent_set = []
    
    def backtrack(available, current_set):
        nonlocal best_independent_set
        if time.time() > deadline:
            return
        if len(current_set) > len(best_independent_set):
            best_independent_set = current_set.copy()
        
        if not available:
            return
        
        # Heuristic: pick a node with minimum degree in the induced subgraph?
        # Or just pick the first one.
        # To ensure we find the *maximum*, we need to explore branches.
        # Branch 1: Include the first available node
        # Branch 2: Exclude the first available node
        
        node = list(available)[0]
        available_list = list(available)
        
        # Option 1: Include node
        new_available = set(available)
        new_available.remove(node)
        # Remove neighbors of node
        for neighbor in conflicts[node]:
            if neighbor in new_available:
                new_available.remove(neighbor)
        
        current_set.append(node)
        backtrack(new_available, current_set)
        current_set.pop()
        
        # Option 2: Exclude node
        new_available = set(available)
        new_available.remove(node)
        backtrack(new_available, current_set)
        
    backtrack(set(range(n_blocks)), [])
    
    # 4. Build the decomposition using the best independent set found
    if best_independent_set:
        selected_blocks = [all_blocks[i] for i in best_independent_set]
        U, V, W = _build_decomposition_with_blocks(selected_blocks)
        if U is not None and _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 5. Fallback: Schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
