import time
import itertools
import numpy as np

_DIMS = (3, 4, 7)

def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
                
    num_blocks = len(blocks)
    
    # Build conflict graph
    adj = [[False] * num_blocks for _ in range(num_blocks)]
    
    def are_incompatible(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
            return True
        return False

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if are_incompatible(blocks[i], blocks[j]):
                adj[i][j] = True
                adj[j][i] = True

    # Compatibility graph
    compat_mask = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for j in range(num_blocks):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        compat_mask[i] = mask

    best_clique_size = 0
    best_clique_nodes = []

    # Tomita's algorithm for Maximum Clique
    # Uses greedy coloring to bound the search
    
    def color_sort(candidates_mask):
        # Returns (ordered_vertices, max_color) where ordered_vertices is sorted by color
        # and max_color is an upper bound on clique size
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        if not candidates:
            return [], 0
        
        # Order by degree in the subgraph
        degs = []
        for v in candidates:
            d = bin(candidates_mask & compat_mask[v]).count('1')
            degs.append((d, v))
        degs.sort(reverse=True)
        
        # Proper greedy coloring: assign smallest available color
        color_map = {}
        max_color = 0
        uncolored = list(candidates)
        
        while uncolored:
            # Pick vertex with highest degree among uncolored
            v = None
            best_deg = -1
            for d, u in degs:
                if u in uncolored:
                    if d > best_deg:
                        best_deg = d
                        v = u
            if v is None:
                break
            
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for u in uncolored:
                if u != v and (compat_mask[v] >> u) & 1:
                    if u in color_map:
                        neighbor_colors.add(color_map[u])
            
            c = 1
            while c in neighbor_colors:
                c += 1
            color_map[v] = c
            max_color = max(max_color, c)
            uncolored.remove(v)
        
        # Sort by color for ordering
        ordered = sorted(candidates, key=lambda x: (color_map[x], -bin(candidates_mask & compat_mask[x]).count('1')))
        return ordered, max_color

    # Improved Backtracking with better initial solution
    # Try to find a good initial solution using greedy with multiple strategies
    def greedy_clique():
        mask = (1 << num_blocks) - 1
        clique = []
        while mask:
            # Pick vertex with max degree in mask
            best_v = -1
            best_deg = -1
            for v in range(num_blocks):
                if (mask >> v) & 1:
                    d = bin(mask & compat_mask[v]).count('1')
                    if d > best_deg:
                        best_deg = d
                        best_v = v
            if best_v == -1: break
            clique.append(best_v)
            mask &= compat_mask[best_v]
            mask &= ~(1 << best_v)
        return clique
    
    import random
    random.seed(42)
    
    # Try multiple greedy strategies
    best_init = greedy_clique()
    
    # Randomized greedy: at each step, pick from top-k candidates
    for trial in range(50):
        mask = (1 << num_blocks) - 1
        clique = []
        while mask:
            candidates = [v for v in range(num_blocks) if (mask >> v) & 1]
            if not candidates:
                break
            # Get degrees
            degs = [(bin(mask & compat_mask[v]).count('1'), v) for v in candidates]
            degs.sort(reverse=True)
            # Pick from top 3 with randomization
            k = min(3, len(degs))
            v = degs[random.randint(0, k-1)][1]
            clique.append(v)
            mask &= compat_mask[v]
            mask &= ~(1 << v)
        if len(clique) > len(best_init):
            best_init = clique
    
    best_clique_size = len(best_init)
    best_clique_nodes = best_init[:]

    # Now search for better with Tomita-style coloring bound
    def expand(clique, candidates_mask):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > _t0 + 115.0:
            return
        
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        if not candidates:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique_nodes = clique[:]
            return
        
        # Use coloring bound
        ordered, max_color = color_sort(candidates_mask)
        
        if len(clique) + max_color <= best_clique_size:
            return
        
        for v in ordered:
            if not ((candidates_mask >> v) & 1):
                continue
                
            new_candidates_mask = candidates_mask & compat_mask[v] & ~(1 << v)
            
            if len(clique) + 1 + bin(new_candidates_mask).count('1') <= best_clique_size:
                continue
                
            expand(clique + [v], new_candidates_mask)
            
            # Remove v from candidates for remaining iterations
            candidates_mask &= ~(1 << v)

    initial_mask = (1 << num_blocks) - 1
    expand([], initial_mask)

    best_block_indices = best_clique_nodes
    best_blocks = [blocks[i] for i in best_block_indices]
    
    num_selected = len(best_blocks)
    
    covered_entries = set()
    for b in best_blocks:
        i0, j0, k0 = b
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered_entries.add((i, j, k))
    
    uncovered_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_entries:
                    uncovered_entries.append((i, j, k))
    
    rank = 7 * num_selected + len(uncovered_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b in best_blocks:
        add_strassen(*b)
        
    for (i, j, k) in uncovered_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    if time.time() < _t0 + 110.0:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if val != 1:
                        raise ValueError(f"Decomposition failed at ({i},{j},{k}): {val}")

    return {"U": U, "V": V, "W": W}
