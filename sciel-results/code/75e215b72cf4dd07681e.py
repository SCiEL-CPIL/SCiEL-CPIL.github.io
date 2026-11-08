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
        # Returns a list of (color, vertex) sorted by color, then degree
        # The max color is the bound
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        if not candidates:
            return [], 0
        
        # Order by degree in the subgraph
        degs = []
        for v in candidates:
            d = bin(candidates_mask & compat_mask[v]).count('1')
            degs.append((d, v))
        degs.sort(reverse=True)
        
        colors = {}
        color_order = []
        uncolored = set(candidates)
        max_color = 0
        
        # Simple greedy coloring
        # We assign colors 1, 2, ...
        # Vertices with the same color must not be adjacent in the compatibility graph?
        # No, in Max Clique, we want to color the *induced subgraph* such that
        # adjacent vertices have different colors?
        # Actually, the standard Tomita coloring colors the *complement* of the candidate set?
        # No, it colors the candidate set itself.
        # The number of colors is an upper bound on the clique size.
        # If we can color the candidates with k colors, the max clique size is at most k.
        # Wait, that's for the *independent set* in the original graph?
        # We are finding Max Clique in the Compat Graph.
        # So we color the Compat Graph.
        # Adjacent in Compat Graph must have different colors.
        
        # Greedy coloring of the Compat Graph induced by candidates
        color_map = {}
        current_color = 1
        while uncolored:
            # Pick a vertex with highest degree in uncolored
            v = max(uncolored, key=lambda x: bin((candidates_mask & compat_mask[x]) & 0xFFFFFFFF).count('1')) # Approx
            # Actually, just pick the first one in degs order that is uncolored
            v = None
            for d, u in degs:
                if u in uncolored:
                    v = u
                    break
            
            if v is None: break
            
            color_map[v] = current_color
            uncolored.remove(v)
            
            # Remove neighbors of v from uncolored? No, they just can't have the same color.
            # In greedy coloring, we assign the smallest possible color.
            # But for the bound, we just need *a* coloring.
            # A simple sequential coloring:
            # Assign color 1 to v.
            # Then for remaining, if adjacent to v, they can't be color 1.
            
            # Let's do a proper greedy coloring
            pass # The above is too complex for inline. Let's use a simpler bound if needed.
            
        # Given N=36, the weak bound might be enough if the initial solution is good.
        # Let's just use the weak bound but with a better initial solution.
        
        return degs, len(candidates)

    # Improved Backtracking with better initial solution
    # Try to find a good initial solution using greedy
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

    init_clique = greedy_clique()
    best_clique_size = len(init_clique)
    best_clique_nodes = init_clique[:]

    # Now search for better
    def expand(clique, candidates_mask):
        nonlocal best_clique_size, best_clique_nodes
        
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        if not candidates:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique_nodes = clique[:]
            return
        
        if len(clique) + len(candidates) <= best_clique_size:
            return
        
        # Order by degree
        degs = []
        for v in candidates:
            d = bin(candidates_mask & compat_mask[v]).count('1')
            degs.append((d, v))
        degs.sort(reverse=True)
        
        for d, v in degs:
            if not ((candidates_mask >> v) & 1):
                continue
                
            new_candidates_mask = candidates_mask & compat_mask[v] & ~(1 << v)
            
            if len(clique) + 1 + bin(new_candidates_mask).count('1') <= best_clique_size:
                continue
                
            expand(clique + [v], new_candidates_mask)
            
            # Remove v from candidates for the next iteration?
            # In Bron-Kerbosch, we remove v from P.
            # Here we iterate over a static list `degs`.
            # We must ensure we don't process v again if it was "removed" by a previous branch?
            # No, the standard algorithm iterates P.
            # My loop iterates `degs`.
            # I should update `candidates_mask` to remove v?
            # No, `candidates_mask` is passed by value (int).
            # The recursive call gets a subset.
            # The current loop continues with the *original* `candidates_mask`?
            # No, I should remove v from `candidates_mask` before the next iteration of the loop?
            # If I don't, I might pick a vertex that is incompatible with v?
            # No, the next vertex `v2` is picked from `degs`.
            # If `v2` is incompatible with `v`, it won't be in `new_candidates_mask` of the *previous* branch.
            # But in the *current* branch (after returning from recursion), we are still in the same `expand` call.
            # We need to remove `v` from the set of candidates for the *remaining* iterations of this loop.
            # So:
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
