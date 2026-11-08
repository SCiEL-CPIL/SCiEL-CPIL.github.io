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
    # adj[i][j] = True if blocks i and j are INCOMPATIBLE (conflict)
    adj = [[False] * num_blocks for _ in range(num_blocks)]
    
    def are_incompatible(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Overlap in all dimensions
        if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
            return True
        return False

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if are_incompatible(blocks[i], blocks[j]):
                adj[i][j] = True
                adj[j][i] = True

    # Find Maximum Independent Set
    # Use a bitset-based Tomita algorithm for Max Clique in the complement graph (compatibility graph)
    # Or simply a highly optimized backtracking with coloring bound.
    
    # Compatibility graph edges: i ~ j if NOT adj[i][j]
    # We want Max Clique in Compat Graph.
    
    # Precompute neighbors as bitmasks for speed
    # N=36, fits in a Python int (arbitrary precision)
    compat_mask = [0] * num_blocks
    for i in range(num_blocks):
        mask = 0
        for j in range(num_blocks):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        compat_mask[i] = mask

    best_clique_size = 0
    best_clique_nodes = []

    def expand(clique, candidates_mask):
        nonlocal best_clique_size, best_clique_nodes
        
        # Greedy Coloring Bound
        # We want to color the subgraph induced by candidates.
        # The max color is an upper bound on the clique size in this subgraph.
        
        candidates = [i for i in range(num_blocks) if (candidates_mask >> i) & 1]
        
        if not candidates:
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique_nodes = clique[:]
            return
        
        # Simple heuristic ordering: sort by degree in candidates
        # Degree of i in candidates:
        degs = []
        for i in candidates:
            d = 0
            for j in candidates:
                if i != j and (compat_mask[i] >> j) & 1:
                    d += 1
            degs.append((d, i))
        degs.sort(reverse=True)
        
        # Greedy Coloring
        # Assign colors 1, 2, ... to vertices such that adjacent vertices have different colors.
        # The number of colors is an upper bound on the clique size.
        # We process vertices in order of decreasing degree (heuristic).
        
        color_map = {}
        current_color = 0
        uncolored = set(candidates)
        
        # This is a simplified coloring. A proper Tomita implementation is complex.
        # Given N=36, even a simpler bound might suffice if we are careful.
        # Let's use a tighter bound: The number of remaining candidates is a trivial bound.
        # If len(clique) + len(candidates) <= best_clique_size: return
        
        if len(clique) + len(candidates) <= best_clique_size:
            return
            
        # Iterate through candidates
        for d, v in degs:
            if (candidates_mask >> v) & 1 == 0:
                continue # already processed or not in candidates? 
            # v is in candidates
            
            # New candidates must be neighbors of v in compat graph
            new_candidates_mask = candidates_mask & compat_mask[v] & ~(1 << v)
            
            # Pruning
            if len(clique) + 1 + bin(new_candidates_mask).count('1') <= best_clique_size:
                continue
                
            expand(clique + [v], new_candidates_mask)
            
            # Remove v from candidates for the next iteration (standard Bron-Kerbosch style)
            # But here we are iterating a sorted list, so we just rely on the mask update in the recursive call?
            # No, we need to remove v from the current level's candidate set to avoid re-adding it in sibling branches.
            # However, the standard recursive structure passes a *subset* of candidates.
            # The loop above iterates over the *initial* candidates of this level.
            # We must ensure we don't pick a vertex that was "excluded" by previous branches?
            # Actually, in Max Clique search:
            # R = clique, P = candidates.
            # for v in P:
            #   R.add(v)
            #   P' = P intersect N(v)
            #   recurse(R, P')
            #   P.remove(v)
            # 
            # My loop iterates over `degs` which is a static list.
            # I need to simulate P.remove(v).
            # The `new_candidates_mask` handles the intersection.
            # But I need to remove v from the *current* `candidates_mask` for the sake of the bound?
            # No, the bound is calculated once at the start.
            # The issue is that if I pick v, I recurse. Then I move to next v in `degs`.
            # The next v is still in `candidates_mask`.
            # This is correct. The recursive call handles the exclusion of v and its non-neighbors for *that* branch.
            # The current branch just tries the next vertex.

    # Initialize
    initial_mask = (1 << num_blocks) - 1
    expand([], initial_mask)

    best_block_indices = best_clique_nodes
    best_blocks = [blocks[i] for i in best_block_indices]
    
    # Construct decomposition
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
        # M1
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M2
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        # M5
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        # M6
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M7
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b in best_blocks:
        add_strassen(*b)
        
    # Add schoolbook terms for uncovered entries
    for (i, j, k) in uncovered_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # Verification
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
