# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 140.0  # safety margin before 150s kill

def _gen_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    t0 = time.time()
    N = 20
    trees = _gen_trees(N, 611)
    
    # Precompute tree degrees and adjacency
    tree_info = []
    for k in range(1, N + 1):
        edges = trees[k - 1]
        deg = [0] * k
        adj = [set() for _ in range(k)]
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        tree_info.append((k, edges, deg, adj))
    
    # Embed in decreasing order of size (hardest first)
    order = list(range(N - 1, -1, -1))  # indices: 19, 18, ..., 0
    # i.e., tree k=20 first, then k=19, ..., k=1
    
    # Host state
    used_edges = set()  # frozenset of (u,v) with u<v
    used_neighbors = [set() for _ in range(N)]  # used_neighbors[v] = set of neighbors already used with v
    
    result = [None] * N
    
    def avail_degree(v):
        return N - 1 - len(used_neighbors[v])
    
    def edge_free(u, v):
        if u > v:
            u, v = v, u
        return (u, v) not in used_edges
    
    def mark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.add((u, v))
        used_neighbors[u].add(v)
        used_neighbors[v].add(u)
    
    def unmark_edge(u, v):
        if u > v:
            u, v = v, u
        used_edges.discard((u, v))
        used_neighbors[u].discard(v)
        used_neighbors[v].discard(u)
    
    def embed_tree(k_idx, rng):
        """Try to embed tree k_idx (0-indexed, tree size k_idx+1). Returns mapping or None."""
        k = k_idx + 1
        edges, deg, adj = tree_info[k_idx][1], tree_info[k_idx][2], tree_info[k_idx][3]
        
        if k == 1:
            # Single vertex, pick any host vertex
            for h in range(N):
                result[k_idx] = [h]
                return True
            return False
        
        if k == 2:
            # Single edge
            for u in range(N):
                for v in range(u + 1, N):
                    if edge_free(u, v):
                        mark_edge(u, v)
                        result[k_idx] = [u, v]
                        return True
            return False
        
        # For k >= 3: use backtracking
        # Order vertices by descending degree for placement, with tie-breaking by randomization
        v_order = sorted(range(k), key=lambda v: (-deg[v], rng.random()))
        
        # We'll assign host vertices in v_order sequence
        # mapping: tree_vertex -> host_vertex
        mapping = [None] * k
        assigned = [False] * k
        
        # Precompute for each tree vertex, which other tree vertices are its neighbors
        # and which are already placed (to check edge availability)
        
        def backtrack(pos):
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == k:
                return True
            
            tv = v_order[pos]
            d_needed = deg[tv]
            
            # Collect already-placed neighbors of tv in tree
            placed_neighbors = []
            for nb in adj[tv]:
                if assigned[nb]:
                    placed_neighbors.append(nb)
            
            # Candidate host vertices: unused, avail_degree >= d_needed,
            # and edge_free with all placed neighbor hosts
            candidates = []
            for h in range(N):
                if any(mapping[x] == h for x in range(k) if assigned[x]):
                    continue
                if avail_degree(h) < d_needed:
                    continue
                ok = True
                for nb in placed_neighbors:
                    if not edge_free(h, mapping[nb]):
                        ok = False
                        break
                if ok:
                    candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort candidates by ascending avail_degree (most constrained first), with random tie-breaking
            candidates.sort(key=lambda h: (avail_degree(h), rng.random()))
            
            for h in candidates:
                # Assign
                mapping[tv] = h
                assigned[tv] = True
                
                # Mark edges to placed neighbors
                for nb in placed_neighbors:
                    mark_edge(h, mapping[nb])
                
                if backtrack(pos + 1):
                    return True
                
                # Unmark
                for nb in placed_neighbors:
                    unmark_edge(h, mapping[nb])
                
                mapping[tv] = None
                assigned[tv] = False
            
            return False
        
        if backtrack(0):
            # Build result: mapping[i] is host for tree vertex i
            result[k_idx] = [int(mapping[i]) for i in range(k)]
            return True
        return False
    
    # Main loop: embed trees in order with randomized restarts
    max_restarts = 50
    best_result = None
    best_count = 0
    
    for restart in range(max_restarts):
        if time.time() - t0 > _DEADLINE:
            break
        
        # Reset state for each restart
        used_edges.clear()
        for i in range(N):
            used_neighbors[i].clear()
        result = [None] * N
        
        rng = random.Random(611 + restart)
        
        count = 0
        for k_idx in order:
            if time.time() - t0 > _DEADLINE:
                break
            if embed_tree(k_idx, rng):
                count += 1
            else:
                break
        
        if count > best_count:
            best_count = count
            best_result = [r[:] if r is not None else None for r in result]
        
        # Early termination if we've embedded all trees
        if best_count == N:
            break
    
    # Use the best result found
    if best_result is not None:
        for i in range(N):
            if best_result[i] is not None:
                best_result[i] = [int(x) for x in best_result[i]]
        return best_result
    
    # Fallback: return empty result if nothing was found
    return [None] * N
# EVOLVE-BLOCK-END
