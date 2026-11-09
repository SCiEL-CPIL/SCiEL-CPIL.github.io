# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 138.0

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
    
    # Precompute tree info
    tree_data = []
    for idx in range(N):
        edges = trees[idx]
        k = idx + 1
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].append(b)
            adj[b].append(a)
        tree_data.append((k, edges, deg, adj))
    
    used_edges = set()
    used_neighbors = [set() for _ in range(N)]
    result = [None] * N
    
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
    
    def reset_state():
        used_edges.clear()
        for i in range(N):
            used_neighbors[i].clear()
        for i in range(N):
            result[i] = None
    
    def embed_tree_fast(k_idx, rng, max_cands=25):
        k = k_idx + 1
        edges, deg, adj = tree_data[k_idx][1], tree_data[k_idx][2], tree_data[k_idx][3]
        
        if k == 1:
            for h in range(N):
                result[k_idx] = [h]
                return True
            return False
        
        if k == 2:
            for u in range(N):
                for v in range(u + 1, N):
                    if edge_free(u, v):
                        mark_edge(u, v)
                        result[k_idx] = [u, v]
                        return True
            return False
        
        # Order vertices by degree descending
        v_order = sorted(range(k), key=lambda v: -deg[v])
        # Break ties randomly for diversity
        for i in range(len(v_order) - 1, 0, -1):
            if deg[v_order[i]] == deg[v_order[i-1]]:
                j = rng.randint(0, i)
                v_order[i], v_order[j] = v_order[j], v_order[i]
        
        adj_list = [adj[v] for v in v_order]
        deg_ordered = [deg[v_order[i]] for i in range(k)]
        
        mapping = [None] * k
        used_host_set = set()
        depth = 0
        
        def backtrack(pos):
            nonlocal depth
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == k:
                return True
            
            depth += 1
            d_needed = deg_ordered[pos]
            
            # Collect placed neighbors (those before pos in v_order)
            placed = []
            for nb in adj_list[pos]:
                # Find position of nb in v_order
                # This is a bit slow, let's precompute
                pass
            
            # Precompute: for each pos, which tree vertices are placed before it
            # This needs to be done outside
            
            candidates = []
            for h in range(N):
                if h in used_host_set:
                    continue
                if N - 1 - len(used_neighbors[h]) < d_needed:
                    continue
                ok = True
                for nb_pos in placed_idx[pos]:
                    nh = mapping[nb_pos]
                    if not edge_free(h, nh):
                        ok = False
                        break
                if not ok:
                    continue
                candidates.append(h)
            
            if not candidates:
                depth -= 1
                return False
            
            # Score candidates: prefer those with fewer available degrees
            # and more connections to placed neighbors
            def score(h):
                avail = N - 1 - len(used_neighbors[h])
                conn = sum(1 for nb_pos in placed_idx[pos] if edge_free(h, mapping[nb_pos]))
                return (avail, -conn, rng.random())
            
            candidates.sort(key=score)
            max_c = min(len(candidates), max_cands)
            
            for h in candidates[:max_c]:
                mapping[pos] = h
                used_host_set.add(h)
                
                for nb_pos in placed_idx[pos]:
                    mark_edge(h, mapping[nb_pos])
                
                if backtrack(pos + 1):
                    return True
                
                for nb_pos in placed_idx[pos]:
                    unmark_edge(h, mapping[nb_pos])
                
                used_host_set.discard(h)
                mapping[pos] = None
            
            depth -= 1
            return False
        
        # Precompute placed_idx: for each position, which earlier positions are neighbors
        placed_idx = []
        for pos in range(k):
            idx_set = set()
            for other in range(pos):
                if other in adj_list[pos]:
                    idx_set.add(other)
            placed_idx.append(idx_set)
        
        if backtrack(0):
            # Build result in original vertex order
            res = [0] * k
            for i, v in enumerate(v_order):
                res[v] = mapping[i]
            result[k_idx] = res
            return True
        return False
    
    def embed_tree_safe(k_idx, rng):
        """More robust embedding with retry logic."""
        # Try multiple vertex orderings
        k = k_idx + 1
        edges, deg, adj = tree_data[k_idx][1], tree_data[k_idx][2], tree_data[k_idx][3]
        
        if k == 1:
            for h in range(N):
                result[k_idx] = [h]
                return True
            return False
        
        if k == 2:
            for u in range(N):
                for v in range(u + 1, N):
                    if edge_free(u, v):
                        mark_edge(u, v)
                        result[k_idx] = [u, v]
                        return True
            return False
        
        # Try different orderings based on degree patterns
        for attempt in range(5):
            v_order = list(range(k))
            if attempt % 2 == 0:
                v_order.sort(key=lambda v: -deg[v])
            else:
                # Sort by (degree, randomization)
                v_order.sort(key=lambda v: (-deg[v], rng.random()))
            
            adj_list = [adj[v] for v in v_order]
            deg_ordered = [deg[v_order[i]] for i in range(k)]
            
            mapping = [None] * k
            used_host_set = set()
            
            def backtrack(pos):
                if time.time() - t0 > _DEADLINE:
                    return False
                if pos == k:
                    return True
                
                d_needed = deg_ordered[pos]
                placed_neighbors = [nb for nb in adj_list[pos] if nb not in v_order[pos+1:]]
                placed_pos = [v_order.index(pn) for pn in placed_neighbors]
                
                candidates = []
                for h in range(N):
                    if h in used_host_set:
                        continue
                    if N - 1 - len(used_neighbors[h]) < d_needed:
                        continue
                    ok = True
                    for pp in placed_pos:
                        if not edge_free(h, mapping[pp]):
                            ok = False
                            break
                    if not ok:
                        continue
                    candidates.append(h)
                
                if not candidates:
                    return False
                
                def score(h):
                    avail = N - 1 - len(used_neighbors[h])
                    conn = sum(1 for pp in placed_pos if edge_free(h, mapping[pp]))
                    return (avail, -conn, rng.random())
                
                candidates.sort(key=score)
                max_c = min(len(candidates), 20)
                
                for h in candidates[:max_c]:
                    mapping[pos] = h
                    used_host_set.add(h)
                    
                    for pp in placed_pos:
                        mark_edge(h, mapping[pp])
                    
                    if backtrack(pos + 1):
                        return True
                    
                    for pp in placed_pos:
                        unmark_edge(h, mapping[pp])
                    
                    used_host_set.discard(h)
                    mapping[pos] = None
                
                return False
            
            if backtrack(0):
                res = [0] * k
                for i, v in enumerate(v_order):
                    res[v] = mapping[i]
                result[k_idx] = res
                return True
        
        return False
    
    best_result = None
    best_count = 0
    
    # Phase 1: Greedy with randomized restarts
    for restart in range(150):
        if time.time() - t0 > _DEADLINE - 5.0:
            break
        
        reset_state()
        rng = random.Random(611 + restart * 7919)
        
        count = 0
        # Embed larger trees first
        for k_idx in range(N - 1, -1, -1):
            if time.time() - t0 > _DEADLINE - 1.0:
                break
            if embed_tree_fast(k_idx, rng):
                count += 1
            else:
                break
        
        if count > best_count:
            best_count = count
            best_result = [r[:] if r is not None else None for r in result]
        
        if best_count == N:
            break
    
    # Phase 2: Try to embed remaining trees if any
    if best_result is not None and best_count < N:
        for retry in range(50):
            if time.time() - t0 > _DEADLINE - 2.0:
                break
            
            reset_state()
            rng = random.Random(999 + retry * 104729)
            
            count = 0
            # Embed in order of difficulty (larger first, then by some heuristic)
            order = sorted(range(N), key=lambda i: -(i + 1))
            for k_idx in order:
                if time.time() - t0 > _DEADLINE - 1.0:
                    break
                if embed_tree_safe(k_idx, rng):
                    count += 1
                else:
                    break
            
            if count > best_count:
                best_count = count
                best_result = [r[:] if r is not None else None for r in result]
            
            if best_count == N:
                break
    
    if best_result is not None:
        for i in range(N):
            if best_result[i] is not None:
                best_result[i] = [int(x) for x in best_result[i]]
        return best_result
    
    return [None] * N
# EVOLVE-BLOCK-END
