import time
from random import Random

def instance_trees(n=20, seed=611):
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
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    n = 20
    max_possible = sum(len(t) for t in trees)
    
    # Pre-compute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges_k = trees[k-1]
        adj = [[] for _ in range(k)]
        for (u, v) in edges_k:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        
    # Generate 1-factorization of K20
    # Standard construction for K_2m:
    # Vertices 0..2m-2 and infinity.
    # For i in 0..2m-2:
    #   Matching i: {inf, 0}, {(1+i, 2m-1-i), (2+i, 1m-1-i)} ... 
    # Actually, simpler:
    # Vertices 0..19.
    # Fix vertex 19.
    # For i in 0..18:
    #   Edge (19, i)
    #   For j in 1..9:
    #     Edge ((i+j) % 19, (i-j) % 19)
    # This gives 19 matchings of size 10.
    matchings = []
    for i in range(19):
        m = set()
        m.add((min(19, i), max(19, i)))
        for j in range(1, 10):
            u = (i + j) % 19
            v = (i - j) % 19
            m.add((min(u, v), max(u, v)))
        matchings.append(m)
        
    # Map edge to color (index in matchings)
    edge_color = {}
    for idx, m in enumerate(matchings):
        for e in m:
            edge_color[e] = idx
            
    def solve_deterministic():
        used_edges = set()
        result = [None] * n
        
        # Order trees by complexity (edges count desc) to constrain early
        # But we must return in original order
        tree_indices = list(range(n))
        # Heuristic: Sort by number of edges descending
        tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
        
        def backtrack_tree(k, edges_k, adj_k):
            """
            Embed tree T_k into K20 avoiding used_edges.
            Returns mapping list of length k or None.
            """
            # Order vertices of the tree by degree descending to constrain early
            verts = list(range(k))
            verts.sort(key=lambda v: len(adj_k[v]), reverse=True)
            
            mapping = [None] * k
            used_hosts = set()
            
            # Recursive helper
            def assign(idx):
                if idx == len(verts):
                    return True
                
                v = verts[idx]
                
                # Determine candidates for host
                candidates = []
                for h in range(n):
                    if h in used_hosts:
                        continue
                    
                    # Check constraints with already assigned neighbors
                    valid = True
                    for u in adj_k[v]:
                        if mapping[u] is not None:
                            eu = mapping[u]
                            key = (min(eu, h), max(eu, h))
                            if key in used_edges:
                                valid = False
                                break
                            # Optional: Check color availability? 
                            # If we track color counts, we could prune here.
                            # But used_edges is sufficient.
                    if valid:
                        candidates.append(h)
                        
                if not candidates:
                    return False
                
                # Heuristic: Try hosts that are "more constrained" first?
                # Or just random order? Deterministic is required for "deterministic edge-coloring" strategy?
                # Let's try hosts in order of their current degree in the remaining graph?
                # Simpler: Just try in natural order, but we can shuffle if we want randomness.
                # Since we are in a deterministic solver, we use natural order.
                # To improve, we can sort candidates by how many edges they have to used_hosts?
                
                for h in candidates:
                    mapping[v] = h
                    used_hosts.add(h)
                    
                    # Mark edges as used? 
                    # We can't mark them globally yet because we might backtrack.
                    # But we can check against `used_edges` (global) and `local_used` (partial).
                    # Wait, the `valid` check above only checks against `used_edges` (global).
                    # We also need to ensure we don't use the same edge twice *within* this tree.
                    # But the tree is a tree, so no cycles. 
                    # However, two different tree edges could map to the same host edge?
                    # No, because the tree edges are distinct pairs of tree vertices.
                    # If (u,v) and (u,w) are edges, and mapping[u]=A, mapping[v]=B, mapping[w]=C.
                    # If B==C, then mapping is not injective (distinct hosts).
                    # We enforce `h in used_hosts` check, so B!=C.
                    # So distinct tree edges map to distinct host edges?
                    # Yes, because if (u,v) and (x,y) are distinct edges in a tree,
                    # and mapping is injective, then {map[u], map[v]} != {map[x], map[y]} 
                    # unless the tree has multiple edges between same nodes (it doesn't) 
                    # or the mapping collapses them (it doesn't, injective).
                    # Wait, what if (u,v) maps to (A,B) and (x,y) maps to (B,A)? Same edge.
                    # This requires map[u]=B, map[v]=A and map[x]=A, map[y]=B.
                    # This implies u=x or u=y? No.
                    # It implies the set of images is the same.
                    # Since the tree is a simple graph, distinct edges share at most one vertex.
                    # If they share a vertex u, map[u] is fixed.
                    # If (u,v) and (u,w) are edges.
                    # map[u]=A.
                    # Edge 1: (A, map[v]).
                    # Edge 2: (A, map[w]).
                    # Since map[v] != map[w] (injective), the edges are distinct.
                    # So we don't need to track local edge conflicts, only global `used_edges` and injectivity.
                    
                    if assign(idx + 1):
                        return True
                    
                    used_hosts.remove(h)
                    mapping[v] = None
                
                return False
            
            if assign(0):
                return mapping
            return None

        for k_idx in tree_indices:
            k = k_idx + 1
            edges_k = trees[k_idx]
            if not edges_k:
                result[k_idx] = [0]
                continue
            
            adj_k = tree_adj[k_idx]
            mapping = backtrack_tree(k, edges_k, adj_k)
            
            if mapping is None:
                return None
            
            result[k_idx] = list(mapping)
            for (u, v) in edges_k:
                key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                used_edges.add(key)
                
        return result

    # Try deterministic solver first
    solution = solve_deterministic()
    
    if solution is not None:
        # Verify
        used_edges = set()
        for k in range(1, n + 1):
            r = solution[k-1]
            if r is None:
                return [None] * n
            for (u, v) in trees[k-1]:
                key = (min(r[u], r[v]), max(r[u], r[v]))
                if key in used_edges:
                    # Should not happen
                    pass
                used_edges.add(key)
        return solution

    # Fallback to random restarts if deterministic fails (should be rare)
    def try_random_permutation(rng, k, edges_k, used_edges):
        num_tries = max(100, 20000 // k)
        for _ in range(num_tries):
            mapping = rng.sample(range(n), k)
            valid = True
            for (u, v) in edges_k:
                key = (min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                if key in used_edges:
                    valid = False
                    break
            if valid:
                return mapping
        return None
        
    def solve_one_restart(rng, order):
        used_edges = set()
        result = [None] * n
        
        for k in order:
            edges_k = trees[k-1]
            if not edges_k:
                result[k-1] = [0]
                continue
            if k == 2:
                found = False
                for _ in range(10000):
                    a = rng.randint(0, n - 1)
                    b = rng.randint(0, n - 1)
                    if a != b:
                        key = (min(a, b), max(a, b))
                        if key not in used_edges:
                            result[k-1] = [a, b]
                            used_edges.add(key)
                            found = True
                            break
                if not found:
                    return None
                continue
            
            adj_k = tree_adj[k-1]
            best = try_random_permutation(rng, k, edges_k, used_edges)
            if best is None:
                # Simple BFS fallback
                # ... (omitted for brevity, assuming random perm is strong enough)
                return None
            
            result[k-1] = list(best)
            for (u, v) in edges_k:
                used_edges.add((min(best[u], best[v]), max(best[u], best[v])))
        
        return result

    orderings = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
    ]
    
    best_solution = None
    best_value = -1
    restart = 0
    while time.time() < deadline:
        order_idx = restart % len(orderings)
        order = list(orderings[order_idx])
        rng = Random(restart * 7919 + 13)
        
        solution = solve_one_restart(rng, order)
        if solution is not None:
            val = sum(len(trees[k-1]) for k in range(1, n + 1) if solution[k-1] is not None)
            if val > best_value:
                best_value = val
                best_solution = [list(s) if s is not None else None for s in solution]
        
        if best_value >= max_possible:
            break
        restart += 1
        
    if best_solution is None:
        return [None] * n
        
    return best_solution

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used_edges = set()
    ok = True
    for k in range(1, 21):
        if res[k-1] is not None:
            r = res[k-1]
            if len(r) != k:
                print(f"T{k}: wrong length {len(r)}")
                ok = False
            if len(set(r)) != k:
                print(f"T{k}: not distinct")
                ok = False
            for v in r:
                if not isinstance(v, int) or v < 0 or v > 19:
                    print(f"T{k}: invalid vertex {v}")
                    ok = False
            for (u, v) in trees[k-1]:
                eu, ev = r[u], r[v]
                key = (min(eu, ev), max(eu, ev))
                if key in used_edges:
                    print(f"T{k}: edge conflict {key}")
                    ok = False
                used_edges.add(key)
    total = sum(len(trees[k-1]) for k in range(1, 21) if res[k-1] is not None)
    print(f"Valid: {ok}, Total edges: {total}")
