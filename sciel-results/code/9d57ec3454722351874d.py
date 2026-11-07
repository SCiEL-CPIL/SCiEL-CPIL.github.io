import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    n = 15
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute pair index
    pair_idx = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_idx[(u, v)] = idx
    
    def _build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_valid(edge_set):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
        return True
    
    def _edge_ok(adj, deg, u, v):
        """Check if adding edge (u,v) keeps constraint. Assumes (u,v) not present."""
        # Paths with (u,v) as middle edge: s-u-v-t
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[t] == ds:
                    return False
        # Paths with (u,v) as first edge: u-v-x-y (endpoints u, y)
        du = deg[u]
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == v or y == u:
                    continue
                if deg[y] == du:
                    return False
        # Paths with (u,v) as last edge: x-y-v-u (endpoints x, u)
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == v or y == u:
                    continue
                if deg[x] == du:
                    return False
        return True
    
    # Generate good initial solutions via randomized greedy
    best_set = set()
    best_count = 0
    
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        rng = random.Random(trial * 7919 + 1)
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edge_set = set()
        pairs = list(all_pairs)
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edge_set.add((u, v))
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_set = set(edge_set)
    
    # Local search to improve
    def local_search(edge_set):
        adj = _build_adj(edge_set)
        deg = [len(adj[i]) for i in range(n)]
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            # Try adding missing edges
            for u, v in all_pairs:
                if time.time() > deadline - 25:
                    break
                if (u, v) in edge_set:
                    continue
                if _edge_ok(adj, deg, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edge_set.add((u, v))
                    improved = True
            # Try swaps
            if time.time() < deadline - 25:
                for (a, b) in list(edge_set):
                    if time.time() > deadline - 25:
                        break
                    for (c, d) in all_pairs:
                        if (c, d) in edge_set:
                            continue
                        # Remove (a,b), try add (c,d)
                        adj[a].discard(b)
                        adj[b].discard(a)
                        deg[a] -= 1
                        deg[b] -= 1
                        edge_set.discard((a, b))
                        if _edge_ok(adj, deg, c, d):
                            adj[c].add(d)
                            adj[d].add(c)
                            deg[c] += 1
                            deg[d] += 1
                            edge_set.add((c, d))
                            improved = True
                        else:
                            # Restore
                            adj[a].add(b)
                            adj[b].add(a)
                            deg[a] += 1
                            deg[b] += 1
                            edge_set.add((a, b))
        return edge_set
    
    best_set = local_search(best_set)
    best_count = len(best_set)
    
    # Branch and bound to find optimal
    # We'll use a recursive approach with pruning
    # State: set of edges, set of candidate edges to consider
    # Upper bound: current edges + number of remaining candidates
    
    best_global = best_count
    best_global_set = set(best_set)
    
    def branch_and_bound(current_edges, candidates, depth):
        nonlocal best_global, best_global_set
        
        if time.time() > deadline - 2:
            return
        
        current_count = len(current_edges)
        remaining = len(candidates)
        
        # Upper bound: current + remaining
        if current_count + remaining <= best_global:
            return
        
        if not candidates:
            if current_count > best_global:
                best_global = current_count
                best_global_set = set(current_edges)
            return
        
        # Pick the next candidate
        # Use a heuristic: pick the edge that conflicts with the most other candidates
        u, v = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include this edge
        adj = _build_adj(current_edges)
        deg = [len(adj[i]) for i in range(n)]
        
        if _edge_ok(adj, deg, u, v):
            new_edges = set(current_edges)
            new_edges.add((u, v))
            # Filter candidates that conflict with adding (u,v)
            # An edge (a,b) conflicts if adding it to new_edges would violate constraint
            new_candidates = []
            for a, b in rest:
                # Check if (a,b) can still be added
                adj[a].add(b)
                adj[b].add(a)
                deg[a] += 1
                deg[b] += 1
                ok = _edge_ok(adj, deg, a, b) if (a, b) not in new_edges else True
                # Actually we need to check if (a,b) is compatible with new_edges
                # Simpler: just check if (a,b) can be added to new_edges
                adj[a].remove(b)
                adj[b].remove(a)
                deg[a] -= 1
                deg[b] -= 1
                
                # Check compatibility
                adj2 = set(adj)  # This won't work well, let me redo
                # Actually let's just check if (a,b) can be added to new_edges
                adj_new = _build_adj(new_edges)
                deg_new = [len(adj_new[i]) for i in range(n)]
                if (a, b) in new_edges:
                    new_candidates.append((a, b))
                elif _edge_ok(adj_new, deg_new, a, b):
                    new_candidates.append((a, b))
            
            branch_and_bound(new_edges, new_candidates, depth + 1)
        
        # Option 2: Exclude this edge
        branch_and_bound(current_edges, rest, depth + 1)
    
    # Run branch and bound with a time limit
    # Start with the best solution's edges as base
    # Candidates are the missing edges
    missing = [p for p in all_pairs if p not in best_set]
    
    # For efficiency, only branch on a subset
    if time.time() < deadline - 10:
        # Try B&B from empty set with all pairs as candidates
        # But this is too large, so use the best set as starting point
        # and try to add more
        branch_and_bound(set(best_set), missing, 0)
    
    # Final verification and cleanup
    if not check_valid(best_global_set):
        # Fallback to best_set if it's valid
        if check_valid(best_set):
            best_global_set = best_set
            best_global = len(best_set)
    
    result = []
    seen = set()
    for u, v in best_global_set:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    
    return result
