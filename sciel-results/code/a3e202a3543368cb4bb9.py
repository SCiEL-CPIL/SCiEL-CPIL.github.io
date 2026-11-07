import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    # Initialize adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute all candidate edges that do not form a triangle with the base graph
    # Note: Triangle-freeness must be maintained with ALL added edges, not just base.
    # So we can't just pre-filter against base. We must check against current adj.
    # However, we can pre-filter candidates that are "impossible" due to base graph triangles?
    # No, base is a path, so no triangles. Any edge (u,v) is valid initially.
    # But adding (u,v) might make (u,w) invalid later.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
                
    def get_distances(adj):
        """Returns list of lists of distances, or None if disconnected."""
        dists = [[-1]*n for _ in range(n)]
        for src in range(n):
            dists[src][src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dists[src][v] == -1:
                        dists[src][v] = dists[src][u] + 1
                        q.append(v)
            if any(d == -1 for d in dists[src]):
                return None
        return dists

    def get_diameter(dists):
        if dists is None:
            return 999
        max_d = 0
        for i in range(n):
            for j in range(i+1, n):
                if dists[i][j] > max_d:
                    max_d = dists[i][j]
        return max_d

    def is_triangle_free(u, v, adj):
        # Check if adding (u,v) creates a triangle
        # i.e., is there a w such that (u,w) and (v,w) are in adj?
        common = adj[u] & adj[v]
        return len(common) == 0

    # Initial state
    current_adj = {i: set(adj[i]) for i in range(n)}
    current_dists = get_distances(current_adj)
    current_diam = get_diameter(current_dists)
    
    # Identify "bad" pairs: pairs with distance > 4
    bad_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            if current_dists[i][j] > 4:
                bad_pairs.add((i, j))
                
    best_solution = None
    best_solution_size = float('inf')
    
    # Precompute for each candidate edge, which bad pairs it "fixes" (reduces distance to <= 4)
    # This is expensive to compute dynamically, so we do it on the fly in the search,
    # but we can use a static lower bound based on the number of bad pairs.
    
    # Branch and Bound
    # State: (added_edges, adj, dists, bad_pairs, depth)
    # We use a recursive search with pruning.
    
    # To speed up, we can pre-compute the "potential" of each edge on the base graph.
    # If an edge doesn't reduce the distance of any bad pair in the base graph, it's likely useless?
    # Not necessarily, because other edges might change the topology.
    # But for a lower bound:
    # Lower Bound on edges needed = ceil(|bad_pairs| / max_pairs_fixed_by_any_edge)
    
    # Let's compute max_pairs_fixed for each candidate in the BASE graph first.
    # This is a static property that gives a lower bound.
    max_fix_count = 0
    edge_fix_counts = {}
    for u, v in candidates:
        # Tentatively add edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        if not is_triangle_free(u, v, current_adj):
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            continue
            
        new_dists = get_distances(current_adj)
        if new_dists is None:
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            continue
            
        fixed = 0
        for (i, j) in bad_pairs:
            if new_dists[i][j] <= 4:
                fixed += 1
                
        edge_fix_counts[(u, v)] = fixed
        if fixed > max_fix_count:
            max_fix_count = fixed
            
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        
    if max_fix_count == 0:
        # No single edge fixes any bad pair? Then it's impossible or requires complex interaction.
        # But path diameter is 23, so edges should fix things.
        pass

    # Recursive search
    def search(added_edges, adj, dists, bad_pairs, depth, current_time_limit):
        nonlocal best_solution, best_solution_size
        
        if time.time() > current_time_limit:
            return
            
        # Pruning: if current size + lower bound >= best known, prune
        lb = 0
        if max_fix_count > 0:
            lb = (len(bad_pairs) + max_fix_count - 1) // max_fix_count
        else:
            lb = len(bad_pairs) # Very weak bound
            
        if len(added_edges) + lb >= best_solution_size:
            return
            
        # Check if we are done
        if not bad_pairs:
            if len(added_edges) < best_solution_size:
                best_solution = list(added_edges)
                best_solution_size = len(added_edges)
            return
            
        # If we have too many edges, prune
        if len(added_edges) >= best_solution_size:
            return
            
        # Select an edge to branch on
        # Heuristic: pick an edge that fixes the most bad pairs
        # Or pick a bad pair, and branch on edges that fix it.
        # The latter is more efficient for set cover.
        
        if bad_pairs:
            # Pick a random or strategic bad pair
            # Strategic: pick the pair that is "hardest" to fix?
            # For now, just pick the first one.
            target_pair = next(iter(bad_pairs))
            i, j = target_pair
            
            # Find all candidate edges that could fix this pair
            # An edge (u, v) fixes (i, j) if dist(i, j) in new graph <= 4.
            # We don't know this without adding it.
            # So we must branch on ALL valid candidates? That's too many.
            
            # Alternative: Branch on the set of edges that are "plausible" for this pair.
            # Plausible if they are "close" to i or j?
            
            # Let's stick to branching on candidates, but ordered by fix count.
            # We can filter candidates:
            # 1. Must be triangle-free with current adj.
            # 2. Must not be in added_edges.
            
            valid_candidates = []
            for u, v in candidates:
                if (u, v) in added_edges:
                    continue
                if not is_triangle_free(u, v, adj):
                    continue
                valid_candidates.append((u, v))
                
            # Sort by static fix count (higher is better)
            valid_candidates.sort(key=lambda x: edge_fix_counts.get(x, 0), reverse=True)
            
            # Limit branching factor to avoid explosion?
            # With n=24, candidates ~276.
            # If we branch on all, it's huge.
            # We should only branch on edges that actually reduce the distance of the target_pair.
            
            promising_candidates = []
            for u, v in valid_candidates:
                # Tentatively add
                adj[u].add(v)
                adj[v].add(u)
                
                # Check if it fixes the target pair
                # We can do a quick BFS from i to j
                dist_ij = -1
                q = deque([i])
                visited = {i}
                while q:
                    curr = q.popleft()
                    if curr == j:
                        dist_ij = dists[i][j] # Wait, dists is old.
                        # We need to compute new dist.
                        break
                    for nb in adj[curr]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                            
                # Actually, computing full dist is better for the new bad_pairs set.
                new_dists = get_distances(adj)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_dists is None:
                    continue
                    
                if new_dists[i][j] <= 4:
                    # This edge is promising for this pair
                    # Also, does it keep the graph triangle-free? Yes, checked above.
                    promising_candidates.append((u, v, new_dists))
                    
            if not promising_candidates:
                # No edge fixes this pair? Then this branch is dead.
                return
                
            # Sort promising candidates by how many total bad pairs they fix
            def score_candidate(edge_data):
                u, v, new_dists = edge_data
                fixed = 0
                for (pi, pj) in bad_pairs:
                    if new_dists[pi][pj] <= 4:
                        fixed += 1
                return fixed
                
            promising_candidates.sort(key=lambda x: score_candidate(x), reverse=True)
            
            # Branch on top K candidates to limit width
            # K depends on depth. Early on, we can be wider.
            K = 10 if depth < 3 else 5
            
            for u, v, new_dists in promising_candidates[:K]:
                # Add edge
                new_added = list(added_edges) + [(u, v)]
                
                # Update bad_pairs
                new_bad_pairs = set()
                for (pi, pj) in bad_pairs:
                    if new_dists[pi][pj] > 4:
                        new_bad_pairs.add((pi, pj))
                        
                # Recurse
                search(new_added, adj, new_dists, new_bad_pairs, depth + 1, current_time_limit)
                
                # Note: We don't remove u,v from adj in the loop because we passed new_dists
                # But we need to ensure adj is consistent for the next iteration?
                # In the loop, we iterate over promising_candidates which were computed
                # with the ORIGINAL adj. So adj is unchanged.
                # However, the recursive call uses `adj` (which is the original) and `new_dists`.
                # This is inconsistent! The recursive call expects `adj` to match `new_dists`.
                
                # FIX: We must pass a modified adj or reconstruct it.
                # Reconstructing is expensive.
                # Better: Modify adj, recurse, undo.
                
    # Revised Search with proper state management
    def search_v2(added_edges, adj, dists, bad_pairs, depth):
        nonlocal best_solution, best_solution_size
        
        if time.time() > deadline:
            return
            
        # Pruning
        lb = 0
        if max_fix_count > 0:
            lb = (len(bad_pairs) + max_fix_count - 1) // max_fix_count
        else:
            lb = len(bad_pairs)
            
        if len(added_edges) + lb >= best_solution_size:
            return
            
        if not bad_pairs:
            if len(added_edges) < best_solution_size:
                best_solution = list(added_edges)
                best_solution_size = len(added_edges)
            return
            
        if len(added_edges) >= best_solution_size:
            return
            
        # Pick target bad pair
        target_pair = next(iter(bad_pairs))
        i, j = target_pair
        
        # Find edges that fix this pair
        valid_candidates = []
        for u, v in candidates:
            if (u, v) in added_edges:
                continue
            if not is_triangle_free(u, v, adj):
                continue
            
            # Test if this edge fixes (i, j)
            adj[u].add(v)
            adj[v].add(u)
            
            # Quick check: is distance i->j <= 4?
            # We can use the precomputed new_dists if we had them, but we don't.
            # Let's compute new_dists for the whole graph to update bad_pairs.
            new_dists = get_distances(adj)
            
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_dists is None:
                continue
                
            if new_dists[i][j] <= 4:
                # Count how many bad pairs this fixes
                fixed_count = 0
                for (pi, pj) in bad_pairs:
                    if new_dists[pi][pj] <= 4:
                        fixed_count += 1
                valid_candidates.append((u, v, new_dists, fixed_count))
                
        if not valid_candidates:
            return
            
        # Sort by fixed_count
        valid_candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Branch
        K = 8 if depth < 2 else 4
        for u, v, new_dists, fixed_count in valid_candidates[:K]:
            # Commit edge
            adj[u].add(v)
            adj[v].add(u)
            new_added = list(added_edges) + [(u, v)]
            
            # Update bad pairs
            new_bad_pairs = set()
            for (pi, pj) in bad_pairs:
                if new_dists[pi][pj] > 4:
                    new_bad_pairs.add((pi, pj))
                    
            search_v2(new_added, adj, new_dists, new_bad_pairs, depth + 1)
            
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)

    # Initial call
    # To ensure we have a valid solution, let's run a quick greedy first to set best_solution_size
    # Greedy: pick edge that fixes most bad pairs
    greedy_adj = {i: set(adj[i]) for i in range(n)}
    greedy_dists = get_distances(greedy_adj)
    greedy_bad = set(bad_pairs)
    greedy_added = []
    
    while greedy_bad and time.time() < deadline - 5:
        best_edge = None
        best_fixed = -1
        best_dists = None
        
        for u, v in candidates:
            if (u, v) in greedy_added:
                continue
            if not is_triangle_free(u, v, greedy_adj):
                continue
            
            greedy_adj[u].add(v)
            greedy_adj[v].add(u)
            
            new_dists = get_distances(greedy_adj)
            
            greedy_adj[u].remove(v)
            greedy_adj[v].remove(u)
            
            if new_dists is None:
                continue
                
            fixed = 0
            for (pi, pj) in greedy_bad:
                if new_dists[pi][pj] <= 4:
                    fixed += 1
                    
            if fixed > best_fixed:
                best_fixed = fixed
                best_edge = (u, v)
                best_dists = new_dists
                
        if best_edge is None or best_fixed == 0:
            break
            
        u, v = best_edge
        greedy_added.append((u, v))
        greedy_adj[u].add(v)
        greedy_adj[v].add(u)
        
        # Update greedy_bad
        new_greedy_bad = set()
        for (pi, pj) in greedy_bad:
            if best_dists[pi][pj] > 4:
                new_greedy_bad.add((pi, pj))
        greedy_bad = new_greedy_bad
        
    if not greedy_bad:
        best_solution = greedy_added
        best_solution_size = len(greedy_added)
    else:
        # Fallback to hub if greedy fails
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        hub_added = []
        for v in targets:
            if (min(hub, v), max(hub, v)) not in base_set:
                hub_added.append((hub, v))
        best_solution = hub_added
        best_solution_size = len(hub_added)
        
    # Now run B&B to improve
    # Reset adj to base
    for i in range(n):
        adj[i] = set(base_edges[i-1][1] if i > 0 else -1) # Dummy, will re-init properly
    for i in range(n):
        adj[i] = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    search_v2([], adj, current_dists, bad_pairs, 0)
    
    if best_solution is None:
        # Should not happen if hub fallback worked
        return []
        
    return [[int(u), int(v)] for u, v in best_solution]

# EVOLVE-BLOCK-END
