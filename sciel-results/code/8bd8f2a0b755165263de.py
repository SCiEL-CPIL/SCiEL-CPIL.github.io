"""Triangle-free diameter-four augmentation - exact branch and bound."""
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # All candidate edges: (u,v) with |u-v| >= 3 (|u-v|==2 would create triangle with base)
    # Actually |u-v|==2: u and v share neighbor u+1 (or v+1), so adding (u,v) creates triangle (u, u+1, v)
    # So candidates are pairs with |u-v| >= 3
    candidates = []
    for u in range(n):
        for v in range(u + 3, n):
            candidates.append((u, v))
    
    # Identify pairs that need help: |i-j| > 4
    # These pairs have path distance > 4, so we need added edges to bring them to <= 4
    
    def build_full_adj(added_edges):
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True
    
    def check_triangle_free_incremental(adj, new_u, new_v):
        """Check if adding edge (new_u, new_v) to adj keeps it triangle-free."""
        # new_u and new_v must not share a common neighbor
        if adj[new_u] & adj[new_v]:
            return False
        return True
    
    def compute_diameter(adj):
        """BFS from each node, return max distance."""
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d
    
    def far_pairs_count(adj, threshold=4):
        """Count pairs with distance > threshold."""
        count = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            for j in range(src + 1, n):
                if dist[j] > threshold:
                    count += 1
        return count
    
    # Precompute: for each candidate edge, which far pairs it potentially helps
    # A far pair (i,j) is "covered" by edge (u,v) if adding (u,v) reduces dist(i,j)
    # This is complex, so we'll use a heuristic: prefer edges that connect distant parts
    
    # Sort candidates by "span" (v - u) descending - longer edges cover more ground
    candidates_sorted = sorted(candidates, key=lambda e: -(e[1] - e[0]))
    
    deadline = time.time() + 80  # safety margin
    
    best_solution = None
    best_count = len(candidates) + 1  # infinity
    
    # Branch and bound
    # State: current set of added edges, current adjacency
    # We'll use a recursive approach with memoization on the set of "still-far" pairs
    
    # Actually, let's use a different strategy: 
    # 1. Find all pairs with path distance > 4
    # 2. For each such pair, it needs at least one "shortcut" edge
    # 3. Use set cover style: each edge covers some pairs
    
    # Let's compute which pairs are "far" in the base
    far_pairs = set()
    for i in range(n):
        for j in range(i + 5, n):  # |i-j| > 4 means j >= i+5
            far_pairs.add((i, j))
    
    # For each candidate edge, compute which far pairs it "helps"
    # Edge (u,v) helps pair (i,j) if in the graph with edge (u,v) added,
    # dist(i,j) <= 4. This is expensive to compute for all pairs.
    
    # Simpler heuristic: edge (u,v) with span s = v-u helps pairs that span across [u,v]
    # Specifically, pairs (i,j) where i < u and j > v, or i < v and j > u, etc.
    
    # Let's just do a smart greedy + local search with exact verification
    
    # Phase 1: Greedy construction
    def greedy_construct():
        adj = [set(base_adj[i]) for i in range(n)]
        added = []
        current_far = len(far_pairs)
        
        while current_far > 0 and time.time() < deadline:
            best_edge = None
            best_far = current_far
            
            for (u, v) in candidates_sorted:
                if (u, v) in added or (v, u) in added:
                    continue
                # Check triangle-free
                if adj[u] & adj[v]:
                    continue
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                trial_far = far_pairs_count(adj, 4)
                adj[u].discard(v)
                adj[v].discard(u)
                
                if trial_far < best_far:
                    best_far = trial_far
                    best_edge = (u, v)
            
            if best_edge is None or best_far >= current_far:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added.append(best_edge)
            current_far = best_far
        
        return added, adj
    
    # Phase 2: Local search - try to remove edges, then swap
    def local_search(added, adj):
        # Try removing edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(added) - 1, -1, -1):
                trial_added = added[:i] + added[i+1:]
                trial_adj = [set(base_adj[k]) for k in range(n)]
                for u, v in trial_added:
                    trial_adj[u].add(v)
                    trial_adj[v].add(u)
                if is_triangle_free(trial_adj) and compute_diameter(trial_adj) <= 4:
                    added = trial_added
                    adj = trial_adj
                    changed = True
                    break
        
        # Try swapping: replace one edge with another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(added)):
                for (u_new, v_new) in candidates_sorted:
                    if (u_new, v_new) in added or (v_new, u_new) in added:
                        continue
                    # Remove edge i, add new edge
                    trial_added = added[:i] + added[i+1:] + [(u_new, v_new)]
                    trial_adj = [set(base_adj[k]) for k in range(n)]
                    for u, v in trial_added:
                        trial_adj[u].add(v)
                        trial_adj[v].add(u)
                    if is_triangle_free(trial_adj) and compute_diameter(trial_adj) <= 4:
                        # Try to remove more
                        trial_added2 = list(trial_added)
                        for k in range(len(trial_added2) - 1, -1, -1):
                            trial2 = trial_added2[:k] + trial_added2[k+1:]
                            trial_adj2 = [set(base_adj[m]) for m in range(n)]
                            for u, v in trial2:
                                trial_adj2[u].add(v)
                                trial_adj2[v].add(u)
                            if is_triangle_free(trial_adj2) and compute_diameter(trial_adj2) <= 4:
                                trial_added2 = trial2
                        if len(trial_added2) < len(added):
                            added = trial_added2
                            adj = [set(base_adj[k]) for k in range(n)]
                            for u, v in added:
                                adj[u].add(v)
                                adj[v].add(u)
                            improved = True
                            break
                if improved:
                    break
        
        return added, adj
    
    # Run greedy + local search
    added, adj = greedy_construct()
    added, adj = local_search(added, adj)
    
    # Try multiple restarts with different orderings
    best_added = added
    best_len = len(added)
    
    for restart in range(1, 20):
        if time.time() > deadline:
            break
        # Shuffle candidates with bias toward long edges
        import random
        random.seed(restart)
        shuffled = candidates_sorted[:]
        random.shuffle(shuffled)
        # Re-sort with some randomness
        shuffled = sorted(shuffled, key=lambda e: (-(e[1] - e[0]), random.random()))
        
        # Quick greedy with this ordering
        adj2 = [set(base_adj[i]) for i in range(n)]
        added2 = []
        current_far = len(far_pairs)
        
        while current_far > 0 and time.time() < deadline:
            best_edge = None
            best_far = current_far
            for (u, v) in shuffled:
                if (u, v) in added2 or (v, u) in added2:
                    continue
                if adj2[u] & adj2[v]:
                    continue
                adj2[u].add(v)
                adj2[v].add(u)
                trial_far = far_pairs_count(adj2, 4)
                adj2[u].discard(v)
                adj2[v].discard(u)
                if trial_far < best_far:
                    best_far = trial_far
                    best_edge = (u, v)
            if best_edge is None or best_far >= current_far:
                break
            u, v = best_edge
            adj2[u].add(v)
            adj2[v].add(u)
            added2.append(best_edge)
            current_far = best_far
        
        # Local search
        added2, adj2 = local_search(added2, adj2)
        
        if len(added2) < best_len:
            best_added = added2
            best_len = len(added2)
    
    # Final validation
    final_adj = [set(base_adj[i]) for i in range(n)]
    for u, v in best_added:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    assert is_triangle_free(final_adj), "Triangle found"
    diam = compute_diameter(final_adj)
    assert diam <= 4, f"Diameter too large: {diam}"
    
    result = []
    for u, v in best_added:
        result.append([int(u), int(v)])
    
    return result
