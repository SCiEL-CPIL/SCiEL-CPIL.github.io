import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 105  # safety margin before 120s limit
    
    N = 20
    all_pairs = list(combinations(range(N), 2))
    num_pairs = len(all_pairs)  # 190
    
    # Index for pairs
    pair_to_idx = {}
    for i, (u, v) in enumerate(all_pairs):
        pair_to_idx[(u, v)] = i
        if u > v:
            pair_to_idx[(v, u)] = i
    
    # For each pair of edges that share no vertices, check if adding both could create K2,3
    # Actually, K2,3 means two vertices with 3 common neighbors.
    # Two edges (a,b) and (c,d) form part of a K2,3 if {a,b} are the "2" part and there exist 3 common neighbors.
    # But we're building edge by edge. The constraint is: no two vertices have 3 common neighbors.
    # 
    # Let me think differently. I'll use a greedy + local search approach.
    # 
    # Strategy: 
    # 1. Start with a good initial graph (maybe random dense, then fix violations)
    # 2. Local search: try to add edges, remove edges to fix violations
    # 3. Or: branch and bound with bitsets
    #
    # Given 20 vertices and 190 possible edges, let's try a smart greedy approach:
    # - Sort all edges by some heuristic
    # - Add edges greedily if they don't violate K2,3
    # - Then try local improvements (swap edges)
    #
    # Actually, let me think about what a good K2,3-free graph looks like.
    # The seed has 5 copies of K4 = 5*6 = 30 edges.
    # We can probably do much better.
    #
    # A K2,3-free graph on 20 vertices: each pair of vertices can have at most 2 common neighbors.
    # This is equivalent to saying the adjacency matrix squared (diagonal removed) has entries <= 2.
    #
    # Let me try a constructive approach using known dense graphs.
    # 
    # For now, let me use a randomized local search:
    # 1. Start with random graph, fix K2,3 violations
    # 2. Try to add random edges
    # 3. Try swaps (remove one edge, add another)
    
    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v."""
        cnt = 0
        for w in range(N):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                cnt += 1
        return cnt
    
    def has_k23_violation(adj):
        """Check if any pair has >= 3 common neighbors. Return (u,v) if found."""
        for u in range(N):
            for v in range(u+1, N):
                if common_neighbors_count(adj, u, v) >= 3:
                    return (u, v)
        return None
    
    def count_violations(adj):
        """Count number of pairs with >= 3 common neighbors."""
        cnt = 0
        for u in range(N):
            for v in range(u+1, N):
                if common_neighbors_count(adj, u, v) >= 3:
                    cnt += 1
        return cnt
    
    def num_edges(adj):
        cnt = 0
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    cnt += 1
        return cnt
    
    def edge_list(adj):
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    # Random seed for reproducibility in case we need it
    random.seed(42)
    
    # Phase 1: Generate initial graph by greedy addition
    # Try multiple random orderings and keep the best
    best_edges = 0
    best_adj = None
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        adj = [[False] * N for _ in range(N)]
        edge_order = list(all_pairs)
        random.shuffle(edge_order)
        
        for (u, v) in edge_order:
            # Check if adding (u,v) would create a K2,3 violation
            # Adding edge (u,v) affects common neighbor counts for pairs involving u or v
            ok = True
            for w in range(N):
                if w == u or w == v:
                    continue
                # Check pair (u, w): adding (u,v) means v becomes a common neighbor if adj[w][v]
                if adj[w][v]:
                    if common_neighbors_count(adj, u, w) >= 2:  # would become 3
                        ok = False
                        break
                # Check pair (v, w): similar
                if adj[u][w]:
                    if common_neighbors_count(adj, v, w) >= 2:
                        ok = False
                        break
            if ok:
                adj[u][v] = adj[v][u] = True
        
        ne = num_edges(adj)
        if ne > best_edges:
            best_edges = ne
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4*k+a, 4*k+b] for a in range(4) for b in range(a+1, 4)]
        return edges
    
    adj = best_adj
    current_edges = best_edges
    
    # Phase 2: Local search - try to add more edges
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        for (u, v) in random.sample(all_pairs, min(200, num_pairs)):
            if adj[u][v]:
                continue
            # Try adding edge (u,v)
            ok = True
            for w in range(N):
                if w == u or w == v:
                    continue
                if adj[w][v] and common_neighbors_count(adj, u, w) >= 2:
                    ok = False
                    break
                if adj[u][w] and common_neighbors_count(adj, v, w) >= 2:
                    ok = False
                    break
            if ok:
                adj[u][v] = adj[v][u] = True
                current_edges += 1
                improved = True
                break  # Restart scan after modification
        
        if not improved:
            # No single edge can be added; try swaps
            swapped = False
            for (u1, v1) in random.sample(all_pairs, min(100, num_pairs)):
                if not adj[u1][v1]:
                    continue
                # Remove (u1,v1)
                adj[u1][v1] = adj[v1][u1] = False
                current_edges -= 1
                
                # Try to add a different edge
                for (u2, v2) in random.sample(all_pairs, min(50, num_pairs)):
                    if adj[u2][v2] or (u2==u1 and v2==v1):
                        continue
                    ok = True
                    for w in range(N):
                        if w == u2 or w == v2:
                            continue
                        if adj[w][v2] and common_neighbors_count(adj, u2, w) >= 2:
                            ok = False
                            break
                        if adj[u2][w] and common_neighbors_count(adj, v2, w) >= 2:
                            ok = False
                            break
                    if ok:
                        adj[u2][v2] = adj[v2][u2] = True
                        current_edges += 1
                        swapped = True
                        break
                
                if not swapped:
                    # Restore
                    adj[u1][v1] = adj[v1][u1] = True
                    current_edges += 1
        
        if time.time() > deadline:
            break
    
    # Phase 3: Try more aggressive local search with different random seeds
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        trial_adj = [row[:] for row in adj]
        trial_edges = current_edges
        
        # Random perturbation: remove some edges, add others
        removed = []
        for (u, v) in random.sample(all_pairs, min(20, trial_edges)):
            if trial_adj[u][v]:
                trial_adj[u][v] = trial_adj[v][u] = False
                trial_edges -= 1
                removed.append((u, v))
        
        # Try to add edges back and more
        for (u, v) in random.sample(all_pairs, min(300, num_pairs)):
            if trial_adj[u][v]:
                continue
            ok = True
            for w in range(N):
                if w == u or w == v:
                    continue
                if trial_adj[w][v] and common_neighbors_count(trial_adj, u, w) >= 2:
                    ok = False
                    break
                if trial_adj[u][w] and common_neighbors_count(trial_adj, v, w) >= 2:
                    ok = False
                    break
            if ok:
                trial_adj[u][v] = trial_adj[v][u] = True
                trial_edges += 1
        
        if trial_edges > current_edges:
            adj = trial_adj
            current_edges = trial_edges
    
    # Final validation
    violation = has_k23_violation(adj)
    if violation:
        # Remove edges until valid
        changed = True
        while changed and time.time() < deadline:
            changed = False
            v = has_k23_violation(adj)
            if v is None:
                break
            u, w = v
            # Remove an edge incident to one of u or w that contributes to common neighbors
            common = [x for x in range(N) if x != u and x != w and adj[u][x] and adj[w][x]]
            if common:
                # Remove a random common neighbor connection
                cn = random.choice(common)
                adj[u][cn] = adj[cn][u] = False
                adj[w][cn] = adj[cn][w] = False
                current_edges -= 2
                changed = True
            else:
                break
    
    if time.time() > deadline - 5:
        # Just return what we have, but make sure it's valid
        # Quick check and fix
        v = has_k23_violation(adj)
        while v is not None:
            u, w = v
            common = [x for x in range(N) if x != u and x != w and adj[u][x] and adj[w][x]]
            if common:
                cn = common[0]
                adj[u][cn] = adj[cn][u] = False
                adj[w][cn] = adj[cn][w] = False
            else:
                break
            v = has_k23_violation(adj)
    
    return edge_list(adj)
