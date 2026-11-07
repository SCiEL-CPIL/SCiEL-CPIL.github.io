import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Precompute all candidate edges (non-base)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    # For each candidate edge, precompute which other candidates it conflicts with
    # (i.e., which other candidate edges would form a triangle with it, given base edges)
    # Two added edges (a,b) and (c,d) conflict if there exists w such that
    # (a,w) and (b,w) are in base, OR (a,w) in base and (c,w) in added, etc.
    # More precisely: adding edge (u,v) creates a triangle if there's w with (u,w) and (v,w) both in the graph.
    # For two added edges e1=(a,b) and e2=(c,d):
    #   They conflict if:
    #   - a==c or a==d or b==c or b==d (share a vertex) AND the other vertex forms triangle
    #   - Actually, let me think more carefully.
    #   
    #   Adding (a,b) and (c,d) to base creates a triangle if:
    #   Case 1: (a,b) creates triangle with base: exists w with (a,w),(b,w) in base
    #   Case 2: (c,d) creates triangle with base: exists w with (c,w),(d,w) in base  
    #   Case 3: (a,b) and (c,d) together create triangle: 
    #     - a==c: then need (b,d) in base or added... but we only have base+these two
    #       If a==c, triangle (a,b,d) exists if (b,d) in base. Similarly (a,c,d) impossible since a==c.
    #     - More generally: triangle uses 3 edges from {base} ∪ {(a,b),(c,d)}
    #       With exactly 2 added edges, the third must be base.
    #       So we need 3 vertices x,y,z such that exactly 2 of (x,y),(y,z),(z,x) are in {(a,b),(c,d)} and the third is in base.
    #       Sub-cases:
    #       - (a,b) and (c,d) share vertex a=c: then need (b,d) in base → triangle (a,b,d)
    #       - (a,b) and (c,d) share vertex a=d: then need (b,c) in base → triangle (a,b,c)
    #       - (a,b) and (c,d) share vertex b=c: then need (a,d) in base → triangle (b,a,d)
    #       - (a,b) and (c,d) share vertex b=d: then need (a,c) in base → triangle (b,a,c)
    #       - (a,b) and (c,d) disjoint: no triangle with exactly 2 added edges
    #   Case 4: One added edge creates triangle with the other added edge and a base edge - covered above
    #   Case 5: Both added edges + one base edge form triangle - covered in case 3
    
    # Let me precompute conflict sets
    conflict = [set() for _ in range(len(candidates))]
    for i, (a, b) in enumerate(candidates):
        # Check if (a,b) alone creates triangle with base
        for w in range(n):
            if w != a and w != b:
                if w in base_adj[a] and w in base_adj[b]:
                    # (a,b) is incompatible with base alone - skip this candidate entirely
                    conflict[i].add(i)  # mark as self-conflict
                    break
    
    # Check pairwise conflicts
    for i in range(len(candidates)):
        if i in conflict[i]:
            continue
        a, b = candidates[i]
        for j in range(i + 1, len(candidates)):
            if j in conflict[j]:
                continue
            c, d = candidates[j]
            # Check if adding both creates a triangle
            # They share a vertex?
            shared = False
            if a == c:
                if d in base_adj[b]:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    shared = True
            elif a == d:
                if c in base_adj[b]:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    shared = True
            elif b == c:
                if d in base_adj[a]:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    shared = True
            elif b == d:
                if c in base_adj[a]:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    shared = True
            
            # Also check: does edge i create triangle with edge j via a base edge not sharing vertices?
            # Actually if they don't share a vertex, 2 edges can't form a triangle (need 3 vertices)
            # So only shared-vertex cases matter. But wait:
            # Triangle (a, c, w) where (a,c) is not one of our edges... no.
            # The triangle must use exactly the edges we're adding plus base.
            # With 2 added edges (a,b) and (c,d), a triangle needs 3 edges.
            # If a,b,c,d are all distinct, we can't form a triangle with just 2 of them.
            # If they share a vertex, say a=c, then edges are (a,b) and (a,d).
            # Triangle (a,b,d) needs (b,d) in base. ✓ (checked)
            # Any other triangle? (a,b,x) needs (a,x) and (b,x) - (a,x) could be (a,d) only if x=d, back to same.
            # So shared-vertex check is complete.
    
    # Filter out self-conflicting candidates
    valid_cand_indices = [i for i in range(len(candidates)) if i not in conflict[i]]
    
    # Now solve: find minimum subset S of valid_cand_indices such that:
    # 1. No two edges in S conflict (independent set in conflict graph)
    # 2. The resulting graph has diameter ≤ 4
    
    # For diameter checking, we need an efficient method.
    # Let's use BFS from each node.
    
    def check_diameter(added_indices):
        """Check if base + added edges has diameter ≤ 4."""
        adj = [set(s) for s in base_adj]
        for idx in added_indices:
            u, v = candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if -1 in dist:
                return False
            if max(dist) > 4:
                return False
        return True
    
    # Use branch and bound to find minimum independent set satisfying diameter constraint
    # Sort candidates by some heuristic (e.g., how much they reduce diameter)
    
    # Precompute for each candidate edge, which pairs (u,v) it helps connect
    # An edge (a,b) helps pair (u,v) if it creates a shorter path
    
    # For now, use a simpler approach: iterative deepening on number of edges
    # Try k=1,2,3,... and check if there's a valid subset of size k
    
    # First, let's try a smarter initialization: 
    # The path has diameter 23. We need to reduce it to ≤4.
    # Key insight: with diameter ≤4, every node must be within distance 2 of some "hub"
    # (since from any node to any other is ≤4, so to a central node is ≤2).
    
    # Let's try: for each potential hub h, find minimum edges to connect all nodes within dist 2 of h
    # But we also need inter-hub connectivity if multiple hubs.
    
    # Actually, let's just do a systematic search with pruning.
    
    # Precompute: for each subset of candidates (represented as bitmask), check validity
    # But 252 candidates is too many for bitmask. Let's limit to a promising subset.
    
    # Heuristic: only consider edges of length ≥ 3 (shorter edges are less useful for diameter)
    # Actually, even length-2 edges can be useful. Let's consider all valid ones.
    
    # Let's try a different approach: 
    # 1. Find all maximal independent sets in the conflict graph (or at least large ones)
    # 2. For each, check diameter and minimize
    
    # Actually, the conflict graph structure is simple: edges only conflict if they share a vertex
    # and the "other" endpoints are adjacent in base. Let's exploit this.
    
    # For a fixed vertex v, the edges incident to v that are "compatible" form a set.
    # Two edges (v,a) and (v,b) conflict iff (a,b) is a base edge.
    # In a path, (a,b) is a base edge iff |a-b| = 1.
    # So (v,a) and (v,b) conflict iff |a-b| = 1.
    # This means: for edges incident to v, we can pick a set where no two endpoints are adjacent in the path.
    # This is an independent set in a path graph! Easy to solve.
    
    # But edges can also be non-incident. Two non-incident edges never conflict.
    # So the conflict graph is a union of "stars" (one per vertex) with edges only within each star.
    # Wait, that's not quite right. Edge (a,b) is in the star of a AND in the star of b.
    # So the conflict graph has cliques at each vertex's incident edges, but edges belong to two cliques.
    
    # Let me just do a direct search with good pruning.
    
    # Start with a good initial solution using the hub strategy
    def hub_solution(hub):
        """Find edges to connect all nodes within dist 4 of hub, minimizing count."""
        # Nodes at distance 1,2,3,4 from hub are already connected via path.
        # Nodes at distance > 4 need direct or indirect connections.
        # With hub h, node v at distance d from h needs:
        #   - If d ≤ 4: already OK via path
        #   - If d > 4: need edge (h,v) or path through other added edges
        
        # Simple approach: connect hub to all nodes at distance > 4
        added = []
        for v in range(n):
            if v == hub:
                continue
            if abs(v - hub) > 4:
                e = (min(hub, v), max(hub, v))
                if e not in base_set:
                    # Check if this edge is valid (no self-conflict)
                    # Find its index
                    found = False
                    for idx in valid_cand_indices:
                        if candidates[idx] == e:
                            found = True
                            break
                    if found:
                        added.append((hub, v))
        
        # Check if this creates any conflicts
        # For edges from same hub: (hub, a) and (hub, b) conflict iff |a-b| = 1
        # In our set, a and b are both > 4 from hub, so |a-b| could be 1
        # Example: hub=11, a=16, b=17: |16-17|=1, conflict!
        
        # So we need to be more careful. Let's use a greedy approach per hub.
        added_set = set()
        result = []
        
        # Sort targets by distance, farthest first
        targets = []
        for v in range(n):
            if v != hub and abs(v - hub) > 4:
                targets.append(v)
        targets.sort(key=lambda v: abs(v - hub), reverse=True)
        
        for v in targets:
            e = (min(hub, v), max(hub, v))
            if e in base_set:
                continue
            # Check compatibility with existing added edges
            compatible = True
            for (hu, hv) in result:
                # Check if (hub, v) and (hu, hv) conflict
                if hub == hu:
                    # Same hub: conflict iff |v - hv| == 1
                    if abs(v - hv) == 1:
                        compatible = False
                        break
                elif hub == hv:
                    if abs(v - hu) == 1:
                        compatible = False
                        break
                elif v == hu:
                    if abs(hub - hv) == 1:
                        compatible = False
                        break
                elif v == hv:
                    if abs(hub - hu) == 1:
                        compatible = False
                        break
            if compatible:
                result.append((hub, v))
                added_set.add(e)
        
        return result
    
    # Try all hubs and pick best
    best_edges = None
    best_count = float('inf')
    
    for hub in range(n):
        edges = hub_solution(hub)
        # Check diameter
        if check_diameter([i for i in valid_cand_indices if candidates[i] in edges]):
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = edges
        
        # Also try greedy removal
        if edges:
            # Try removing each edge and check if still valid
            changed = True
            while changed:
                changed = False
                for i in range(len(edges)):
                    e = edges[i]
                    new_edges = edges[:i] + edges[i+1:]
                    if check_diameter([idx for idx in valid_cand_indices if candidates[idx] in new_edges]):
                        edges = new_edges
                        changed = True
                        break
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = edges
    
    # Now try to improve with local search: swap edges
    if best_edges is not None:
        current = list(best_edges)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                # Try removing edge i and adding a different one
                for j in range(len(candidates)):
                    if j not in valid_cand_indices:
                        continue
                    new_e = candidates[j]
                    if new_e in current:
                        continue
                    if new_e in base_set:
                        continue
                    test = current[:i] + current[i+1:] + [new_e]
                    # Check no conflicts
                    valid = True
                    for a in range(len(test)):
                        for b in range(a+1, len(test)):
                            ta = test[a]
                            tb = test[b]
                            # Check conflict
                            if ta[0] == tb[0] and abs(ta[1] - tb[1]) == 1:
                                valid = False
                                break
                            if ta[0] == tb[1] and abs(ta[1] - tb[0]) == 1:
                                valid = False
                                break
                            if ta[1] == tb[0] and abs(ta[0] - tb[1]) == 1:
                                valid = False
                                break
                            if ta[1] == tb[1] and abs(ta[0] - tb[0]) == 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        if check_diameter([idx for idx in valid_cand_indices if candidates[idx] in test]):
                            current = test
                            improved = True
                            break
                if improved:
                    break
            
            # Try removal
            for i in range(len(current)):
                test = current[:i] + current[i+1:]
                if check_diameter([idx for idx in valid_cand_indices if candidates[idx] in test]):
                    current = test
                    improved = True
                    break
        
        if len(current) < best_count:
            best_count = len(current)
            best_edges = current
    
    # Final answer
    if best_edges is None:
        # Fallback: hub at 11
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        best_edges = [[hub, v] for v in targets]
    
    return [[int(u), int(v)] for u, v in best_edges]
