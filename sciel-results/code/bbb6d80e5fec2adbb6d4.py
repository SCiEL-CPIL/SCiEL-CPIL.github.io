import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Precompute all candidate edges (non-base edges)
    all_cands = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                all_cands.append((u, v))
    
    # Precompute triangle conflicts: for each edge e=(u,v), which other edges f create a triangle with e
    # A triangle is formed by edges (u,v), (v,w), (u,w) for some w
    # So edge (u,v) conflicts with (u,w) and (v,w) if both exist
    # Actually, adding edge e=(u,v) creates a triangle if there exists w such that
    # (u,w) and (v,w) are both in the current graph (base + added)
    
    # For efficiency, precompute for each edge e, the set of edges that "complete" a triangle
    # tri_conflict[e] = set of edges f such that adding both e and f (with some base edge) creates triangle
    # More precisely: if we add edge e=(u,v), it conflicts with any existing edge (u,w) or (v,w)
    # where the other edge to w also exists.
    
    # Let's think differently. For a set of added edges S, the graph is triangle-free iff
    # for any three vertices a,b,c, we don't have all three edges (a,b),(b,c),(c,a) in base∪S.
    
    # Key insight: base graph is a path, so it has no triangles. Any triangle in base∪S must
    # use at least 2 edges from S (since base has no triangles, and a triangle needs 3 edges).
    # Actually, a triangle could use 1 edge from S and 2 from base, or 2 from S and 1 from base, or 3 from S.
    # 
    # Case 1: 1 edge from S, 2 from base. E.g., S has (u,v), base has (u,w) and (w,v).
    #   This means w is adjacent to both u and v in the path, so u and v are at distance 2 in the path.
    # Case 2: 2 edges from S, 1 from base. E.g., S has (u,v) and (u,w), base has (v,w).
    #   This means v and w are adjacent in the path.
    # Case 3: 3 edges from S. E.g., S has (u,v), (v,w), (w,u).
    
    # For each candidate edge e=(u,v), let's precompute which other candidate edges conflict with it
    # when both are present (creating a triangle with a base edge or with each other).
    
    # Conflict type 1: e=(u,v) and base has (u,w) and (w,v) for some w
    #   i.e., v = u+2 (or u = v+2), and w = u+1 (or v+1)
    #   In this case, e alone doesn't create a triangle (need w to be connected to both)
    #   Actually if base has (u,w) and (w,v), then adding e=(u,v) creates triangle (u,v,w).
    #   So e is "forbidden" by itself if u and v are distance 2 in the path!
    
    # Let me re-examine. The base graph is a path. So for vertices u, v:
    # - If |u-v| == 1: they're adjacent in base. Edge (u,v) is a base edge, not a candidate.
    # - If |u-v| == 2: say u < v, so u, u+1, v. Base has (u,u+1) and (u+1,v).
    #   Adding (u,v) creates triangle (u, u+1, v). So this edge is INVALID to add!
    # - If |u-v| >= 3: no immediate triangle from base edges alone.
    
    # So first filter: only edges with |u-v| >= 3 are valid candidates.
    
    valid_cands = [(u, v) for u, v in all_cands if abs(u - v) >= 3]
    
    # Now for pairs of valid candidates, check if they create a triangle together.
    # Edges e1=(a,b) and e2=(c,d) create a triangle if there exist vertices x,y,z
    # such that {x,y,z} uses vertices from {a,b,c,d} and all three edges among them are in base∪{e1,e2}.
    
    # Since base is a path, let's enumerate:
    # A triangle needs 3 vertices. Let's say the triangle is (p, q, r).
    # The three edges are (p,q), (q,r), (r,p).
    # Each of these is either in base or in {e1, e2}.
    # Since base is a path, at most 2 of the 3 edges can be in base (a path has no triangles).
    # Actually, in a path, for 3 vertices p,q,r, the number of base edges among them is at most 2.
    # (If p<q<r, base has (p,p+1)...(q-1,q) and (q,q+1)...(r-1,r), but only consecutive ones.
    #  So base edges among {p,q,r} are: (p,q) if q=p+1, (q,r) if r=q+1, (p,r) if r=p+1.
    #  At most 2 of these can be true.)
    
    # So for a triangle with edges from {e1, e2} ∪ base:
    # - 0 or 1 edges from {e1,e2}: impossible (base alone has no triangles)
    # - 2 edges from {e1,e2}, 1 from base
    # - 3 edges from {e1,e2} (but e1 and e2 are only 2 edges, so impossible unless e1=e2)
    
    # So only case: 2 edges from {e1,e2} and 1 from base.
    # This means e1 and e2 share a vertex, and the third edge is in base.
    # E.g., e1=(a,b), e2=(a,c), base has (b,c).
    # Or e1=(a,b), e2=(b,c), base has (a,c).
    # Etc.
    
    # So e1=(a,b) and e2=(c,d) conflict if:
    # - They share exactly one vertex, say a=c, and (b,d) is in base (i.e., |b-d|==1)
    # - Or they share exactly one vertex, say a=d, and (b,c) is in base (i.e., |b-c|==1)
    # - Etc.
    
    # Let's precompute conflict pairs.
    cand_set = set((min(u,v), max(u,v)) for u,v in valid_cands)
    # Map each valid cand to its index
    cand_idx = {}
    for i, (u, v) in enumerate(valid_cands):
        cand_idx[(u, v)] = i
    
    num_cands = len(valid_cands)
    # conflicts[i] = set of j such that cand i and cand j create a triangle
    conflicts = [set() for _ in range(num_cands)]
    
    for i in range(num_cands):
        a, b = valid_cands[i]
        for j in range(i + 1, num_cands):
            c, d = valid_cands[j]
            # Check if they share a vertex
            if a == c or a == d or b == c or b == d:
                # They share a vertex. Find the other two endpoints.
                if a == c:
                    # Shared vertex a. Other endpoints b, d. Check if (b,d) in base.
                    if abs(b - d) == 1:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
                elif a == d:
                    if abs(b - c) == 1:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
                elif b == c:
                    if abs(a - d) == 1:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
                elif b == d:
                    if abs(a - c) == 1:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
    
    # Now we have the conflict graph. We need to find a subset S of candidates such that:
    # 1. No two edges in S are in conflict (triangle-free constraint)
    # 2. The diameter of base ∪ S is <= 4
    # 3. |S| is minimized
    
    # For diameter check, we need to verify that for all pairs (u,v), dist(u,v) <= 4 in base∪S.
    # In the base path, dist(u,v) = |u-v|. Adding an edge (a,b) can reduce distances.
    
    # Strategy: Use a branch-and-bound or greedy approach.
    # Given the structure, let's try a greedy approach with local search.
    
    def build_adj(added_indices):
        adj = [set(s) for s in base_adj]
        for idx in added_indices:
            u, v = valid_cands[idx]
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def bfs_dist(adj, src):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
        return dist
    
    def diameter_ok(adj):
        for src in range(n):
            dist = bfs_dist(adj, src)
            if -1 in dist:
                return False
        return True
    
    def check_diameter(added_indices):
        adj = build_adj(added_indices)
        return diameter_ok(adj)
    
    # Greedy construction
    def greedy_construct():
        added = set()
        # Sort candidates by "usefulness" - edges that connect distant nodes
        # For diameter reduction, edges that bridge large gaps are most useful.
        sorted_cands = sorted(range(num_cands), key=lambda i: abs(valid_cands[i][0] - valid_cands[i][1]), reverse=True)
        
        for idx in sorted_cands:
            if time.time() > deadline:
                break
            # Check if idx conflicts with any already-added edge
            if any(j in added for j in conflicts[idx]):
                continue
            # Tentatively add
            new_added = added | {idx}
            if check_diameter(new_added):
                added = new_added
                # Check if diameter is now <= 4
                adj = build_adj(added)
                diam = 0
                valid = True
                for src in range(n):
                    dist = bfs_dist(adj, src)
                    if -1 in dist:
                        valid = False
                        break
                    diam = max(diam, max(dist))
                if valid and diam <= 4:
                    return list(added)
        return list(added)
    
    # Try to remove edges from a valid solution
    def try_remove(added_list):
        added = set(added_list)
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for idx in list(added):
                new_added = added - {idx}
                if check_diameter(new_added):
                    added = new_added
                    changed = True
                    break
        return list(added)
    
    # Local search with swaps
    def local_search(added_list, max_time=5):
        added = set(added_list)
        best = list(added)
        best_size = len(added)
        start_ls = time.time()
        
        # First, try removals
        added = set(try_remove(list(added)))
        best = list(added)
        best_size = len(added)
        
        # Then try swaps
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            for idx in list(added):
                if time.time() > deadline:
                    break
                added_without = added - {idx}
                for new_idx in range(num_cands):
                    if time.time() > deadline:
                        break
                    if new_idx in added_without or new_idx == idx:
                        continue
                    # Check conflicts
                    if any(j in added_without for j in conflicts[new_idx]):
                        continue
                    new_added = added_without | {new_idx}
                    if check_diameter(new_added):
                        # Same size, but might enable removal
                        trial = set(new_added)
                        # Try to remove
                        for rem_idx in list(trial):
                            if trial - {rem_idx} and check_diameter(trial - {rem_idx}):
                                trial = trial - {rem_idx}
                                break
                        if len(trial) < best_size:
                            added = trial
                            best = list(trial)
                            best_size = len(trial)
                            improved = True
                            break
                if improved:
                    break
        return best
    
    # Random restarts
    def random_greedy():
        added = set()
        cands = list(range(num_cands))
        random.shuffle(cands)
        
        for idx in cands:
            if time.time() > deadline:
                break
            if any(j in added for j in conflicts[idx]):
                continue
            new_added = added | {idx}
            if check_diameter(new_added):
                added = new_added
                adj = build_adj(added)
                diam = 0
                valid = True
                for src in range(n):
                    dist = bfs_dist(adj, src)
                    if -1 in dist:
                        valid = False
                        break
                    diam = max(diam, max(dist))
                if valid and diam <= 4:
                    return list(added)
        return list(added)
    
    # Main search
    best_edges = []
    best_count = float('inf')
    
    # Strategy 1: Greedy
    if time.time() < deadline - 10:
        greedy_result = greedy_construct()
        if greedy_result:
            cleaned = try_remove(greedy_result)
            if len(cleaned) < best_count and check_diameter(cleaned):
                best_count = len(cleaned)
                best_edges = cleaned
    
    # Strategy 2: Random restarts with local search
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        init = random_greedy()
        if init and check_diameter(init):
            ls_result = local_search(init, max_time=2)
            if ls_result and len(ls_result) < best_count and check_diameter(ls_result):
                best_count = len(ls_result)
                best_edges = ls_result
    
    # Fallback
    if not best_edges:
        # Hub-based solution
        # Pick a central hub, connect to nodes at distance > 4
        # But need to ensure triangle-free
        # For a path, if hub = h, connect to v where |h-v| >= 3 (to avoid immediate triangles)
        # And check that no two connected nodes are adjacent (which would create triangle with hub)
        # Actually, if hub connects to v1 and v2, triangle if |v1-v2|==1.
        # So we need to pick a set of nodes no two of which are adjacent.
        
        # Let's try hub = 11
        hub = 11
        targets = []
        for v in range(n):
            if v == hub:
                continue
            if abs(v - hub) >= 3:  # Avoid immediate triangle
                # Check if v is adjacent to any already-picked target
                ok = True
                for t in targets:
                    if abs(v - t) == 1:
                        ok = False
                        break
                if ok:
                    targets.append(v)
        
        # Check if this gives diameter <= 4
        added_set = set()
        for v in targets:
            e = (min(hub, v), max(hub, v))
            if e in cand_set:
                added_set.add(cand_idx[e])
        
        if check_diameter(added_set):
            best_edges = list(added_set)
            best_count = len(added_set)
        else:
            # Try multiple hubs
            for hub in range(4, 20):
                if time.time() > deadline:
                    break
                targets = []
                for v in range(n):
                    if v == hub:
                        continue
                    if abs(v - hub) >= 3:
                        ok = True
                        for t in targets:
                            if abs(v - t) == 1:
                                ok = False
                                break
                        if ok:
                            targets.append(v)
                
                added_set = set()
                for v in targets:
                    e = (min(hub, v), max(hub, v))
                    if e in cand_set:
                        added_set.add(cand_idx[e])
                
                if check_diameter(added_set):
                    cleaned = try_remove(list(added_set))
                    if len(cleaned) < best_count and check_diameter(cleaned):
                        best_count = len(cleaned)
                        best_edges = cleaned
                    break
    
    # Convert indices back to edges
    result = []
    for idx in best_edges:
        u, v = valid_cands[idx]
        result.append([int(u), int(v)])
    
    # Final validation
    if result:
        added_set = set((min(u, v), max(u, v)) for u, v in result)
        adj = build_adj([cand_idx[e] for e in added_set if e in cand_idx])
        # Check triangle-free
        tri_free = True
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[u]:
                        if w > v and w in adj[v]:
                            tri_free = False
                            break
                    if not tri_free:
                        break
            if not tri_free:
                break
        if not tri_free or not diameter_ok(adj):
            # Use a known valid fallback
            # Connect 0-12, 1-13, 2-14, 3-15, 4-16, 5-17, 6-18, 7-19, 8-20, 9-21, 10-22
            # Check if triangle-free: edges (i, i+12) for i=0..10
            # Triangle would need 3 vertices with all pairwise edges. 
            # (i, i+12) and (j, j+12) share no vertex unless i=j or i+12=j or j+12=i.
            # If i+12=j, then j=i+12, and edge (j, j+12)=(i+12, i+24) invalid.
            # So no shared vertices between different edges. Triangle-free.
            # Diameter: dist(0, 23) = min(23, 1+12+10+... ) 
            # 0-12 (added), 12-13 (base), 13-14 (base), ... 13 to 23 is 10. So 0-12-13-...-23 = 1+10=11. Still > 4.
            # This won't work.
            
            # Let's use a proper hub solution
            # The key is to find a valid solution.
            # For path P24, diameter 4 requires a specific structure.
            # Let's just return a valid solution from our search.
            # If we found nothing, there might be an issue.
            # Let's try a simple valid construction.
            
            # Actually, let's just use the first valid solution we can find.
            # Try hub = 11 with careful target selection.
            hub = 11
            targets = [0, 3, 6, 9, 15, 18, 21, 23]
            result = []
            for v in targets:
                e = (min(hub, v), max(hub, v))
                if e not in base_set and abs(hub - v) >= 3:
                    result.append([hub, v])
            
            # Validate
            added_set = set((min(u, v), max(u, v)) for u, v in result)
            adj = [set(s) for s in base_adj]
            for u, v in added_set:
                adj[u].add(v)
                adj[v].add(u)
            
            # Check triangle-free
            tri_free = True
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        for w in adj[u]:
                            if w > v and w in adj[v]:
                                tri_free = False
                                break
                        if not tri_free:
                            break
                if not tri_free:
                    break
            
            # Check diameter
            diam_ok = diameter_ok(adj)
            
            if tri_free and diam_ok:
                return result
            else:
                # Last resort: return empty (invalid, but better than crashing)
                # Actually, we should return something valid.
                # Let's try a different hub.
                for hub in [10, 11, 12, 13]:
                    targets = []
                    for v in range(n):
                        if v == hub:
                            continue
                        if abs(v - hub) >= 3:
                            ok = True
                            for t in targets:
                                if abs(v - t) == 1:
                                    ok = False
                                    break
                            if ok:
                                targets.append(v)
                    
                    result = []
                    for v in targets:
                        e = (min(hub, v), max(hub, v))
                        if e not in base_set:
                            result.append([hub, v])
                    
                    added_set = set((min(u, v), max(u, v)) for u, v in result)
                    adj = [set(s) for s in base_adj]
                    for u, v in added_set:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    tri_free = True
                    for u in range(n):
                        for v in adj[u]:
                            if v > u:
                                for w in adj[u]:
                                    if w > v and w in adj[v]:
                                        tri_free = False
                                        break
                                if not tri_free:
                                    break
                        if not tri_free:
                            break
                    
                    diam_ok = diameter_ok(adj)
                    
                    if tri_free and diam_ok:
                        return result
            
            # If all else fails, return empty
            return []
    
    return result


if __name__ == '__main__':
    result = construct()
    print(f"Added edges: {result}")
    print(f"Count: {len(result)}")
