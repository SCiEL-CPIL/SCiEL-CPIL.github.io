import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    N = 20
    
    def edge_key(u, v):
        return (u, v) if u < v else (v, u)
    
    # Precompute tree structures
    tree_info = []
    for i in range(20):
        k = i + 1
        edges = trees[i]
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'root_order': [0], 'parent': [-2]})
            continue
        if k == 2:
            tree_info.append({'k': 2, 'edges': [(0, 1)], 'adj': [[1], [0]], 'root_order': [0, 1], 'parent': [-2, 0]})
            continue
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v); adj[v].append(u)
        # BFS from 0
        parent = [-1] * k
        parent[0] = -2
        order = [0]
        q = deque([0])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if parent[y] == -1:
                    parent[y] = x
                    q.append(y)
                    order.append(y)
        tree_info.append({'k': k, 'edges': edges, 'adj': adj, 'root_order': order, 'parent': parent})
    
    def try_pack(seed_val, time_limit):
        rng = Random(seed_val)
        used = set()
        result = [None] * 20
        
        # Process trees in order of decreasing size for constraint propagation
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: -(i + 1))
        
        # Maintain free edge counts for each host
        free_deg = [N - 1] * N
        
        for idx in order_idx:
            info = tree_info[idx]
            k = info['k']
            
            if k == 1:
                # Single vertex, any unused host
                # We don't use any edges, so free_deg doesn't change
                # But we must mark the host as used for vertex assignment
                # However, 'used' set in original code was for edges.
                # We need a separate 'used_hosts' for vertices?
                # The problem says: "list of k distinct int host vertices".
                # So vertices must be distinct within a tree.
                # And trees are edge-disjoint.
                # Vertices CAN be shared between different trees?
                # "Embedded trees must be edge-disjoint in K20."
                # It does NOT say vertex-disjoint.
                # So a host vertex can be part of multiple trees, as long as the edges are distinct.
                # This changes the problem significantly!
                # My previous assumption of 'used_hosts' was WRONG.
                # Vertices are reusable. Only edges are unique.
                
                # Let's re-read carefully:
                # "Entry k-1 is None or a list of k distinct int host vertices in 0..19"
                # "position i is the image of vertex i of the fixed tree T_k"
                # "Embedded trees must be edge-disjoint in K20"
                # Yes, vertices can be reused.
                
                # So for k=1, we just pick any host.
                # To maximize flexibility, we might pick a host with high free_deg?
                # But k=1 uses no edges. So it doesn't affect free_deg.
                # We can just pick 0.
                result[idx] = [0]
                continue
            
            if k == 2:
                # Find any unused edge
                placed = False
                for a in range(N):
                    for b in range(a + 1, N):
                        e = (a, b)
                        if e not in used:
                            result[idx] = [a, b]
                            used.add(e)
                            # Update free_deg
                            free_deg[a] -= 1
                            free_deg[b] -= 1
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    return None
                continue
            
            # General tree: backtracking
            adj = info['adj']
            parent = info['parent']
            order = info['root_order']
            
            assignment = [-1] * k
            
            # We need to track which edges are used during backtracking
            # We can't just use the global 'used' set directly because we need to undo.
            # We will maintain a local list of used edges for this tree placement attempt.
            # Or better, we can just check against 'used' and the edges added in this DFS path.
            
            def dfs(idx_v):
                if time.time() > time_limit:
                    return False
                if idx_v == k:
                    return True
                
                v = order[idx_v]
                pv = parent[v]
                
                # Get candidate hosts
                candidates = []
                for h in range(N):
                    if pv >= 0:
                        ph = assignment[pv]
                        e = edge_key(h, ph)
                        if e in used:
                            continue
                        # Also check if this edge is used by a previously placed vertex in THIS tree
                        # We need to track edges used in current partial assignment
                        # This is tricky.
                        # Let's use a set `local_used_edges` passed by reference or closure.
                        # But modifying closure state in recursive DFS is messy.
                        
                        # Alternative:
                        # Since the tree is small, we can just check if the edge is in the global 'used' set.
                        # But we also need to ensure we don't use an edge twice WITHIN this tree.
                        # Since the tree is a tree, each edge is unique.
                        # If we map vertices to hosts, and the tree edges are distinct,
                        # the host edges will be distinct UNLESS two different tree edges map to the same host edge.
                        # This can happen if the mapping is not injective on edges?
                        # No, if (u,v) and (x,y) are distinct edges in the tree,
                        # can (H(u), H(v)) == (H(x), H(y))?
                        # Yes, if H(u)=H(x) and H(v)=H(y).
                        # But H must be injective on vertices? NO.
                        # "list of k distinct int host vertices" -> H is injective on the tree's vertices.
                        # So H(u) != H(v) for u!=v.
                        # If H is injective, then (H(u), H(v)) determines u and v.
                        # So distinct tree edges map to distinct host edges.
                        # So we only need to check against global 'used'.
                        
                        candidates.append(h)
                    else:
                        # Root
                        candidates.append(h)
                
                if not candidates:
                    return False
                
                # Sort candidates by free_deg (ascending)
                candidates.sort(key=lambda h: free_deg[h])
                
                for h in candidates:
                    assignment[v] = h
                    
                    # Check if this placement is valid regarding edges
                    # If pv >= 0, we are adding edge (h, ph)
                    # We already checked it's not in 'used'.
                    # Since H is injective, it won't conflict with other edges in this tree.
                    
                    if dfs(idx_v + 1):
                        return True
                    
                    assignment[v] = -1
                
                return False
            
            if dfs(0):
                result[idx] = [assignment[i] for i in range(k)]
                # Update global 'used' and 'free_deg'
                for (u, v) in info['edges']:
                    e = edge_key(assignment[u], assignment[v])
                    used.add(e)
                    free_deg[assignment[u]] -= 1
                    free_deg[assignment[v]] -= 1
            else:
                return None
        
        # Verify
        seen = set()
        for i in range(20):
            r = result[i]
            if r is None:
                if i == 0:
                    continue
                return None
            if len(set(r)) != len(r):
                return None
            for x in r:
                if not (0 <= x <= 19):
                    return None
            for (u, v) in trees[i]:
                e = edge_key(r[u], r[v])
                if e in seen:
                    return None
                seen.add(e)
        
        return result
    
    # Try multiple seeds with increasing time allocation
    best_result = None
    best_edges = 0
    for s in range(2000):
        elapsed = time.time() - (deadline - 145.0)
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Allocate more time to later attempts as we get closer to deadline
        time_per_attempt = min(remaining * 0.3, 2.0)
        res = try_pack(s, time.time() + time_per_attempt)
        if res is not None:
            # Count edges
            edge_count = 0
            for i in range(20):
                if res[i] is not None:
                    edge_count += len(trees[i])
            
            if edge_count > best_edges:
                best_edges = edge_count
                best_result = res
                
                # If we found a complete solution, return it
                if edge_count == 190:
                    out = []
                    for i in range(20):
                        r = res[i]
                        if r is None:
                            out.append(None)
                        else:
                            out.append([int(x) for x in r])
                    return out
    
    # Return best partial solution found
    if best_result is not None:
        out = []
        for i in range(20):
            r = best_result[i]
            if r is None:
                out.append(None)
            else:
                out.append([int(x) for x in r])
        return out
    
    # Fallback: return seed solution
    return [None] * 19 + [list(range(20))]
