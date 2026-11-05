"""Rewritten program: longest path via structural decomposition (Tree/Unicyclic) and DP for small components."""
import random
from collections import defaultdict, deque

# Build the pinned graph once at import time (deterministic, matches the seed graph)
def _build_graph():
    n = 100
    rng = random.Random(748)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 1.8 / 100:
                edges.add((i, j))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

N, ADJ = _build_graph()

def _connected_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

COMPS = _connected_components()

def _longest_path_in_component(comp):
    n = len(comp)
    if n == 0:
        return []
    if n == 1:
        return [comp[0]]
    
    # Map local indices
    idx = {v: i for i, v in enumerate(comp)}
    local_adj = [set() for _ in range(n)]
    edge_count = 0
    for u in comp:
        for v in ADJ[u]:
            if v in idx:
                local_adj[idx[u]].add(idx[v])
                edge_count += 1
    edge_count //= 2

    # 1. Small component: Use DP
    if n <= 18:
        size = 1 << n
        dp = [bytearray(n) for _ in range(size)]
        for i in range(n):
            dp[1 << i][i] = 1
        
        best_len = 1
        best_mask = 1
        best_last = 0
        
        for mask in range(size):
            for last in range(n):
                if not dp[mask][last]:
                    continue
                for nxt in local_adj[last]:
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    dp[new_mask][nxt] = 1
                    if new_mask.bit_count() > best_len:
                        best_len = new_mask.bit_count()
                        best_mask = new_mask
                        best_last = nxt
        
        # Reconstruct
        path = [best_last]
        mask = best_mask
        last = best_last
        while mask.bit_count() > 1:
            prev_mask = mask ^ (1 << last)
            for prev in range(n):
                if prev_mask & (1 << prev) and dp[prev_mask][prev] and prev in local_adj[last]:
                    path.append(prev)
                    mask = prev_mask
                    last = prev
                    break
            else:
                break
        path.reverse()
        return [comp[i] for i in path]

    # 2. Tree: E = N - 1
    if edge_count == n - 1:
        # Longest path is the diameter
        # Find diameter by BFS from an arbitrary node, then BFS from the farthest node
        def bfs(start):
            dist = [-1] * n
            q = deque([start])
            dist[start] = 0
            farthest = start
            max_dist = 0
            while q:
                u = q.popleft()
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in local_adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            return farthest, dist

        # Get a farthest node from 0
        u, _ = bfs(0)
        # Get farthest from u
        v, dist = bfs(u)
        
        # Reconstruct path from u to v
        # We need the actual path, not just length.
        # BFS from u to get parents
        parent = [-1] * n
        q = deque([u])
        parent[u] = -2
        while q:
            curr = q.popleft()
            if curr == v:
                break
            for nei in local_adj[curr]:
                if parent[nei] == -1:
                    parent[nei] = curr
                    q.append(nei)
        
        path = []
        curr = v
        while curr != -2:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return [comp[i] for i in path]

    # 3. Unicyclic: E = N
    if edge_count == n:
        # Find the cycle using topological sort (remove leaves)
        degree = [len(local_adj[i]) for i in range(n)]
        q = deque([i for i in range(n) if degree[i] == 1])
        in_cycle = [True] * n
        while q:
            u = q.popleft()
            in_cycle[u] = False
            for v in local_adj[u]:
                if in_cycle[v]:
                    degree[v] -= 1
                    if degree[v] == 1:
                        q.append(v)
        
        cycle_nodes = [i for i in range(n) if in_cycle[i]]
        if not cycle_nodes:
            # Should not happen if E=N and connected, but fallback
            return [comp[0]]
            
        # Map cycle nodes to 0..L-1
        L = len(cycle_nodes)
        cycle_idx = {v: i for i, v in enumerate(cycle_nodes)}
        
        # Compute heights and diameters for trees attached to cycle nodes
        # For each cycle node, we have a tree attached (excluding cycle edges)
        # We need to find the height (max dist to leaf) and diameter of this tree
        heights = [0] * L
        diameters = [0] * L
        
        for i, c in enumerate(cycle_nodes):
            # BFS/DFS from c, excluding other cycle nodes
            # We treat c as root
            # To find height and diameter of the tree rooted at c (excluding cycle edges)
            # We can do a DFS
            # But careful: the tree is attached at c.
            # We need the max distance from c to any leaf in this tree (height)
            # And the diameter of this tree (which might be between two leaves in this tree)
            
            # Let's use a DFS that returns (max_depth, diameter)
            def dfs_tree(u, parent):
                max_depth = 0
                diameter = 0
                second_max = 0
                for v in local_adj[u]:
                    if v == parent or in_cycle[v]:
                        continue
                    d, diam_child = dfs_tree(v, u)
                    d += 1
                    if d > max_depth:
                        second_max = max_depth
                        max_depth = d
                    elif d > second_max:
                        second_max = d
                    diameter = max(diameter, diam_child)
                # Diameter could be through u: max_depth + second_max
                diameter = max(diameter, max_depth + second_max)
                return max_depth, diameter

            h, d = dfs_tree(c, -1)
            heights[i] = h
            diameters[i] = d
            
        # Candidate 1: Max diameter of any attached tree
        best_len = max(diameters) if diameters else 0
        best_path_local = None
        
        # We need to reconstruct the path for the best candidate.
        # This is complex. Let's just find the length first, then reconstruct if it's the global max?
        # Or reconstruct all candidates and pick the longest.
        
        # For simplicity, let's identify the best pair (u, v) on the cycle.
        # Path length = heights[u] + heights[v] + dist_cycle(u, v)
        # Note: if u == v, dist is 0, path is 2*heights[u]? No, a path cannot go up and down the same branch.
        # If u == v, the path is just within the branch? No, if it enters u and leaves u, it's a cycle or backtracking.
        # A simple path entering u from branch and leaving to cycle must use distinct neighbors.
        # If it stays in the branch, it's handled by diameters.
        # If it goes through u to the cycle, it uses one edge from branch to u, and one edge from u to cycle.
        # So the length contributed by the branch is heights[u] (distance from leaf to u).
        # So for u != v, length = heights[u] + heights[v] + dist(u, v).
        # For u == v, we can't form a path that goes leaf->u->cycle->u->leaf because u is repeated.
        # So u != v.
        
        best_cycle_len = 0
        best_u, best_v = 0, 0
        
        for i in range(L):
            for j in range(i + 1, L):
                # dist on cycle
                # cycle is 0..L-1
                d1 = abs(i - j)
                d2 = L - d1
                dist_c = min(d1, d2)
                cand = heights[i] + heights[j] + dist_c
                if cand > best_cycle_len:
                    best_cycle_len = cand
                    best_u, best_v = i, j
                    
        # Compare with max diameter of branches
        # We need to reconstruct the path for the winner.
        
        # To avoid complex reconstruction logic for both cases, let's just generate the paths.
        
        # Path from Candidate 1 (Branch Diameter)
        # We need to find which branch had the max diameter and reconstruct it.
        # This requires storing parent pointers or similar during DFS.
        # Let's do a simpler approach: 
        # Since n is small for unicyclic components (likely < 30), we can just run the DFS/DP from the original code on this component?
        # No, that's what we're trying to avoid if it's large.
        # But unicyclic components in G(100, 0.018) are likely small.
        # Let's check size. If n <= 25, we can just use the DP from case 1?
        # DP for 25 is $2^{25} \approx 33$ million states. Too slow in Python.
        
        # Let's stick to the structural reconstruction.
        
        # Reconstruct path for best_cycle_len
        if best_cycle_len > best_len:
            # Path: Leaf in branch u -> ... -> u -> cycle -> v -> ... -> Leaf in branch v
            # We need the specific leaf.
            # Let's find the leaf for u and v.
            
            def get_leaf(u, parent):
                # Returns the farthest leaf from u in the tree attached to u (excluding cycle)
                # And the path to it
                best_path = []
                def dfs_path(curr, par, path):
                    nonlocal best_path
                    if len(path) > len(best_path):
                        best_path = path[:]
                    for v in local_adj[curr]:
                        if v == par or in_cycle[v]:
                            continue
                        dfs_path(v, curr, path + [v])
                dfs_path(u, -1, [u])
                return best_path

            path_u = get_leaf(cycle_nodes[best_u], -1)
            path_v = get_leaf(cycle_nodes[best_v], -1)
            
            # path_u is [u, ..., leaf_u] or [leaf_u, ..., u]?
            # My dfs_path starts with u, so it's [u, ..., leaf].
            # We want leaf -> ... -> u. So reverse it.
            path_u.reverse() # Now [leaf_u, ..., u]
            path_v.reverse() # Now [leaf_v, ..., v]
            
            # Cycle path from u to v
            # Determine direction on cycle
            # We want the shorter path? No, we calculated dist_c as min.
            # So we take the shorter arc.
            i, j = best_u, best_v
            d1 = abs(i - j)
            d2 = L - d1
            if d1 <= d2:
                # Go i -> i+1 -> ... -> j
                if i < j:
                    cycle_path = cycle_nodes[i:j+1]
                else:
                    cycle_path = cycle_nodes[i:] + cycle_nodes[:j+1]
            else:
                # Go i -> i-1 -> ... -> j
                if i > j:
                    cycle_path = cycle_nodes[j:i+1][::-1]
                else:
                    cycle_path = cycle_nodes[j:] + cycle_nodes[:i+1][::-1]
                    
            # Combine: path_u (ends at u) + cycle_path (starts at u, ends at v) + path_v (starts at v)
            # path_u ends at u. cycle_path starts at u.
            # We need to avoid duplicating u.
            # path_u = [leaf_u, ..., u]
            # cycle_path = [u, ..., v]
            # path_v = [v, ..., leaf_v]
            
            full_path = path_u + cycle_path[1:] + path_v[1:]
            
            # Verify length
            # len(full_path) should be best_cycle_len + 1?
            # heights[u] is dist(leaf, u). So path_u length is heights[u] + 1.
            # dist_c is number of edges. cycle_path length is dist_c + 1.
            # heights[v] is dist(leaf, v). path_v length is heights[v] + 1.
            # Total vertices = (h_u + 1) + (d_c + 1 - 1) + (h_v + 1 - 1) = h_u + h_v + d_c + 1.
            # My best_cycle_len was h_u + h_v + d_c.
            # So len(full_path) = best_cycle_len + 1.
            
            return [comp[i] for i in full_path]
        else:
            # Best is in a branch
            # Find which branch has max diameter
            max_diam = max(diameters)
            best_branch_idx = diameters.index(max_diam)
            c = cycle_nodes[best_branch_idx]
            
            # Reconstruct diameter of tree attached to c
            # We need to find the two endpoints of the diameter in that tree.
            # We can run BFS from c to find farthest, then BFS from farthest to find diameter path.
            # But we must exclude cycle neighbors.
            
            def bfs_tree(start, blocked):
                dist = [-1] * n
                parent = [-1] * n
                q = deque([start])
                dist[start] = 0
                parent[start] = -2
                farthest = start
                while q:
                    u = q.popleft()
                    if dist[u] > dist[farthest]:
                        farthest = u
                    for v in local_adj[u]:
                        if v in blocked or dist[v] != -1:
                            continue
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                return farthest, dist, parent

            blocked = set(cycle_nodes)
            # Remove c from blocked for the first BFS? No, we start at c.
            # We want to explore the tree attached to c.
            # The "blocked" set should contain all cycle nodes EXCEPT c?
            # Actually, we just don't traverse into other cycle nodes.
            
            # Find farthest from c
            u, dist1, parent1 = bfs_tree(c, set(cycle_nodes) - {c})
            # Find farthest from u
            v, dist2, parent2 = bfs_tree(u, set(cycle_nodes) - {c})
            
            # Reconstruct path from u to v using parent2
            path = []
            curr = v
            while curr != -2:
                path.append(curr)
                curr = parent2[curr]
            path.reverse()
            
            return [comp[i] for i in path]

    # 4. Complex (E > N) or other
    # Fallback to DFS with pruning
    best_path = [comp[0]]
    for start in comp:
        start_idx = idx[start]
        path = [start_idx]
        used = {start_idx}
        
        def dfs(current, path, used):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
            remaining = n - len(used)
            if len(path) + remaining <= len(best_path):
                return
            for nxt in local_adj[current]:
                if nxt not in used:
                    used.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, used)
                    path.pop()
                    used.discard(nxt)
        
        dfs(start_idx, path, used)
        
    return [comp[i] for i in best_path]

# Compute longest path in each component
component_paths = []
for comp in COMPS:
    lp = _longest_path_in_component(comp)
    if lp:
        component_paths.append(lp)

# The overall longest path is the longest among all components
if component_paths:
    BEST_PATH = max(component_paths, key=len)
else:
    BEST_PATH = [0]

def construct():
    return list(BEST_PATH)
