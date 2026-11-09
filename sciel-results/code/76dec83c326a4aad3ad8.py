# EVOLVE-BLOCK-START
import time
import random
from random import Random
from collections import deque

_DEADLINE = 140.0  # safety margin before 150s kill

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
    tree_info = []
    for k in range(1, N + 1):
        edges = trees[k - 1]
        deg = [0] * k
        adj = [set() for _ in range(k)]
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        tree_info.append((k, edges, deg, adj))
    
    # Embed in decreasing order of size (hardest first)
    order = list(range(N - 1, -1, -1))
    
    # Host state
    used_edges = set()
    used_neighbors = [set() for _ in range(N)]
    
    result = [None] * N
    
    def avail_degree(v):
        return N - 1 - len(used_neighbors[v])
    
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
    
    def _find_centroid(k, adj, deg):
        """Find a good root: prefer a leaf that's far from the center."""
        if k <= 2:
            return 0
        leaves = [v for v in range(k) if deg[v] == 1]
        if not leaves:
            return 0
        # Pick leaf with minimum degree (all leaves have degree 1, so just pick first)
        # Better: pick a leaf that is "far" from high-degree nodes
        # Use BFS from each leaf to find the one with max eccentricity
        # For small k, just pick a random leaf
        return leaves[0]
    
    def _bfs_order(k, root, adj):
        """BFS ordering from root, returns (order, parent) arrays."""
        order = [root]
        parent = [-1] * k
        visited = [False] * k
        visited[root] = True
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for w in sorted(adj[u]):
                if not visited[w]:
                    visited[w] = True
                    parent[w] = u
                    queue.append(w)
                    order.append(w)
        return order, parent
    
    def embed_tree(k_idx, rng):
        """Try to embed tree k_idx (0-indexed, tree size k_idx+1). Returns True/False."""
        k = k_idx + 1
        edges, deg, adj = tree_info[k_idx][1], tree_info[k_idx][2], tree_info[k_idx][3]
        
        if k == 1:
            # Any host vertex works
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
        
        # BFS-based backtracking
        root = _find_centroid(k, adj, deg)
        bfs_order, parent = _bfs_order(k, root, adj)
        
        # For each vertex in BFS order, we know its parent
        # We'll assign hosts in BFS order
        mapping = [None] * k
        assigned = [False] * k
        used_host_set = set()
        
        # Precompute adjacency lists for quick access
        adj_list = [list(adj[v]) for v in range(k)]
        
        def forward_check(pos):
            """Check that each unassigned vertex has at least one valid host."""
            for i in range(pos, k):
                tv = bfs_order[i]
                p = parent[tv]
                
                # The parent must already be assigned (BFS order guarantees this)
                if p >= 0 and not assigned[p]:
                    return False
                
                d_needed = deg[tv]
                
                # Check if there's any valid host
                has_candidate = False
                for h in range(N):
                    if h in used_host_set:
                        continue
                    if avail_degree(h) < d_needed:
                        continue
                    # Check edge to parent
                    if p >= 0:
                        hp = mapping[p]
                        if not edge_free(h, hp):
                            continue
                    # Check edges to already-assigned children (shouldn't exist in BFS)
                    # In BFS, children come after parent, so only parent edge matters
                    has_candidate = True
                    break
                if not has_candidate:
                    return False
            return True
        
        def score_candidate(h, tv):
            """Score a candidate host: prefer hosts with fewer remaining options (scarcity)."""
            # Lower score = more preferred (more constrained)
            ad = avail_degree(h)
            # Also consider how many edges from h are already used
            used_cnt = len(used_neighbors[h])
            return ad + 0.1 * used_cnt
        
        def backtrack(pos):
            if time.time() - t0 > _DEADLINE:
                return False
            if pos == k:
                return True
            
            tv = bfs_order[pos]
            p = parent[tv]
            d_needed = deg[tv]
            
            candidates = []
            for h in range(N):
                if h in used_host_set:
                    continue
                if avail_degree(h) < d_needed:
                    continue
                if p >= 0:
                    hp = mapping[p]
                    if not edge_free(h, hp):
                        continue
                candidates.append(h)
            
            if not candidates:
                return False
            
            # Sort by scarcity score
            candidates.sort(key=lambda h: (score_candidate(h, tv), rng.random()))
            max_cands = min(len(candidates), 12)
            
            for h in candidates[:max_cands]:
                mapping[tv] = h
                assigned[tv] = True
                used_host_set.add(h)
                
                # Mark edge to parent
                if p >= 0:
                    mark_edge(h, mapping[p])
                
                if not forward_check(pos + 1):
                    if p >= 0:
                        unmark_edge(h, mapping[p])
                    mapping[tv] = None
                    assigned[tv] = False
                    used_host_set.discard(h)
                    continue
                
                if backtrack(pos + 1):
                    return True
                
                if p >= 0:
                    unmark_edge(h, mapping[p])
                
                mapping[tv] = None
                assigned[tv] = False
                used_host_set.discard(h)
            
            return False
        
        if backtrack(0):
            result[k_idx] = [int(mapping[i]) for i in range(k)]
            return True
        return False
    
    max_restarts = 50
    best_result = None
    best_count = 0
    
    for restart in range(max_restarts):
        if time.time() - t0 > _DEADLINE:
            break
        
        used_edges.clear()
        for i in range(N):
            used_neighbors[i].clear()
        result = [None] * N
        
        rng = random.Random(611 + restart * 7919)
        
        count = 0
        for k_idx in order:
            if time.time() - t0 > _DEADLINE:
                break
            if embed_tree(k_idx, rng):
                count += 1
            else:
                break
        
        # Polish pass: try to embed remaining trees in different order
        if count < N and time.time() - t0 < _DEADLINE - 5:
            remaining = [k_idx for k_idx in range(N) if result[k_idx] is None]
            # Try smallest first for polish
            remaining.sort(key=lambda x: x + 1)
            for k_idx in remaining:
                if time.time() - t0 > _DEADLINE:
                    break
                if embed_tree(k_idx, rng):
                    count += 1
        
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
