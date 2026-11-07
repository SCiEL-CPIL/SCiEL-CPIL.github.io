"""Packing trees T1..T20 into K20."""
import time
from random import Random
from collections import deque

# Reproduce the instance trees exactly as in the task description
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

# Precompute the trees once
_TREES = instance_trees()

def construct():
    """Return 20 entries; entry k-1 is None or a list of k distinct ints in 0..19."""
    deadline = time.time() + 110.0  # safety margin
    
    # We need to embed trees T1..T20 into K20 with edge-disjoint embeddings.
    # T1 is a single vertex (no edges), T2 is a single edge, etc.
    # Strategy: embed trees from largest to smallest (T20 down to T1)
    # using randomized greedy with backtracking.
    
    n = 20
    # Track which edges of K20 are used
    used_edges = set()
    # For each tree k (1-indexed), store the embedding (list of k host vertices)
    embeddings = [None] * n
    
    # Order: embed larger trees first
    order = list(range(19, -1, -1))  # T20, T19, ..., T1 (0-indexed)
    
    def get_tree_edges(tree_idx):
        """Return the edge list of tree at index tree_idx (0-based)."""
        return _TREES[tree_idx]
    
    def can_embed(tree_idx, host_verts, used_edges_set):
        """Check if mapping tree vertices via host_verts uses only unused edges."""
        tree_edges = get_tree_edges(tree_idx)
        for (a, b) in tree_edges:
            ha, hb = host_verts[a], host_verts[b]
            e = (min(ha, hb), max(ha, hb))
            if e in used_edges_set:
                return False
        return True
    
    def embed_tree(tree_idx, used_edges_set):
        """
        Try to embed tree tree_idx (0-based) into K20 using unused edges.
        Returns a list of host vertices (length = tree_idx+1) or None.
        Uses randomized greedy with some backtracking.
        """
        k = tree_idx + 1  # number of vertices in this tree
        tree_edges = get_tree_edges(tree_idx)
        
        if k == 1:
            # Single vertex, no edges. Just pick any vertex.
            # Prefer a vertex with high residual degree for flexibility
            return [0]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find a root: use a BFS/DFS ordering
        # Root at vertex 0
        parent = [-1] * k
        order_list = [0]
        visited = [False] * k
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    order_list.append(v)
        
        # Greedy assignment with randomization
        # Assign vertices in BFS order
        host_verts = [None] * k
        host_verts[0] = 0  # Start at vertex 0
        
        # For each vertex in BFS order (after root), assign a host vertex
        # such that the edge to parent is unused
        success = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            host_verts = [None] * k
            host_verts[0] = 0
            ok = True
            for u in order_list[1:]:
                pu = parent[u]
                hp = host_verts[pu]
                # Find host vertices for u such that edge (hp, hu) is unused
                candidates = []
                for v in range(n):
                    if v == hp:
                        continue
                    e = (min(hp, v), max(hp, v))
                    if e not in used_edges_set:
                        # Also check that v isn't already assigned to another tree vertex
                        # Actually, host vertices must be distinct within this tree
                        if v not in host_verts[:k]:  # this is O(k^2), let's use a set
                            pass
                        # Better: track used host vertices
                        pass
                    candidates.append(v)
                
                # Use a set for already-assigned host vertices
                assigned_hosts = set(host_verts[i] for i in range(k) if host_verts[i] is not None)
                candidates = []
                for v in range(n):
                    if v in assigned_hosts:
                        continue
                    e = (min(hp, v), max(hp, v))
                    if e not in used_edges_set:
                        candidates.append(v)
                
                if not candidates:
                    ok = False
                    break
                # Pick randomly with some bias toward high-degree vertices
                host_verts[u] = candidates[attempt % len(candidates)]
            
            if ok:
                # Verify all edges
                if can_embed(tree_idx, host_verts, used_edges_set):
                    success = True
                    break
            else:
                # Try again with different randomization
                pass
        
        if success:
            return host_verts
        return None
    
    # Better approach: use proper randomized search with backtracking
    def embed_tree_v2(tree_idx, used_edges_set, rng):
        """Embed tree tree_idx using randomized DFS with backtracking."""
        k = tree_idx + 1
        tree_edges = get_tree_edges(tree_idx)
        
        if k == 1:
            return [0]
        
        # Build tree structure
        adj = [[] for _ in range(k)]
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Find root and BFS order
        parent = [-1] * k
        bfs_order = [0]
        visited = [False] * k
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    bfs_order.append(v)
        
        # Assign in BFS order using backtracking
        host_verts = [None] * k
        assigned = [False] * n  # host vertex already used
        
        def backtrack(idx):
            if idx == len(bfs_order):
                # All assigned, verify
                for (a, b) in tree_edges:
                    ha, hb = host_verts[a], host_verts[b]
                    e = (min(ha, hb), max(ha, hb))
                    if e in used_edges_set:
                        return False
                return True
            
            u = bfs_order[idx]
            if idx == 0:
                # Root: try all vertices
                for v in range(n):
                    host_verts[u] = v
                    assigned[v] = True
                    if backtrack(idx + 1):
                        return True
                    assigned[v] = False
                    host_verts[u] = None
            else:
                pu = parent[u]
                hp = host_verts[pu]
                # Try vertices connected to hp via unused edge
                for v in range(n):
                    if assigned[v]:
                        continue
                    e = (min(hp, v), max(hp, v))
                    if e in used_edges_set:
                        continue
                    host_verts[u] = v
                    assigned[v] = True
                    if backtrack(idx + 1):
                        return True
                    assigned[v] = False
                    host_verts[u] = None
            return False
        
        # Use randomized order for better performance
        # Shuffle the candidate lists at each step
        def backtrack_random(idx, rng):
            if idx == len(bfs_order):
                for (a, b) in tree_edges:
                    ha, hb = host_verts[a], host_verts[b]
                    e = (min(ha, hb), max(ha, hb))
                    if e in used_edges_set:
                        return False
                return True
            
            u = bfs_order[idx]
            if idx == 0:
                candidates = list(range(n))
                rng.shuffle(candidates)
                for v in candidates:
                    host_verts[u] = v
                    assigned[v] = True
                    if backtrack_random(idx + 1, rng):
                        return True
                    assigned[v] = False
                    host_verts[u] = None
            else:
                pu = parent[u]
                hp = host_verts[pu]
                candidates = []
                for v in range(n):
                    if assigned[v]:
                        continue
                    e = (min(hp, v), max(hp, v))
                    if e not in used_edges_set:
                        candidates.append(v)
                rng.shuffle(candidates)
                for v in candidates:
                    host_verts[u] = v
                    assigned[v] = True
                    if backtrack_random(idx + 1, rng):
                        return True
                    assigned[v] = False
                    host_verts[u] = None
            return False
        
        # Try multiple random restarts
        for seed in range(200):
            rng = Random(seed + tree_idx * 1000)
            # Reset
            host_verts = [None] * k
            assigned = [False] * n
            if backtrack_random(0, rng):
                return host_verts
            if time.time() > deadline:
                break
        
        return None
    
    # Main loop: embed trees from largest to smallest
    for tree_idx in order:
        if time.time() > deadline:
            break
        result = embed_tree_v2(tree_idx, used_edges, Random(tree_idx))
        if result is None:
            # Try with different random seeds
            for s in range(100):
                result = embed_tree_v2(tree_idx, used_edges, Random(s + tree_idx * 7))
                if result is not None:
                    break
                if time.time() > deadline:
                    break
        if result is not None:
            embeddings[tree_idx] = result
            # Mark edges as used
            for (a, b) in get_tree_edges(tree_idx):
                ha, hb = result[a], result[b]
                e = (min(ha, hb), max(ha, hb))
                used_edges.add(e)
        else:
            # Could not embed this tree; skip it (leave as None)
            embeddings[tree_idx] = None
    
    # Convert to proper Python int lists
    final = []
    for k in range(n):
        if embeddings[k] is None:
            final.append(None)
        else:
            final.append([int(v) for v in embeddings[k]])
    
    return final
