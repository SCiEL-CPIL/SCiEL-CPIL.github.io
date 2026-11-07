"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def instance_trees(n=20, seed=611):
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

def hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp maximum bipartite matching.
    adj: list of lists, adj[u] = list of v in right side
    Returns (matching_right, matching_left) where matching_right[v] = u, matching_left[u] = v
    """
    INF = float('inf')
    pair_u = [-1] * n_left
    pair_v = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return pair_v, pair_u

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Total edges in K20
    total_k20_edges = n * (n - 1) // 2  # 190
    
    # Precompute all edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # We'll embed trees one by one, using a greedy + backtracking approach
    # For each tree, we try to find an embedding that uses available edges
    
    # State: which edges of K20 are used
    used_edges = set()
    
    # Result: list of embeddings, entry k-1 is None or list of k host vertices
    result = [None] * n
    
    # T1 is trivial (no edges)
    result[0] = [0]  # single vertex
    
    # For each tree from T2 to T20, try to embed
    # We use a recursive backtracking with pruning
    
    def try_embed_tree(tree_idx, tree_edges, current_used, depth=0):
        """
        Try to embed tree tree_idx into K20 avoiding current_used edges.
        Returns a list of n host vertices (the embedding) or None.
        """
        if time.time() > deadline:
            return None
        
        k = tree_idx + 1  # tree has k vertices
        if k == 1:
            return [0]
        
        # For a tree with k vertices, we need to assign k distinct host vertices
        # such that each tree edge maps to a distinct unused K20 edge
        
        # Strategy: build the embedding incrementally
        # Start with vertex 0 of the tree, try all host vertices for it
        # Then BFS/DFS to extend the embedding
        
        # First, let's build the tree adjacency
        tree_adj = defaultdict(list)
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Find a root (vertex 0)
        root = 0
        
        # We'll do a DFS to build the embedding
        # state: tree_vertex -> host_vertex mapping, and the set of used edges so far
        
        # To make this tractable, we'll use a heuristic:
        # Try to embed using a spanning tree of the tree (it's already a tree)
        # Use a greedy approach with backtracking
        
        best_embedding = None
        
        # Try different starting host vertices for the root
        for start_host in range(n):
            if time.time() > deadline:
                return None
            
            embedding = [-1] * k  # tree vertex -> host vertex
            embedding[root] = start_host
            used_in_embed = set()  # edges used in this embedding
            valid = True
            
            # BFS from root
            queue = deque([root])
            visited = {root}
            
            while queue:
                if time.time() > deadline:
                    return None
                
                tv = queue.popleft()
                for neighbor in tree_adj[tv]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    
                    # Try to assign a host vertex to neighbor
                    host_tv = embedding[tv]
                    found = False
                    
                    for host_neighbor in range(n):
                        if host_neighbor == host_tv:
                            continue
                        if any(embedding[vt] == host_neighbor for vt in range(k) if embedding[vt] != -1):
                            continue
                        
                        edge = (min(host_tv, host_neighbor), max(host_tv, host_neighbor))
                        if edge in current_used or edge in used_in_embed:
                            continue
                        
                        # Check if this assignment is consistent
                        embedding[neighbor] = host_neighbor
                        used_in_embed.add(edge)
                        queue.append(neighbor)
                        found = True
                        break
                    
                    if not found:
                        # Backtracking would be needed here, but for simplicity
                        # we'll mark this attempt as failed
                        valid = False
                        break
            
            if valid and len(visited) == k:
                # Check that all tree edges are mapped to valid, unused edges
                all_ok = True
                for u, v in tree_edges:
                    hu, hv = embedding[u], embedding[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in current_used:
                        all_ok = False
                        break
                if all_ok:
                    best_embedding = embedding[:]
                    break
        
        return best_embedding
    
    # Greedy embedding with some backtracking
    for tree_idx in range(1, n):
        if time.time() > deadline:
            break
        
        tree_edges = trees[tree_idx]
        k = tree_idx + 1
        
        if k == 1:
            result[tree_idx] = [0]
            continue
        
        # Try to embed this tree
        embedding = try_embed_tree(tree_idx, tree_edges, used_edges)
        
        if embedding is not None:
            result[tree_idx] = embedding
            # Update used edges
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
        else:
            # Couldn't embed this tree
            result[tree_idx] = None
    
    # Count total embedded edges
    total_embedded = 0
    for i in range(n):
        if result[i] is not None and len(result[i]) > 1:
            total_embedded += len(trees[i])
    
    return result
# EVOLVE-BLOCK-END
