"""Packing trees T1..T20 into K20 via backtracking with forward checking."""
import time
from random import Random

# Generate the fixed trees exactly as specified
def _instance_trees(n=20, seed=611):
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

def _build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for v in adj:
        adj[v] = sorted(adj[v])
    return adj

def _embed_tree(tree_edges, n, used_edges_set, deadline):
    """
    Find an embedding of the tree into K_n such that no edge conflicts with used_edges_set.
    Returns a list mapping tree-vertex -> host-vertex, or None if not found.
    """
    if not tree_edges:
        # Single vertex tree (k=1): any free vertex works
        for v in range(n):
            return [v]
        return None

    k = len(tree_edges) + 1  # number of vertices in tree
    adj = _build_adjacency(tree_edges)
    
    # Order vertices by degree (most constrained first), then by index
    # For the root, we'll try all vertices; for others, use backtracking
    vertex_order = sorted(range(k), key=lambda v: (-len(adj.get(v, [])), v))
    
    # We need to find an injective mapping from tree vertices {0..k-1} to host vertices {0..n-1}
    # such that for every tree edge (u,v), (map[u], map[v]) is not in used_edges_set
    # and map is injective.
    
    # Strategy: pick a root, then do DFS/backtracking assigning vertices in a good order
    # Use the tree structure: root the tree, then assign in BFS/DFS order
    
    # Find a root (vertex with highest degree, or 0)
    root = max(range(k), key=lambda v: len(adj.get(v, [])))
    
    # Build parent/children structure by BFS from root
    parent = [-1] * k
    children = [[] for _ in range(k)]
    order = []  # BFS order
    visited = {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for nb in adj.get(v, []):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = v
                children[v].append(nb)
                queue.append(nb)
    
    # Now do backtracking: assign host vertices in BFS order
    # map_tree[v] = host vertex for tree vertex v
    map_tree = [-1] * k
    used_hosts = set()
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        v = order[idx]
        # Determine constraints: if v has a parent, the edge (parent[v], v) must not be in used_edges
        p = parent[v]
        for h in range(n):
            if h in used_hosts:
                continue
            if p != -1:
                # Check edge (map_tree[p], h) is not in used_edges
                a, b = map_tree[p], h
                if (a, b) in used_edges_set or (b, a) in used_edges_set:
                    continue
            # Forward checking: verify that each child of v can still be assigned
            # (at least one free host exists that doesn't conflict with the edge to child)
            feasible = True
            for ch in children[v]:
                # For child ch, we need some free host h2 != h, h2 not in used_hosts
                # such that (h, h2) not in used_edges_set
                found = False
                for h2 in range(n):
                    if h2 == h or h2 in used_hosts:
                        continue
                    if (h, h2) in used_edges_set or (h2, h) in used_edges_set:
                        continue
                    found = True
                    break
                if not found:
                    feasible = False
                    break
            if not feasible:
                continue
            
            map_tree[v] = h
            used_hosts.add(h)
            if backtrack(idx + 1):
                return True
            used_hosts.remove(h)
            map_tree[v] = -1
        return False
    
    # Try different roots and different host assignments for the root
    # To speed up, try root at various host vertices
    best_result = None
    for root_host in range(n):
        map_tree = [-1] * k
        used_hosts = set()
        map_tree[root] = root_host
        used_hosts.add(root_host)
        
        # Reset order to start after root
        order_no_root = [v for v in order if v != root]
        
        def backtrack2(idx):
            if time.time() > deadline:
                return False
            if idx == len(order_no_root):
                return True
            v = order_no_root[idx]
            p = parent[v]
            for h in range(n):
                if h in used_hosts:
                    continue
                a, b = map_tree[p], h
                if (a, b) in used_edges_set or (b, a) in used_edges_set:
                    continue
                # Forward check for children
                feasible = True
                for ch in children[v]:
                    found = False
                    for h2 in range(n):
                        if h2 == h or h2 in used_hosts:
                            continue
                        if (h, h2) in used_edges_set or (h2, h) in used_edges_set:
                            continue
                        found = True
                        break
                    if not found:
                        feasible = False
                        break
                if not feasible:
                    continue
                map_tree[v] = h
                used_hosts.add(h)
                if backtrack2(idx + 1):
                    return True
                used_hosts.remove(h)
                map_tree[v] = -1
            return False
        
        if backtrack2(0):
            best_result = map_tree[:]
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 20
    trees = _instance_trees(n, 611)
    
    # Total edges available in K20 = 190
    # We want to maximize total embedded tree edges
    # Process trees in order of decreasing size (most constrained first)
    # But T1 has 0 edges, T2 has 1, etc.
    
    # Order: largest trees first
    tree_indices = list(range(n))  # 0..19, tree k has k+1 vertices, k edges
    # Tree index i has (i+1) vertices and i edges (for i>=1), tree 0 has 0 edges
    # Sort by number of edges descending
    tree_indices.sort(key=lambda i: -i)  # larger i = more edges
    
    used_edges_set = set()
    result = [None] * n
    total_edges = 0
    
    for idx in tree_indices:
        if time.time() > deadline:
            break
        tree_edges = trees[idx]
        if not tree_edges:
            # T1: single vertex, always embeddable
            result[idx] = [0]
            continue
        embedding = _embed_tree(tree_edges, n, used_edges_set, deadline)
        if embedding is None:
            continue
        # Add edges to used set
        for a, b in tree_edges:
            ha, hb = embedding[a], embedding[b]
            used_edges_set.add((ha, hb))
            used_edges_set.add((hb, ha))
        result[idx] = embedding
        total_edges += len(tree_edges)
    
    # Verify and convert to proper int lists
    final_result = []
    for i in range(n):
        if result[i] is None:
            final_result.append(None)
        else:
            # Ensure all are Python ints
            final_result.append([int(x) for x in result[i]])
    
    return final_result
