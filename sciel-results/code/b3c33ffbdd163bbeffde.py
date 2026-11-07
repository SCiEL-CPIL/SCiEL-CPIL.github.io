"""Packing trees T1..T20 into K20 via decomposition."""

import time
from itertools import combinations

# Deadline guard: finish well within 150s
_DEADLINE = time.time() + 140.0

def _remaining_time():
    return _DEADLINE - time.time()

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

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    """Build adjacency list from edge list."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj

def _backtrack_embed(tree_edges, available_edges, n=20):
    """
    Try to embed tree_edges into available_edges (set of (u,v) keys).
    Returns a list of n host vertices (image of tree vertex i) or None.
    Uses backtracking with MRV heuristic.
    """
    if not tree_edges:
        return list(range(n))  # identity for empty/1-vertex tree
    
    # Build tree adjacency
    adj = {}
    for u, v in tree_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    
    n_tree = len(tree_edges) + 1
    mapping = [None] * n_tree  # tree vertex -> host vertex
    used_hosts = set()
    
    # Start from vertex 0 of tree
    def bt(vertex, host):
        if time.time() > _DEADLINE:
            return False
        mapping[vertex] = host
        used_hosts.add(host)
        
        # Find next unassigned tree vertex (MRV: fewest available options)
        best_next = None
        best_options = None
        for tv in range(n_tree):
            if mapping[tv] is not None:
                continue
            # Get neighbors that are already assigned
            assigned_neighbors = [nb for nb in adj.get(tv, []) if mapping[nb] is not None]
            if not assigned_neighbors:
                continue  # not connected to assigned part yet, skip for now
            # Options: host must be adjacent to all assigned neighbors' hosts
            options = set()
            for nb in assigned_neighbors:
                nb_host = mapping[nb]
                for h in range(n):
                    if h in used_hosts:
                        continue
                    if _edge_key(nb_host, h) in available_edges:
                        options.add(h)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_next = tv
                best_options = options
                if len(best_options) == 1:
                    break
        
        if best_next is None:
            # All tree vertices assigned - check success
            used_hosts.discard(host)
            del mapping[vertex]
            return True
        
        for h in sorted(best_options, key=lambda x: -len(adj.get(x, []))):
            if bt(best_next, h):
                return True
        
        used_hosts.discard(host)
        del mapping[vertex]
        return False
    
    # Try each possible host for tree vertex 0
    for h in range(n):
        mapping[0] = h
        used_hosts = {h}
        # Check if we can proceed
        if bt(0, h):
            return mapping[:]
    
    return None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    trees = instance_trees(20, 611)
    
    # All 190 edges of K20
    all_edges = set()
    for i in range(20):
        for j in range(i + 1, 20):
            all_edges.add((i, j))
    
    available = set(all_edges)
    result = [None] * 20
    
    # Embed trees from largest (T20) to smallest (T1)
    # T_k has k vertices and k-1 edges (for k>=2), T1 has 0 edges
    order = list(range(19, -1, -1))  # indices 19 down to 0, i.e., T20 to T1
    
    for idx in order:
        if time.time() > _DEADLINE - 5:
            break
        k = idx + 1  # tree size
        tree_edges = trees[idx]
        
        if not tree_edges:
            # T1: single vertex, no edges
            # Assign any unused host
            for h in range(20):
                if h not in set().union(*[set(m) for m in result if m]):
                    result[idx] = [h]
                    break
            else:
                result[idx] = [0]
            continue
        
        # Try to embed
        mapping = _backtrack_embed(tree_edges, available, n=20)
        
        if mapping is None:
            # Failed - this shouldn't happen for a valid decomposition
            # Try with a different approach or give up
            return result
        
        result[idx] = mapping
        
        # Remove used edges from available
        for u, v in tree_edges:
            eu, ev = mapping[u], mapping[v]
            ek = _edge_key(eu, ev)
            if ek not in available:
                # Conflict - shouldn't happen
                return result
            available.discard(ek)
    
    # Verify all are Python ints and lists
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    
    return result
