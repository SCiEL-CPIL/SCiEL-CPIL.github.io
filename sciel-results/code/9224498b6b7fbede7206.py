"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

N = 20
DEADLINE = 115.0  # safety margin before 120s kill

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

def _edge_id(a, b):
    if a > b:
        a, b = b, a
    return a * N + b

def _count_edges(m, edges):
    """Count how many edges of the tree under mapping m are already used (waste)."""
    cnt = {}
    for (a, b) in edges:
        e = _edge_id(m[a], m[b])
        cnt[e] = cnt.get(e, 0) + 1
    # Waste = edges that overlap with previously placed trees
    # Actually we need to count against the global used set
    pass

def construct():
    start_time = time.time()
    trees = instance_trees(N, 611)
    
    # We'll embed trees in order 1..20
    # For tree k (0-indexed), we need to find a mapping of its k vertices to distinct 
    # vertices in 0..19 such that all edges are distinct from edges used by previously embedded trees.
    
    # Strategy: greedy with backtracking / local search for each tree
    # Keep track of used edges as a set
    used_edges = set()
    result = [None] * N
    
    # Order: embed from largest to smallest? No, we must embed in order 1..20.
    # Actually the task says "Entry k-1 is None or a list..." and "Embedded trees must be 
    # edge-disjoint in K20." So we need all 20 trees to be simultaneously embedded 
    # with edge-disjointness. We can choose any order for the search, but the result 
    # must have entries 0..19 corresponding to trees T1..T20.
    
    # Let's embed in order 1 to 20, using a greedy approach with random restarts.
    # For each tree, find a mapping that avoids all edges in used_edges.
    
    rng = Random(42)
    
    # Precompute adjacency for K20 - all pairs
    all_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.add((i, j))
    
    # For each tree k (1-indexed, 0-indexed as k-1), find an embedding
    for k_idx in range(N):
        k = k_idx + 1  # number of vertices
        tree_edges = trees[k_idx]
        
        if k == 1:
            # Single vertex, no edges - any vertex works
            result[k_idx] = [0]
            continue
        
        if k == 2:
            # Two vertices connected by an edge
            found = False
            for attempt in range(100000):
                if time.time() - start_time > DEADLINE:
                    break
                a = rng.randint(0, N - 1)
                b = rng.randint(0, N - 1)
                if a == b:
                    continue
                edge = (a, b) if a < b else (b, a)
                if edge not in used_edges:
                    result[k_idx] = [a, b]
                    used_edges.add(edge)
                    found = True
                    break
            if not found:
                # Try all pairs
                for a in range(N):
                    for b in range(a + 1, N):
                        edge = (a, b)
                        if edge not in used_edges:
                            result[k_idx] = [a, b]
                            used_edges.add(edge)
                            found = True
                            break
                    if found:
                        break
            if not found:
                # Cannot embed this tree, mark as None (but this shouldn't happen for K20)
                result[k_idx] = None
            continue
        
        # For larger trees, use random search with constraint that no edge is in used_edges
        found = False
        max_attempts = 500000
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE:
                break
            # Try random permutation of k vertices chosen from 0..19
            if k <= N:
                mapping = rng.sample(range(N), k)
            else:
                break
            
            # Check if all edges are valid (not in used_edges and endpoints distinct)
            valid = True
            tree_used_edges = []
            for (u, v) in tree_edges:
                a, b = mapping[u], mapping[v]
                if a == b:
                    valid = False
                    break
                edge = (a, b) if a < b else (b, a)
                if edge in used_edges:
                    valid = False
                    break
                tree_used_edges.append(edge)
            
            if valid:
                # Check for duplicate edges within this tree (shouldn't happen for a tree)
                if len(set(tree_used_edges)) == len(tree_used_edges):
                    result[k_idx] = mapping
                    used_edges.update(tree_used_edges)
                    found = True
                    break
        
        if not found:
            # Try a more structured approach: use a spanning tree of K20 restricted to k vertices
            # or use a greedy placement
            found2 = False
            for restart in range(1000):
                if time.time() - start_time > DEADLINE:
                    break
                # Assign vertices one by one
                mapping = [None] * k
                available = list(range(N))
                rng.shuffle(available)
                
                # Try to place each vertex
                for vertex in range(k):
                    if vertex == 0:
                        # Place first vertex randomly
                        mapping[vertex] = available.pop()
                    else:
                        # Place subsequent vertex: pick from available
                        # But we need to ensure edges to already-placed neighbors don't conflict
                        # This is complex; let's just do random
                        if not available:
                            break
                        mapping[vertex] = available.pop()
                
                if any(m is None for m in mapping):
                    continue
                
                # Check validity
                valid = True
                tree_used_edges = []
                for (u, v) in tree_edges:
                    a, b = mapping[u], mapping[v]
                    if a is None or b is None or a == b:
                        valid = False
                        break
                    edge = (a, b) if a < b else (b, a)
                    if edge in used_edges:
                        valid = False
                        break
                    tree_used_edges.append(edge)
                
                if valid and len(set(tree_used_edges)) == len(tree_used_edges):
                    result[k_idx] = mapping
                    used_edges.update(tree_used_edges)
                    found2 = True
                    break
            
            if found2:
                found = True
            elif not found:
                # Last resort: leave as None (will reduce score)
                result[k_idx] = None
    
    # Verify all entries are lists of distinct ints in 0..19
    for i in range(N):
        if result[i] is not None:
            if not isinstance(result[i], list):
                result[i] = list(result[i])
            # Ensure all are Python ints
            result[i] = [int(x) for x in result[i]]
            # Check distinctness
            if len(set(result[i])) != len(result[i]):
                result[i] = None
    
    return result
# EVOLVE-BLOCK-END
