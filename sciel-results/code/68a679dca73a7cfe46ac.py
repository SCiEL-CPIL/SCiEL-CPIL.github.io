"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def _generate_trees(n=20, seed=611):
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

def _build_tree_adjacency(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree_backtrack(tree_edges, tree_n, used_edges, n=20, rng=None, max_nodes_to_try=50000):
    """Find an embedding of the tree into K20 such that no edge conflicts with used_edges."""
    if not tree_edges:
        return list(range(tree_n))
    
    adj = _build_tree_adjacency(tree_edges, tree_n)
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(tree_n), key=lambda v: -len(adj[v]))
    order_index = {v: i for i, v in enumerate(order)}
    
    assignment = [-1] * tree_n
    used_local = set()
    
    def backtrack(idx):
        if idx == tree_n:
            return True
        v = order[idx]
        # Get neighbors of v that are already assigned
        assigned_neighbors = [(nb, assignment[nb]) for nb in adj[v] if assignment[nb] != -1]
        
        # Candidate host vertices
        candidates = list(range(n))
        if rng is not None:
            rng.shuffle(candidates)
        
        for hv in candidates:
            if hv in used_local:
                continue
            # Check edge constraints with assigned neighbors
            valid = True
            for nb, nb_hv in assigned_neighbors:
                edge = (min(hv, nb_hv), max(hv, nb_hv))
                if edge in used_edges:
                    valid = False
                    break
            if not valid:
                continue
            # Also check that we don't create duplicate edges among assigned
            # (already handled by used_edges check above)
            
            assignment[v] = hv
            used_local.add(hv)
            
            if backtrack(idx + 1):
                return True
            
            assignment[v] = -1
            used_local.remove(hv)
        
        return False
    
    if backtrack(0):
        return assignment
    return None

def _try_full_packing(trees, n=20, seed=0, deadline=None):
    """Try to pack all trees. Returns list of embeddings or None."""
    rng = Random(seed)
    used_edges = set()
    embeddings = [None] * len(trees)
    
    # Process trees in order of decreasing number of edges (hardest first)
    tree_indices = sorted(range(len(trees)), key=lambda i: -len(trees[i]))
    
    for idx in tree_indices:
        tree = trees[idx]
        tree_n = idx + 1  # Tree T_k has k vertices
        
        if not tree:
            embeddings[idx] = [0]  # Single vertex, map to any
            continue
        
        # Try to embed this tree
        assignment = _embed_tree_backtrack(tree, tree_n, used_edges, n, rng)
        if assignment is None:
            return None
        
        embeddings[idx] = assignment
        # Mark edges as used
        for u, v in tree:
            hu, hv = assignment[u], assignment[v]
            edge = (min(hu, hv), max(hu, hv))
            if edge in used_edges:
                return None  # Shouldn't happen, but safety
            used_edges.add(edge)
    
    return embeddings

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    trees = _generate_trees(20, 611)
    n = 20
    
    # Try with different seeds and strategies
    best_result = None
    
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        result = _try_full_packing(trees, n, seed, deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback: just embed T20 as identity
        return [None] * 19 + [list(range(20))]
    
    # Convert to required format
    output = []
    for k in range(1, 21):
        idx = k - 1
        emb = best_result[idx]
        if emb is None:
            output.append(None)
        else:
            # Ensure it's a list of Python ints
            output.append([int(v) for v in emb])
    
    return output
# EVOLVE-BLOCK-END
