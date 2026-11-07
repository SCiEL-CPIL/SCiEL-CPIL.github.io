"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds budget
    
    trees = instance_trees(20, 611)
    
    # Start with just T20 embedded by identity
    result = [None] * 20
    result[19] = list(range(20))
    
    # Track used edges
    used_edges = set()
    for i in range(19, 20):
        pass
    # T20 is the complete graph K20 itself? No, T20 is a tree on 20 vertices.
    # Wait, T20 is a tree with 20 vertices, so it has 19 edges.
    # The identity map embeds T20 into K20 using 19 edges.
    for u, v in trees[19]:
        used_edges.add(frozenset((u, v)))
    
    # Try to add smaller trees, starting from largest to smallest
    # For each tree k, try to find an embedding that doesn't conflict with used edges
    
    # We'll process trees in order from 19 down to 1 (index 18 down to 0)
    # For each tree, we need to find a mapping of its vertices to K20 vertices
    # such that no edge maps to an already-used edge
    
    # Simple greedy approach: for each tree, try random permutations of vertex assignments
    # and check if the embedding is valid
    
    def try_embed_tree(tree_idx, max_attempts=1000):
        k = tree_idx + 1
        if k <= 1:
            return []  # T1 has no vertices
        if k == 2:
            # T2 is just an edge (0, 1)
            # Find any unused edge in K20
            for u in range(20):
                for v in range(u + 1, 20):
                    if frozenset((u, v)) not in used_edges:
                        return [u, v]
            return None
        if k > 20:
            return None
            
        tree_edges = trees[tree_idx]
        
        # Try random mappings
        for attempt in range(max_attempts):
            # Random permutation of 20 vertices, take first k
            perm = list(range(20))
            random.shuffle(perm)
            mapping = perm[:k]
            
            # Check if this embedding is valid
            valid = True
            for u, v in tree_edges:
                eu, ev = mapping[u], mapping[v]
                if frozenset((eu, ev)) in used_edges:
                    valid = False
                    break
            if valid:
                return mapping
        return None
    
    # Process trees from 18 down to 1 (index 18 down to 0)
    for tree_idx in range(18, -1, -1):
        if time.time() > deadline - 5:
            break
            
        k = tree_idx + 1
        if k <= 1:
            # T1 has 1 vertex. We need to return a list of 1 distinct int.
            # Any vertex in 0..19 is valid as there are no edges.
            result[tree_idx] = [0]
            continue
        
        mapping = try_embed_tree(tree_idx, max_attempts=500)
        if mapping is not None:
            result[tree_idx] = mapping
            # Mark edges as used
            for u, v in trees[tree_idx]:
                eu, ev = mapping[u], mapping[v]
                used_edges.add(frozenset((eu, ev)))
        else:
            result[tree_idx] = None
    
    return result

# EVOLVE-BLOCK-END
