"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
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
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    trees = instance_trees(20, 611)
    
    # Tree k has k vertices and k-1 edges
    # We need to embed each tree T_k into K20 such that all embeddings are edge-disjoint
    
    # Total edges in all trees: sum(k-1 for k in 1..20) = sum(0..19) = 190
    # K20 has 190 edges, so we need to use every edge exactly once
    
    # Strategy: Try to find embeddings for all trees
    # Order: embed larger trees first (more constrained)
    
    # For each tree, we need to assign its vertices to distinct vertices in 0..19
    # such that no edge is used by more than one tree
    
    # Let's try a backtracking approach with constraint propagation
    
    # First, let's see the sizes of the trees
    tree_sizes = [len(t) + 1 for t in trees]  # k for tree T_k
    # trees[0] has 1 vertex (T1), trees[1] has 2 vertices (T2), etc.
    
    # Order trees by size (largest first) for backtracking
    tree_order = list(range(20))
    tree_order.sort(key=lambda i: tree_sizes[i], reverse=True)
    
    # We'll track which edges are used
    # Edge (u,v) with u < v, total 190 edges
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return u * 20 + v  # unique key for edge
    
    # Precompute all possible edge keys
    all_edges = set()
    for u in range(20):
        for v in range(u+1, 20):
            all_edges.add(edge_key(u, v))
    
    # For each tree, precompute its edge structure
    # tree_edges[i] = list of (a, b) where a, b are vertex indices in the tree
    
    # Backtracking: assign vertices of each tree to host vertices
    
    # For efficiency, let's try a different approach:
    # Since we need to use ALL edges, this is a complete edge decomposition.
    
    # Let me try a simpler heuristic first: 
    # For each tree, try random permutations of host vertices and check edge-disjointness
    
    best_result = None
    best_score = -1
    
    # Precompute tree edges
    tree_edge_lists = []
    for i in range(20):
        edges = trees[i]
        # Normalize edges: (min, max)
        norm_edges = []
        for a, b in edges:
            if a > b:
                a, b = b, a
            norm_edges.append((a, b))
        tree_edge_lists.append(norm_edges)
    
    # Try multiple random restarts
    rng = random.Random(42)
    
    for restart in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        # For each tree in order, try to find an embedding
        used_edges = set()
        embedding = [None] * 20
        success = True
        
        for tree_idx in tree_order:
            k = tree_sizes[tree_idx]
            tree_edges = tree_edge_lists[tree_idx]
            
            # Try to find a permutation of k distinct vertices from 0..19
            # such that no edge in the tree maps to an already-used edge
            
            # For small trees, we can try all permutations
            # For larger trees, use random sampling with backtracking
            
            if k <= 8:
                # Try all permutations (or a subset)
                found = False
                # Use random sampling with increasing attempts
                attempts = 0
                max_attempts = 10000 if k <= 6 else 50000
                while attempts < max_attempts and not found:
                    if time.time() > deadline - 2.0:
                        success = False
                        break
                    perm = rng.sample(range(20), k)
                    # Check if this embedding works
                    valid = True
                    for a, b in tree_edges:
                        u, v = perm[a], perm[b]
                        ek = edge_key(u, v)
                        if ek in used_edges:
                            valid = False
                            break
                    if valid:
                        # Check that all mapped edges are in K20 (they always are)
                        # Accept this embedding
                        for a, b in tree_edges:
                            u, v = perm[a], perm[b]
                            used_edges.add(edge_key(u, v))
                        embedding[tree_idx] = perm
                        found = True
                        break
                    attempts += 1
                
                if not found:
                    success = False
                    break
            else:
                # For larger trees, use a more structured approach
                # Try random embeddings with more attempts
                found = False
                attempts = 0
                max_attempts = 100000
                while attempts < max_attempts and not found:
                    if time.time() > deadline - 2.0:
                        success = False
                        break
                    perm = rng.sample(range(20), k)
                    valid = True
                    for a, b in tree_edges:
                        u, v = perm[a], perm[b]
                        ek = edge_key(u, v)
                        if ek in used_edges:
                            valid = False
                            break
                    if valid:
                        for a, b in tree_edges:
                            u, v = perm[a], perm[b]
                            used_edges.add(edge_key(u, v))
                        embedding[tree_idx] = perm
                        found = True
                        break
                    attempts += 1
                
                if not found:
                    success = False
                    break
        
        if success:
            # We found a valid embedding for all trees
            # Calculate the score: total number of embedded tree edges
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges > best_score:
                best_score = total_edges
                best_result = embedding
                # Since we need to use all 190 edges, this is optimal
                break
    
    if best_result is not None:
        # Convert to the required format
        result = []
        for i in range(20):
            if best_result[i] is None:
                result.append(None)
            else:
                # Ensure it's a list of Python ints
                result.append([int(x) for x in best_result[i]])
        return result
    else:
        # Fallback: only embed T20
        return [None] * 19 + [list(range(20))]
# EVOLVE-BLOCK-END

