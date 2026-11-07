from random import Random
import time
from itertools import permutations

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


def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    trees = instance_trees(20, 611)
    
    # Tree k has k vertices (indices 0..k-1) and len(trees[k-1]) edges
    # Total edges should be 190
    
    # Order: embed larger trees first (more constrained)
    # Tree index i corresponds to T_{i+1}
    order = list(range(20))  # 0..19, tree at index i has i+1 vertices
    order.sort(key=lambda i: -(i + 1))  # largest first
    
    # Track used edges: set of frozenset({u,v}) for u<v
    used_edges = set()
    embeddings = [None] * 20  # embeddings[i] = list of 20 ints or None
    
    def get_remaining_time():
        return deadline - time.time()
    
    def embed_tree(tree_idx, tree_edges, n_vertices):
        """Try to embed tree with given edges into K20, avoiding used_edges.
        Returns a list of 20 ints (mapping vertex i -> host vertex) or None."""
        if get_remaining_time() < 5:
            return None
        
        # Backtracking search
        # Assign host vertices to tree vertices 0..n_vertices-1
        assignment = [-1] * n_vertices
        used_hosts = [False] * 20
        
        # Build adjacency for the tree
        adj = [[] for _ in range(n_vertices)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by degree descending (more constrained first)
        vertex_order = list(range(n_vertices))
        vertex_order.sort(key=lambda v: -len(adj[v]))
        
        # We'll assign in this order
        assigned_in_order = [False] * n_vertices
        assign_idx = [0] * n_vertices  # position in vertex_order
        
        def backtrack(pos):
            if get_remaining_time() < 3:
                return False
            if pos == len(vertex_order):
                return True
            
            v = vertex_order[pos]
            for host in range(20):
                if used_hosts[host]:
                    continue
                # Check all edges from v to already-assigned neighbors
                valid = True
                for nb in adj[v]:
                    if assigned_in_order[nb]:
                        edge = (min(v, nb), max(v, nb))
                        # In the tree, edge is between vertex v and nb
                        # The host edge would be between assignment[v] and assignment[nb]
                        # But we're assigning v now, so check if (host, assignment[nb]) is used
                        he = frozenset((host, assignment[nb]))
                        if he in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Also need to check: for edges from v to unassigned neighbors,
                # we can't check yet, but we need to ensure that when we assign
                # those neighbors, the edge won't conflict. That's handled later.
                
                assignment[v] = host
                used_hosts[host] = True
                assigned_in_order[v] = True
                
                if backtrack(pos + 1):
                    return True
                
                assignment[v] = -1
                used_hosts[host] = False
                assigned_in_order[v] = False
            
            return False
        
        if backtrack(0):
            # Convert to full 20-element list (unmapped vertices don't matter, but 
            # the interface expects a list of k distinct ints for vertex i)
            # Actually, re-reading: "position i is the image of vertex i of the fixed tree T_k"
            # So we need a list of length k where entry i is the host vertex for tree vertex i
            result = []
            for i in range(n_vertices):
                result.append(assignment[i])
            return result
        return None
    
    # Try embedding trees in order
    success = True
    for tree_idx in order:
        n_vertices = tree_idx + 1
        tree_edges = trees[tree_idx]
        
        if n_vertices == 1:
            # T_1 has 0 edges, just pick any unused vertex... but actually no edges,
            # so any host vertex works. Pick the first available.
            for h in range(20):
                if not any(embeddings[j] is not None and h in embeddings[j] for j in range(20)):
                    embeddings[tree_idx] = [h]
                    break
            else:
                embeddings[tree_idx] = [0]  # fallback
            continue
        
        if n_vertices == 2:
            # T_2 has one edge (0,1). Need two distinct hosts with unused edge between them.
            found = False
            for u in range(20):
                if get_remaining_time() < 3:
                    break
                for v in range(u + 1, 20):
                    he = frozenset((u, v))
                    if he in used_edges:
                        continue
                    # Check u and v not used by other trees
                    u_used = any(embeddings[j] is not None and u in embeddings[j] for j in range(20))
                    v_used = any(embeddings[j] is not None and v in embeddings[j] for j in range(20))
                    if u_used or v_used:
                        continue
                    embeddings[tree_idx] = [u, v]
                    used_edges.add(he)
                    found = True
                    break
                if found:
                    break
            if not found:
                success = False
                break
            continue
        
        # For larger trees, use backtracking
        result = embed_tree(tree_idx, tree_edges, n_vertices)
        if result is None:
            success = False
            break
        
        embeddings[tree_idx] = result
        # Mark edges as used
        for u, v in tree_edges:
            hu = result[u]
            hv = result[v]
            he = frozenset((hu, hv))
            used_edges.add(he)
        
        if get_remaining_time() < 10:
            break
    
    # Validate and return
    if not success:
        # Return best effort: just T20 identity
        return [None] * 19 + [list(range(20))]
    
    # Final validation
    used_edges_check = set()
    for i in range(20):
        if embeddings[i] is None:
            continue
        emb = embeddings[i]
        # Check distinct
        if len(set(emb)) != len(emb):
            return [None] * 19 + [list(range(20))]
        # Check range
        for x in emb:
            if not isinstance(x, int) or x < 0 or x > 19:
                return [None] * 19 + [list(range(20))]
        # Check edges
        for u, v in trees[i]:
            he = frozenset((emb[u], emb[v]))
            if he in used_edges_check:
                return [None] * 19 + [list(range(20))]
            used_edges_check.add(he)
    
    # Ensure all are Python ints
    result = []
    for i in range(20):
        if embeddings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in embeddings[i]])
    return result
