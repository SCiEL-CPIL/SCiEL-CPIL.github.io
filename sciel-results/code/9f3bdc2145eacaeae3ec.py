import time
import sys
from random import Random

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # 1. Generate trees
    n = 20
    seed = 611
    trees = []
    rng = Random(seed)
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
        
    # 2. Prepare for embedding
    # We want to return a list of 20 entries.
    # Entry k-1 is None or list of k distinct ints.
    # We will try to embed trees in descending order of size: 20, 19, ..., 1.
    # This maximizes edges if we run out of time or space.
    
    used_edges = set()
    result = [None] * n
    
    # Helper to check if an edge is available
    def is_edge_available(u, v):
        if u == v: return False
        e = (min(u, v), max(u, v))
        return e not in used_edges

    # Helper to add edges to used set
    def add_edges(edges):
        for u, v in edges:
            e = (min(u, v), max(u, v))
            used_edges.add(e)
            
    # Backtracking function to embed a specific tree
    def embed_tree(tree_edges, k):
        # tree_edges: list of (u, v) in tree vertex indices 0..k-1
        # Returns: list of host vertices [h_0, ..., h_{k-1}] or None
        
        # If k == 1, any vertex works.
        if k == 1:
            # Return the first available vertex? Or just 0?
            # Since T1 has 0 edges, it doesn't consume edges.
            # We can map it to any vertex. Let's say 0.
            # But we must ensure distinctness? "list of k distinct int host vertices".
            # For k=1, just [0].
            return [0]
            
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Order vertices to try. High degree first helps constrain.
        # We will use a recursive search.
        # State: mapping (list of length k, -1 for unmapped), 
        #        list of available host vertices for each tree vertex?
        #        Better: just try all permutations? No, k=20 is too big.
        #        Use DFS on the tree structure.
        
        # Root the tree at 0.
        # Precompute parent/children structure or just use visited set.
        
        mapping = [-1] * k
        mapping[0] = 0 # Fix root at 0 to break symmetry? 
                       # Actually, we can try all 20 vertices for root, 
                       # but fixing to 0 and permuting host labels later?
                       # No, host labels are fixed 0..19.
                       # Symmetry breaking: We can fix the first vertex of the tree to 0.
                       # Then we only need to find if an embedding exists with T[0]->0.
                       # If not, we might need to try T[0]->1, etc.
                       # But due to symmetry of K20, if an embedding exists, 
                       # an embedding with T[0]->0 also exists?
                       # Yes, K20 is vertex-transitive.
                       # So we can fix mapping[0] = 0.
                       
        # However, we must ensure that the edges used are available.
        # The available edges depend on which host vertices are chosen.
        
        # Let's try a DFS that assigns host vertices to tree vertices.
        # We process tree vertices in an order that maximizes constraints.
        # Let's use a simple order: 0, then its neighbors, etc. (BFS/DFS order of tree).
        # Or simply 0..k-1.
        
        # To be more efficient, we can check degree constraints.
        # For a tree vertex v, its host H(v) must have enough available edges 
        # to its already assigned neighbors.
        
        def dfs(idx):
            if time.time() - start_time > timeout:
                return False
                
            if idx == k:
                return True
                
            v = idx
            # Try all available host vertices for v
            for h in range(20):
                if mapping.count(h) > 0: # h already used
                    continue
                
                # Check if edges from v to already mapped neighbors are available
                ok = True
                for u in adj[v]:
                    if mapping[u] != -1:
                        if not is_edge_available(h, mapping[u]):
                            ok = False
                            break
                if not ok:
                    continue
                
                # Also check if h has enough total available edges to support future connections?
                # Optional optimization: 
                # deg_available(h) >= degree_in_tree(v) - (mapped_neighbors_count)
                # Let's skip for now to keep it simple, K20 is dense.
                
                mapping[v] = h
                if dfs(idx + 1):
                    return True
                mapping[v] = -1
                
            return False
            
        # We fixed mapping[0] = 0. 
        # But what if the specific tree structure doesn't fit with 0 as root in the available graph?
        # Since K20 is symmetric, if ANY embedding exists, one with T[0]->0 exists?
        # Wait. The available graph is NOT symmetric after some edges are used.
        # The used edges break the symmetry.
        # So we CANNOT fix mapping[0] = 0 blindly.
        # We must try all 20 possibilities for mapping[0].
        
        # Revised DFS:
        # We will try to assign vertices 0..k-1.
        # To reduce branching, we can try to assign the most constrained vertex first.
        # But the tree structure is fixed.
        
        # Let's try iterating over all possible host vertices for the root (0)
        # and then DFS for the rest.
        
        for root_host in range(20):
            mapping = [-1] * k
            mapping[0] = root_host
            
            # Check if root_host has enough available edges?
            # It needs degree_in_tree(0) edges to its neighbors.
            # We will check this in the DFS step when neighbors are assigned.
            
            if dfs(1):
                return mapping[:]
                
        return None
        
    # Embed trees from largest to smallest
    for k in range(n, 0, -1):
        if time.time() - start_time > timeout:
            break
            
        tree = trees[k-1]
        if not tree:
            # T1 is empty. Map to 0.
            result[k-1] = [0]
            continue
            
        embedding = embed_tree(tree, k)
        if embedding:
            result[k-1] = embedding
            # Add edges to used set
            for u, v in tree:
                # u, v are tree vertex indices
                # embedding[u], embedding[v] are host vertices
                h_u = embedding[u]
                h_v = embedding[v]
                e = (min(h_u, h_v), max(h_u, h_v))
                used_edges.add(e)
        else:
            # Failed to embed this tree. 
            # Try smaller trees?
            # If we skip this one, we might fit smaller ones.
            # But we are iterating descending. 
            # We should continue to smaller trees.
            pass

    # Calculate score (number of edges)
    # Just to be sure, we return the list.
    return result
