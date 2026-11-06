import time
import random
import sys

# EVOLVE-BLOCK-START

def _paley_37_bitset():
    """
    Construct the Paley graph on 37 vertices and return it as a list of bitsets.
    adj[i] is an integer where bit j is 1 if (i, j) is an edge.
    """
    q = 37
    # Quadratic residues modulo 37
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _has_k4_in_set(adj, vertices, mask):
    """
    Check if the subgraph induced by 'vertices' (subset of 'mask') contains a K_4.
    'adj' is the global adjacency list (bitsets).
    'vertices' is a list of vertex indices.
    Returns True if K_4 exists, False otherwise.
    """
    n = len(vertices)
    if n < 4:
        return False
    
    # Map vertex indices to local indices 0..n-1 for easier checking?
    # Or just use the global indices.
    
    # Check all combinations of 4 vertices
    # For speed, we can use bit operations.
    # Let's just iterate combinations for simplicity first, as n is small.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    v1, v2, v3, v4 = vertices[i], vertices[j], vertices[k], vertices[l]
                    # Check if all 6 edges exist
                    if ((adj[v1] >> v2) & 1) and ((adj[v1] >> v3) & 1) and ((adj[v1] >> v4) & 1) and \
                       ((adj[v2] >> v3) & 1) and ((adj[v2] >> v4) & 1) and ((adj[v3] >> v4) & 1):
                        return True
    return False

def _find_k4_in_set(adj, vertices):
    """
    Find a specific K_4 in the set of vertices.
    Returns the tuple of 4 vertices or None.
    """
    n = len(vertices)
    if n < 4:
        return None
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    v1, v2, v3, v4 = vertices[i], vertices[j], vertices[k], vertices[l]
                    if ((adj[v1] >> v2) & 1) and ((adj[v1] >> v3) & 1) and ((adj[v1] >> v4) & 1) and \
                       ((adj[v2] >> v3) & 1) and ((adj[v2] >> v4) & 1) and ((adj[v3] >> v4) & 1):
                        return (v1, v2, v3, v4)
    return None

def _extend_vertex(adj, n_base, new_v, deadline, max_iters=1000):
    """
    Try to add a new vertex 'new_v' to the graph defined by adj[0..n_base-1].
    Returns True if successful, and updates adj.
    """
    n_total = n_base + 1
    # Extend adj list
    adj.append(0)
    
    # We will perform a local search on the edges between new_v and 0..n_base-1.
    # Random seed for this vertex
    random.seed(new_v * 1000 + 7)
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            return False
            
        # Initialize edges randomly
        for v in range(n_base):
            if random.random() < 0.5:
                adj[new_v] |= (1 << v)
                adj[v] |= (1 << new_v)
            else:
                adj[new_v] &= ~(1 << v)
                adj[v] &= ~(1 << new_v)
                
        # Check for conflicts involving new_v
        # 1. K_5 containing new_v: Neighbors of new_v must not contain a K_4.
        # 2. Independent 5-set containing new_v: Non-neighbors of new_v must not contain a K_4 (in original graph, i.e., independent set of size 4).
        
        # Get neighbors and non-neighbors
        neighbors = []
        non_neighbors = []
        for v in range(n_base):
            if (adj[new_v] >> v) & 1:
                neighbors.append(v)
            else:
                non_neighbors.append(v)
                
        conflict = None
        # Check neighbors for K_4
        if len(neighbors) >= 4:
            k4 = _find_k4_in_set(adj, neighbors)
            if k4:
                conflict = (True, k4) # True means clique conflict
        
        # Check non-neighbors for independent 4-set (which is K_4 in complement)
        if len(non_neighbors) >= 4:
            # An independent set of size 4 in G is a K_4 in G_bar.
            # So we check if non_neighbors contains a set of 4 vertices with NO edges between them.
            # This is equivalent to checking if they form a K_4 in the complement.
            # Let's write a helper for independent 4-set.
            n_non = len(non_neighbors)
            found_indep = False
            for i in range(n_non):
                for j in range(i + 1, n_non):
                    for k in range(j + 1, n_non):
                        for l in range(k + 1, n_non):
                            v1, v2, v3, v4 = non_neighbors[i], non_neighbors[j], non_neighbors[k], non_neighbors[l]
                            # Check if NO edges exist
                            if not ((adj[v1] >> v2) & 1) and not ((adj[v1] >> v3) & 1) and not ((adj[v1] >> v4) & 1) and \
                               not ((adj[v2] >> v3) & 1) and not ((adj[v2] >> v4) & 1) and not ((adj[v3] >> v4) & 1):
                                conflict = (False, (v1, v2, v3, v4)) # False means independent conflict
                                found_indep = True
                                break
                        if found_indep: break
                    if found_indep: break
                if found_indep: break
            
        if conflict is None:
            # No conflict, success!
            return True
            
        # Resolve conflict
        is_clique, verts = conflict
        # Pick a random vertex from the conflict set to flip the edge to new_v
        v_to_flip = random.choice(verts)
        
        # Flip edge (new_v, v_to_flip)
        if (adj[new_v] >> v_to_flip) & 1:
            adj[new_v] &= ~(1 << v_to_flip)
            adj[v_to_flip] &= ~(1 << new_v)
        else:
            adj[new_v] |= (1 << v_to_flip)
            adj[v_to_flip] |= (1 << new_v)
            
    return False

def construct():
    deadline = time.time() + 280
    
    # Start with Paley 37
    adj = _paley_37_bitset()
    n_base = 37
    
    # Try to extend to 42
    target_n = 42
    
    # We might need to restart the whole extension process if we get stuck?
    # Or just restart the specific vertex addition.
    
    success = True
    for new_v in range(n_base, target_n):
        # Try multiple times to add this vertex
        added = False
        for restart in range(50):
            if time.time() > deadline:
                success = False
                break
            # Reset adj to state before this vertex was added?
            # This is tricky because _extend_vertex modifies adj.
            # Let's snapshot adj before trying to add a vertex.
            adj_snapshot = adj[:]
            
            # Call extend
            if _extend_vertex(adj, n_base, new_v, deadline, max_iters=200):
                added = True
                n_base += 1
                break
            else:
                # Restore snapshot
                adj = adj_snapshot[:]
                
        if not added:
            success = False
            break
        if not success:
            break
            
    if success:
        # Build edge list
        edges = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return (n_base, edges)
    else:
        # Fallback to base graph
        edges = []
        for i in range(37):
            for j in range(i + 1, 37):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return (37, edges)

# EVOLVE-BLOCK-END
