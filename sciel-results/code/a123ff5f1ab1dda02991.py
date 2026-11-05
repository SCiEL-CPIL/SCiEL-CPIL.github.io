"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    n = 18
    
    # Bitmask adjacency representation
    # adj[i] is a bitmask of neighbors of vertex i
    
    def is_k4_free(adj):
        """Check if graph with adjacency bitmasks is K4-free."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check if i,j have a common neighbor k
                    common = adj[i] & adj[j]
                    # Remove i and j from common
                    common &= ~(1 << i) & ~(1 << j)
                    # For each pair of common neighbors (k,l), check if k-l edge exists
                    # and i-k, i-l, j-k, j-l all exist (they do by definition of common)
                    # Need: for some k,l in common, edge k-l exists
                    # That would make i,j,k,l a K4
                    while common:
                        k = (common & -common).bit_length() - 1
                        common &= common - 1
                        # Check if k has any neighbor l in common
                        if adj[k] & common:
                            return False
        return True
    
    def max_triangle_free_induced(adj):
        """Find the largest subset of vertices inducing a triangle-free subgraph."""
        # Use branch and bound
        best = 0
        best_set = set()
        
        def has_triangle(vertices_mask):
            """Check if vertices_mask induces a triangle."""
            verts = []
            m = vertices_mask
            while m:
                v = (m & -m).bit_length() - 1
                verts.append(v)
                m &= m - 1
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if adj[verts[i]] & (1 << verts[j]):
                        # Check common neighbors in vertices_mask
                        common = adj[verts[i]] & adj[verts[j]] & vertices_mask
                        if common:
                            return True
            return False
        
        # Simple approach: try all subsets of size k from high to low
        # For n=18, this is 2^18 = 262144 subsets, feasible
        # Find largest k such that there exists a k-subset with no triangle
        
        # Precompute: for each vertex, its neighbors mask
        # A subset S is triangle-free iff for every triple in S, not all three edges exist
        
        # Use the fact that we can check incrementally
        # Iterate over all subsets, track if triangle-free
        # 2^18 = 262144, and checking each takes some time but is feasible
        
        # Actually, let's use a smarter approach: 
        # For each subset, check if triangle-free
        # We can use DP or just brute force since 2^18 is manageable
        
        # Let's do it by checking all subsets of size >= current_best+1
        # Start from size 12 down (since we know at most 12 is possible from seed)
        
        for size in range(12, 0, -1):
            if time.time() > deadline - 5.0:
                break
            # Check if there exists a subset of this size that is triangle-free
            # Use combinations
            from itertools import combinations
            found = False
            for combo in combinations(range(n), size):
                if time.time() > deadline - 2.0:
                    return best if best > 0 else size + 1
                # Check if this combo is triangle-free
                triangle_free = True
                for a_idx in range(size):
                    for b_idx in range(a_idx+1, size):
                        a, b = combo[a_idx], combo[b_idx]
                        if adj[a] & (1 << b):
                            # Check if there's a common neighbor in combo
                            common = adj[a] & adj[b]
                            # Check if any vertex in combo is in common
                            for c in combo:
                                if c != a and c != b and (common & (1 << c)):
                                    triangle_free = False
                                    break
                        if not triangle_free:
                            break
                    if not triangle_free:
                        break
                if triangle_free:
                    found = True
                    best = size
                    break
            if found:
                return best
        return best
    
    # Try several random K4-free graphs and find the one with smallest max triangle-free induced subgraph
    
    best_edges = None
    best_score = 13  # worse than seed's 12
    
    random.seed(42)
    
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random K4-free graph
        adj = [0] * n
        edges = []
        
        # Start with empty graph, add edges greedily
        # Random order of potential edges
        potential_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(potential_edges)
        
        # Density parameter: try different densities
        density = 0.3 + 0.4 * (trial % 5) / 5.0
        
        for i, j in potential_edges:
            if time.time() > deadline - 5.0:
                break
            # Check if adding edge (i,j) would create a K4
            # A K4 containing edge (i,j) needs two other vertices x,y
            # such that all edges among {i,j,x,y} exist
            # i.e., x and y are both neighbors of i and j, and x-y edge exists
            common = adj[i] & adj[j]
            common &= ~(1 << i) & ~(1 << j)
            
            # Check if any pair in common has an edge between them
            creates_k4 = False
            temp_common = common
            while temp_common:
                x = (temp_common & -temp_common).bit_length() - 1
                temp_common &= temp_common - 1
                if adj[x] & temp_common:
                    creates_k4 = True
                    break
            
            if not creates_k4:
                # Also check: the edge itself plus existing structure
                # We already checked: if x,y are common neighbors of i,j and x-y edge exists,
                # then {i,j,x,y} is K4
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
        
        # Verify K4-free
        if not is_k4_free(adj):
            continue
        
        # Compute score
        score = max_triangle_free_induced(adj)
        
        if score < best_score:
            best_score = score
            best_edges = edges[:]
        
        if best_score <= 10:
            break
    
    if best_edges is not None:
        return best_edges
    else:
        # Fallback: seed construction
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
# EVOLVE-BLOCK-END
