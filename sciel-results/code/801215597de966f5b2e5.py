import time
import random

def _is_3_colorable(adj, n):
    """Check if graph with adjacency bitmasks is 3-colorable using backtracking."""
    colors = [-1] * n
    
    def backtrack(i):
        if i == n:
            return True
        
        forbidden = 0
        neighbors = adj[i]
        while neighbors:
            lsb = neighbors & -neighbors
            j = lsb.bit_length() - 1
            if colors[j] != -1:
                forbidden |= (1 << colors[j])
            neighbors ^= lsb
        
        available = 7 ^ forbidden
        
        while available:
            lsb = available & -available
            c = lsb.bit_length() - 1
            colors[i] = c
            if backtrack(i + 1):
                return True
            colors[i] = -1
            available ^= lsb
            
        return False
    
    return backtrack(0)

def _count_triangles(adj, n):
    """Count number of triangles in graph with adjacency bitmasks."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                count += bin(common).count('1')
    return count // 3

def _local_search(adj, n, all_edges, start_time, deadline, best_edges, best_count):
    """
    Max-SAT style local search: maximize edges subject to triangle-freeness (hard constraint).
    Uses incremental triangle count updates for efficiency.
    """
    current_adj = list(adj)
    current_edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if current_adj[u] & (1 << v):
                current_edge_set.add((u, v))
    
    # Precompute triangle contributions for each potential edge
    # tri_contrib[u,v] = number of w such that (u,w) and (v,w) are edges
    # When we add (u,v), triangles increase by tri_contrib[u,v]
    # When we remove (u,v), triangles decrease by tri_contrib[u,v] (computed before removal)
    
    def get_tri_contrib(u, v):
        """Count common neighbors of u and v (potential triangles through edge (u,v))."""
        return bin(current_adj[u] & current_adj[v]).count('1')
    
    current_triangles = _count_triangles(current_adj, n)
    
    temp = 0.5
    no_improve = 0
    step = 0
    
    while time.time() < deadline:
        step += 1
        u, v = random.choice(all_edges)
        
        if (u, v) in current_edge_set:
            # Consider removing edge (u,v)
            tri_before = current_triangles
            # Remove edge
            common = current_adj[u] & current_adj[v]
            tri_contrib = bin(common).count('1')
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edge_set.remove((u, v))
            new_tri = tri_before - tri_contrib
            
            # Accept removal if it helps (reduces triangles) or with probability temp
            if new_tri < tri_before or random.random() < temp:
                current_triangles = new_tri
                if new_tri < tri_before:
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edge_set.add((u, v))
                no_improve += 1
        else:
            # Consider adding edge (u,v)
            common = current_adj[u] & current_adj[v]
            tri_contrib = bin(common).count('1')
            
            if tri_contrib == 0:
                # Safe to add, no new triangles
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edge_set.add((u, v))
                no_improve = 0
            else:
                # Would create triangles - accept with low probability
                if random.random() < temp * 0.05:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edge_set.add((u, v))
                    current_triangles += tri_contrib
                    no_improve += 1
                else:
                    no_improve += 1
        
        # Check for valid solution periodically
        if step % 20 == 0 and current_triangles == 0:
            if not _is_3_colorable(current_adj, n):
                edge_count = len(current_edge_set)
                if edge_count > best_count[0]:
                    edges = [[u, v] for u, v in sorted(current_edge_set)]
                    best_edges[0] = edges
                    best_count[0] = edge_count
        
        # Decay temperature
        if no_improve > 50:
            temp *= 0.95
            if temp < 0.01:
                break
    
    return best_edges, best_count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edges = [None]
    best_count = [0]
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    # Multiple restarts with local search
    while time.time() < deadline:
        # Initialize with random triangle-free graph
        adj = [0] * n
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Run local search from this starting point
        best_edges, best_count = _local_search(
            adj, n, all_edges, start_time, deadline, best_edges, best_count
        )
        
        # Early exit if we've found a very good solution
        if best_count[0] >= 40:
            break
    
    if best_edges[0] is None:
        # Fallback: Use a known 4-chromatic triangle-free graph on 16 vertices
        # The Clebsch graph is 16-vertex, 40-edge, triangle-free, 4-chromatic
        # Clebsch graph: complement of the 4-cube Q4
        # Vertices are binary strings of length 4, edges between vertices at Hamming distance 3 or 4
        clebsch_edges = []
        for u in range(16):
            for v in range(u + 1, 16):
                # Hamming distance
                diff = bin(u ^ v).count('1')
                if diff >= 3:
                    clebsch_edges.append([u, v])
        best_edges[0] = clebsch_edges
    
    return best_edges[0]
