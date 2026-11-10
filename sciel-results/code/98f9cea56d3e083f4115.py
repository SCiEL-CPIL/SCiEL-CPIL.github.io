"""Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import time
import random

# Deadline guard
_START = time.time()
_DEADLINE = 145.0
N = 18

def _elapsed():
    return time.time() - _START

def _turan_edges():
    """Generate edges for T(18,3) - complete 3-partite with parts of 6."""
    edges = []
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def _edges_to_adj(edges):
    """Convert edge list to adjacency bitmasks."""
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj):
    """Check if graph is K4-free using bitmasks."""
    for a in range(N):
        for b in range(a + 1, N):
            if not (adj[a] & (1 << b)):
                continue
            # Common neighbors of a and b
            common = adj[a] & adj[b] & ~(1 << a) & ~(1 << b)
            # Check if any pair of common neighbors are adjacent (forming K4)
            c_bits = common
            while c_bits:
                c = (c_bits & -c_bits).bit_length() - 1
                c_bits &= c_bits - 1
                if common & (1 << c) and (adj[c] & common):
                    # c and some other common neighbor are adjacent
                    d_bits = common & adj[c] & ~(1 << c)
                    if d_bits:
                        return False
    return True

def _max_triangle_free_induced(adj):
    """Find the maximum size of a triangle-free induced subgraph using branch and bound."""
    # A set S is triangle-free if it contains no triangle.
    # We want to find the largest such S.
    # Equivalently, we want to remove the minimum number of vertices to make the graph triangle-free.
    
    # Precompute triangles
    triangles = []
    for a in range(N):
        for b in range(a + 1, N):
            if adj[a] & (1 << b):
                common = adj[a] & adj[b] & ~(1 << a) & ~(1 << b)
                while common:
                    c = (common & -common).bit_length() - 1
                    common &= common - 1
                    if c > b:
                        triangles.append((a, b, c))
    
    if not triangles:
        return N
    
    # Branch and bound: try to find a maximum triangle-free set
    # Equivalent to minimum vertex cover of the hypergraph of triangles
    # Use a simple branch and bound on vertex deletion
    
    best = [0]
    
    def bt(remaining_mask, removed_count):
        if _elapsed() > _DEADLINE - 2.0:
            return
        if remaining_mask == 0:
            if N - removed_count > best[0]:
                best[0] = N - removed_count
            return
        
        # Upper bound: current removed + remaining can't beat best
        if (N - removed_count) <= best[0]:
            return
        
        # Check if remaining is triangle-free
        is_tf = True
        for (a, b, c) in triangles:
            if (remaining_mask >> a) & 1 and (remaining_mask >> b) & 1 and (remaining_mask >> c) & 1:
                is_tf = False
                break
        if is_tf:
            if N - removed_count > best[0]:
                best[0] = N - removed_count
            return
        
        # Find a triangle in remaining
        tri = None
        for (a, b, c) in triangles:
            if (remaining_mask >> a) & 1 and (remaining_mask >> b) & 1 and (remaining_mask >> c) & 1:
                tri = (a, b, c)
                break
        
        if tri is None:
            if N - removed_count > best[0]:
                best[0] = N - removed_count
            return
        
        a, b, c = tri
        # Try removing each vertex of the triangle
        for v in [a, b, c]:
            new_mask = remaining_mask & ~(1 << v)
            bt(new_mask, removed_count + 1)
    
    full_mask = (1 << N) - 1
    bt(full_mask, 0)
    return best[0]

def _local_search(adj, max_iter=500):
    """Try to improve by adding/removing edges while maintaining K4-freeness."""
    best_score = _max_triangle_free_induced(adj)
    current_score = best_score
    current_adj = list(adj)
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 3.0:
            break
        
        # Try random edge additions
        improved = False
        attempts = 0
        while attempts < 20 and not improved:
            attempts += 1
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v or (current_adj[u] >> v) & 1:
                continue
            
            # Check if adding edge (u,v) keeps K4-free
            # Need to check no K4 is formed
            new_adj = list(current_adj)
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not _is_k4_free(new_adj):
                continue
            
            score = _max_triangle_free_induced(new_adj)
            if score < current_score:
                current_adj = new_adj
                current_score = score
                if score < best_score:
                    best_score = score
                improved = True
                break
            elif score == current_score:
                # Accept with some probability for exploration
                if random.random() < 0.1:
                    current_adj = new_adj
        
        # Try random edge removals (only if score is still high)
        if not improved and current_score > 10:
            attempts = 0
            while attempts < 10:
                attempts += 1
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v or not ((current_adj[u] >> v) & 1):
                    continue
                
                new_adj = list(current_adj)
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                
                score = _max_triangle_free_induced(new_adj)
                if score < current_score:
                    current_adj = new_adj
                    current_score = score
                    if score < best_score:
                        best_score = score
                    break
        
        if current_score <= 10:
            break
    
    return current_adj, best_score

def construct():
    # Start with T(18,3)
    edges = _turan_edges()
    adj = _edges_to_adj(edges)
    baseline_score = _max_triangle_free_induced(adj)
    best_adj = list(adj)
    best_score = baseline_score
    
    # Try local search from T(18,3)
    if _elapsed() < _DEADLINE - 5.0:
        adj2, score2 = _local_search(adj, max_iter=200)
        if score2 < best_score:
            best_adj = adj2
            best_score = score2
    
    # Try random restarts with different initial graphs
    if _elapsed() < _DEADLINE - 5.0:
        for restart in range(5):
            if _elapsed() > _DEADLINE - 5.0:
                break
            
            # Create a random K4-free graph by starting with T(18,3) and removing random edges
            random.seed(restart * 12345 + 678)
            rand_adj = list(adj)
            for _ in range(15):
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u != v and (rand_adj[u] >> v) & 1:
                    rand_adj[u] &= ~(1 << v)
                    rand_adj[v] &= ~(1 << u)
            
            if not _is_k4_free(rand_adj):
                continue
            
            score = _max_triangle_free_induced(rand_adj)
            if score < best_score:
                best_adj = list(rand_adj)
                best_score = score
            
            # Local search from this starting point
            if _elapsed() < _DEADLINE - 5.0:
                adj3, score3 = _local_search(rand_adj, max_iter=100)
                if score3 < best_score:
                    best_adj = adj3
                    best_score = score3
    
    # Convert best adjacency back to edge list
    result_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
    
    return result_edges
