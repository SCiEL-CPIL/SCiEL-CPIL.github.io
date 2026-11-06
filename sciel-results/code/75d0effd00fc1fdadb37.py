import time
import random

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    Vertices 0..4 form C5.
    Vertices 5..9 are copies of 0..4.
    Vertex 10 is the center.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
        
    # Copies 5-9 correspond to 0-4.
    # Center 10.
    # Edges between original and copies:
    # If (u,v) is an edge in C5, then (v, u') and (u, v') are edges in Mycielski.
    # u' is 5+u, v' is 5+v.
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in orig_edges:
        edges.append([v, 5+u])
        edges.append([u, 5+v])
        
    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5+i])
        
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    # Check if adding (u,v) creates a triangle.
    # This happens if there is a common neighbor w.
    return (adj[u] & adj[v]) == 0

def _is_3_colorable(adj):
    """
    Fast backtracking to check if graph is 3-colorable.
    Uses bitmasks for speed.
    """
    # Precompute neighbor bitmasks
    colors = [-1] * N
    # Order vertices by degree descending for efficiency
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i
    
    # For each vertex, precompute which colors are forbidden by already-colored neighbors
    # We'll use a different approach: assign colors in order, track forbidden colors
    
    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        
        # Find which colors are forbidden by colored neighbors
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:  # Already colored
                forbidden |= (1 << colors[u])
        
        # Try colors 0, 1, 2 that are not forbidden
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def _max_independent_set_branch_and_bound(adj, deadline, rng, min_edges=0):
    """
    Branch and bound to find maximum edge count triangle-free graph
    that is NOT 3-colorable, starting from adj.
    """
    best_cnt = min_edges
    best_adj = None
    
    # Get all possible edges
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    # Sort edges by some heuristic (e.g., sum of degrees)
    def edge_score(e):
        u, v = e
        return bin(adj[u]).count('1') + bin(adj[v]).count('1')
    
    # We'll use a randomized local search with occasional restarts
    # combined with branch-and-bound on promising candidates
    
    current_adj = [a for a in adj]
    current_cnt = _edge_count(current_adj)
    
    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
        best_cnt = current_cnt
        best_adj = [a for a in current_adj]
    
    # Local search phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if not ((current_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_cnt += 1
                improved = True
                
                if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
                break
        
        if not improved:
            # Try edge swaps: remove one edge, add two
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (current_adj[u] >> v) & 1]
            if not existing:
                break
            
            best_swap = None
            best_swap_cnt = current_cnt
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                rem_u, rem_v = rng.choice(existing)
                
                adj_copy = [a for a in current_adj]
                adj_copy[rem_u] &= ~(1 << rem_v)
                adj_copy[rem_v] &= ~(1 << rem_u)
                
                # Try to add edges greedily
                for u in range(N):
                    for v in range(u + 1, N):
                        if not ((adj_copy[u] >> v) & 1) and _is_triangle_free_add(adj_copy, u, v):
                            adj_copy[u] |= (1 << v)
                            adj_copy[v] |= (1 << u)
                
                cnt = _edge_count(adj_copy)
                if cnt > best_swap_cnt and not _is_3_colorable(adj_copy):
                    best_swap_cnt = cnt
                    best_swap = adj_copy
            
            if best_swap:
                current_adj = best_swap
                current_cnt = best_swap_cnt
                improved = True
                if current_cnt > best_cnt:
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
    
    return best_adj, best_cnt

def construct():
    start = time.time()
    deadline = start + 140.0
    
    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # Multiple trials with different strategies
    for trial in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Start with Grötzsch graph
        adj = [a for a in base_adj]
        
        # Randomly perturb: add/remove some edges from the base
        if trial > 0 and trial % 5 == 0:
            # Try removing a random edge from base and see if we can add more
            base_edge_list = base_edges[:]
            if base_edge_list:
                idx = rng.randint(0, len(base_edge_list) - 1)
                u, v = base_edge_list[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Run branch and bound / local search
        result_adj, result_cnt = _max_independent_set_branch_and_bound(adj, deadline, rng, best_cnt)
        
        if result_adj and result_cnt > best_cnt:
            best_cnt = result_cnt
            best_adj = result_adj
    
    # Fallback
    if best_adj is None:
        best_adj = base_adj
        best_cnt = _edge_count(best_adj)
    
    # Final verification
    if _is_3_colorable(best_adj):
        best_adj = base_adj
        best_cnt = _edge_count(best_adj)
    
    # Convert to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
