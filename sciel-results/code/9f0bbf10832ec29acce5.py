"""K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import random
import time

N = 18
DEADLINE = 140  # 20s safety margin before 160s limit

def _to_edges(adj):
    """Convert adjacency matrix (list of int masks) to edge list."""
    edges = []
    for u in range(N):
        mask = adj[u]
        v = u + 1
        while v < N:
            if mask & (1 << v):
                edges.append([u, v])
            v += 1
    return edges

def _is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    for a in range(N):
        for b in range(a+1, N):
            if not (adj[a] & (1 << b)):
                continue
            # Common neighbors of a and b
            common = adj[a] & adj[b]
            # Check if any two common neighbors are adjacent
            c = 0
            while c < N:
                if common & (1 << c):
                    d = c + 1
                    while d < N:
                        if common & (1 << d):
                            if adj[c] & (1 << d):
                                return False
                        d += 1
                c += 1
    return True

def _largest_triangle_free(adj):
    """Find the largest subset of vertices inducing a triangle-free subgraph."""
    best = 0
    for mask in range(1 << N):
        # Quick check: if popcount <= best, skip
        if bin(mask).count('1') <= best:
            continue
        # Check if induced subgraph is triangle-free
        is_tf = True
        for i in range(N):
            if not (mask & (1 << i)):
                continue
            for j in range(i+1, N):
                if not (mask & (1 << j)):
                    continue
                if not (adj[i] & (1 << j)):
                    continue
                # i and j are adjacent, check for common neighbor k in mask
                common = adj[i] & adj[j] & mask
                if common:
                    is_tf = False
                    break
            if not is_tf:
                break
        if is_tf:
            best = bin(mask).count('1')
    return best

def _construct_turan():
    """Complete tripartite graph K6,6,6."""
    parts = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
    adj = [0] * N
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    return adj

def construct():
    start_time = time.time()
    
    # Start with Turan graph T_3(18)
    best_adj = _construct_turan()
    best_score = _largest_triangle_free(best_adj)
    best_edges = _to_edges(best_adj)
    
    # Verify K4-free
    assert _is_k4_free(best_adj)
    
    # Try random perturbations
    rng = random.Random(42)
    
    for attempt in range(20000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Start from current best and make a small random change
        adj = best_adj[:]
        
        # Randomly flip a few edges
        n_flips = rng.randint(1, 5)
        for _ in range(n_flips):
            u = rng.randint(0, N-1)
            v = rng.randint(0, N-1)
            if u == v:
                continue
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        
        # Check K4-free
        if not _is_k4_free(adj):
            continue
        
        # Score
        score = _largest_triangle_free(adj)
        if score < best_score:
            best_score = score
            best_adj = adj
            best_edges = _to_edges(adj)
    
    return best_edges
