import time
import random
from itertools import combinations

N = 18
DEADLINE = 115.0  # safety margin under 150s wall-clock limit


def _is_k4_free(adj):
    """Check if graph is K4-free using bitmask adjacency."""
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            # Check if common neighborhood has any edge (would form K4)
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if k > j:
                    # k is a common neighbor; check if k is adjacent to another common neighbor
                    c2 = common & ~(1 << k)
                    while c2:
                        lsb2 = c2 & (-c2)
                        m = lsb2.bit_length() - 1
                        c2 ^= lsb2
                        if m > k and (adj[k] & (1 << m)):
                            return False
    return True


def _max_triangle_free(adj):
    """Find the size of the largest triangle-free induced subgraph via branch and bound."""
    # Precompute all triangles as bitmasks
    triangles = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return N  # no triangles, entire graph is triangle-free
    
    # Order vertices by degree for better pruning
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1'))
    
    best = 0
    best_set = 0
    
    # Branch and bound: process vertices in order, include or exclude
    # State: (index in order, current_set, current_size)
    # Pruning: if current_size + remaining <= best, prune
    
    def search(idx, S, size):
        nonlocal best, best_set
        if size > best:
            best = size
            best_set = S
        
        if idx == N:
            return
        
        remaining = N - idx
        if size + remaining <= best:
            return
        
        v = order[idx]
        bit_v = 1 << v
        
        # Option 1: exclude v
        search(idx + 1, S, size)
        
        # Option 2: include v if S ∪ {v} is triangle-free
        # Check: for all pairs (a,b) in S with a~b, v is not adjacent to both
        # Equivalently: no triangle in S ∪ {v}
        can_add = True
        for tri in triangles:
            if (S & tri) == (tri & ~(bit_v)):
                # Two vertices of this triangle are in S, third is v
                if (tri & bit_v) and (S & (tri & ~bit_v)) == (tri & ~bit_v):
                    can_add = False
                    break
        
        if can_add:
            search(idx + 1, S | bit_v, size + 1)
    
    search(0, 0, 0)
    return best


def _random_k4_free(rng, density_hint=0.5):
    """Generate a random K4-free graph with approximately density_hint edge density."""
    adj = [0] * N
    edges_to_try = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(edges_to_try)
    
    for (u, v) in edges_to_try:
        # Only add edge with some probability to control density
        if rng.random() > density_hint:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _is_k4_free(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return adj


def _local_search(adj, rng, time_limit):
    """Perform local search to minimize max triangle-free induced subgraph."""
    current_score = _max_triangle_free(adj)
    best_adj = list(adj)
    best_score = current_score
    start_time = time.time()
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() - start_time < time_limit:
        # Try random edge flips
        improved = False
        for _ in range(50):  # try up to 50 random moves per iteration
            if time.time() - start_time >= time_limit:
                break
            
            u, v = rng.choice(all_edges)
            has_edge = bool(adj[u] & (1 << v))
            
            # Try flipping this edge
            if has_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check K4-freedom
            if not _is_k4_free(adj):
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                continue
            
            # Score the new graph
            new_score = _max_triangle_free(adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = list(adj)
                improved = True
                break
            else:
                # Revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        if not improved:
            # Stuck, try a more aggressive move: remove a random edge
            u, v = rng.choice(all_edges)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                if _is_k4_free(adj):
                    current_score = _max_triangle_free(adj)
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = list(adj)
    
    return best_adj, best_score


def construct():
    start_time = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_score = N + 1  # worst case
    
    # Multiple restarts with different initial graphs
    num_restarts = 0
    while time.time() - start_time < DEADLINE:
        num_restarts += 1
        
        # Generate initial graph with varying density
        density = 0.3 + 0.4 * (num_restarts % 5) / 5.0
        adj = _random_k4_free(rng, density)
        
        # Quick score
        score = _max_triangle_free(adj)
        
        if score < best_score:
            best_score = score
            best_adj = list(adj)
        
        # If we already have a good score, we can be more selective
        if best_score <= 8:
            # Do local search with remaining time
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 5:
                adj, score = _local_search(adj, rng, remaining - 2)
                if score < best_score:
                    best_score = score
                    best_adj = list(adj)
            break
        elif best_score <= 10:
            remaining = DEADLINE - (time.time() - start_time)
            if remaining > 10:
                adj, score = _local_search(adj, rng, remaining - 5)
                if score < best_score:
                    best_score = score
                    best_adj = list(adj)
        
        # Continue with more restarts if time permits
        if time.time() - start_time > DEADLINE - 20:
            break
    
    # Convert best adjacency to edge list
    if best_adj is None:
        # Fallback: return K6,6,6 structure
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = []
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        edges.append([u, v])
        return edges
    
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
