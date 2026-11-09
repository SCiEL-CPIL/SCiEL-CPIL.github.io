import time
import random
import math

def _greedy_independent_set_size(adj, n):
    """
    Fast greedy heuristic for independence number.
    Returns the size of the independent set found.
    """
    available = (1 << n) - 1
    count = 0
    while available:
        # Pick vertex with minimum degree in the induced subgraph 'available'
        # to maximize the number of vertices we can keep.
        best_v = -1
        min_deg = n
        temp_avail = available
        while temp_avail:
            v_bit = temp_avail & -temp_avail
            v = v_bit.bit_length() - 1
            # Count neighbors in available
            deg = bin(adj[v] & available).count('1')
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp_avail &= temp_avail - 1
        
        if best_v == -1:
            break
            
        count += 1
        # Remove best_v and its neighbors from available
        available &= ~(adj[best_v] | (1 << best_v))
        
    return count

def _has_independent_set_10(adj, n):
    """
    Exact check for independent set of size 10.
    """
    def dfs(available_mask, size):
        if size >= 10:
            return True
        if available_mask == 0:
            return False
        
        # Pruning: if remaining vertices + current size < 10, fail
        if bin(available_mask).count('1') < 10 - size:
            return False
            
        v_bit = available_mask & -available_mask
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_avail = available_mask & ~(adj[v] | v_bit)
        if dfs(new_avail, size + 1):
            return True
            
        # Branch 2: Exclude v
        new_avail = available_mask & ~v_bit
        if dfs(new_avail, size):
            return True
            
        return False

    return dfs((1 << n) - 1, 0)

def _polarity_graph_pq2(q):
    """
    Construct the polarity graph of PG(2, q).
    For q odd, this is a triangle-free graph on q^2 + q + 1 vertices
    with independence number at most q + sqrt(q) + 1.
    For q = 9: n = 91, alpha <= 9 + 3 + 1 = 13 (too large)
    For q = 5: n = 31, alpha <= 5 + sqrt(5) + 1 ≈ 8.23 -> alpha <= 8
    For q = 7: n = 57, alpha <= 7 + sqrt(7) + 1 ≈ 10.65 -> alpha <= 10 (borderline)
    
    We'll use q=5 to get n=31 with alpha <= 8, which satisfies alpha < 10.
    """
    n = q * q + q + 1
    adj = [0] * n
    
    # PG(2, q) has q^2 + q + 1 points
    # We can represent points as (x, y, 1) for affine points and (1, m, 0) for points at infinity
    # Actually, let's use a simpler representation:
    # Points are indexed 0 to n-1
    # We'll use the standard construction where two points are adjacent if one lies on the polar of the other
    
    # For simplicity, let's use a different approach:
    # The polarity graph of PG(2, q) for q odd has the property that
    # we can construct it using a symmetric bilinear form
    
    # Let's use the standard construction:
    # Points: (0, 0, 1), (1, 0, 0), and (x, 1, 0) for x in GF(q)
    # Actually, let's just use a known construction for q=5
    
    # For q=5, we can use the following:
    # The graph has 31 vertices
    # We'll use a systematic construction
    
    # Alternative: Use the complement of a strongly regular graph
    # The polarity graph of PG(2, q) for q odd is a strongly regular graph
    # with parameters (v, k, lambda, mu) where:
    # v = q^2 + q + 1
    # k = q(q+1)/2
    # lambda = (q-3)/2
    # mu = q(q-1)/4
    
    # For q=5: v=31, k=15, lambda=1, mu=5
    # This is the complement of the Petersen graph? No, that's (10,3,0,1)
    
    # Let's just construct it directly using finite field arithmetic
    
    # GF(5) elements: 0, 1, 2, 3, 4
    # Points in PG(2, 5):
    # Type 1: (x, y, 1) for x,y in GF(5) -> 25 points
    # Type 2: (1, m, 0) for m in GF(5) -> 5 points
    # Type 3: (0, 1, 0) -> 1 point
    # Total: 31 points
    
    # A polarity is defined by a symmetric bilinear form B(p, q)
    # Two points p, q are adjacent if B(p, q) = 0 and p != q
    # For the standard polarity, B((x1,y1,z1), (x2,y2,z2)) = x1*x2 + y1*y2 + z1*z2 (mod q)
    # But this is not quite right for projective space
    
    # Let's use a different approach: since we know the parameters,
    # we can use a known strongly regular graph with these parameters
    
    # Actually, let's just use a simple random construction and verify
    # For now, let's return a random maximal triangle-free graph
    # and improve it with local search
    
    return None  # We'll handle this in construct()

def _random_maximal_triangle_free(n, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    adj = [0] * n
    edges_to_add = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_add)
    
    for u, v in edges_to_add:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _local_search_reduce_alpha(adj, n, rng, max_steps=5000, deadline=None):
    """
    Local search to reduce the independence number.
    Returns (adj, current_alpha)
    """
    current_alpha = _greedy_independent_set_size(adj, n)
    
    T = 1.0
    cooling = 0.995
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        # Generate a move
        move_type = rng.random()
        new_adj = adj[:]  # Copy for evaluation
        
        if move_type < 0.6:
            # Try to add a random valid edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
                
            if not (adj[u] & (1 << v)):
                # Check if adding (u,v) creates a triangle
                common = adj[u] & adj[v]
                if common == 0:
                    # Valid addition
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                else:
                    # Invalid, skip move
                    continue
            else:
                # Edge exists, skip
                continue
        else:
            # Try to remove a random edge
            u = rng.randint(0, n - 1)
            # Pick random v
            v = rng.randint(0, n - 1)
            if u != v and (adj[u] & (1 << v)):
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                continue

        # Evaluate new state
        new_alpha = _greedy_independent_set_size(new_adj, n)
        
        # Acceptance criteria
        delta = new_alpha - current_alpha
        if delta <= 0:
            # Better or equal
            adj = new_adj
            current_alpha = new_alpha
        else:
            # Worse, accept with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / T):
                adj = new_adj
                current_alpha = new_alpha
        
        T *= cooling
    
    return adj, current_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    # We want to maximize n. Try from high to low.
    # R(3,10) <= 41, so max n is 40.
    # Known best is 39. Seed is 26.
    # Let's try to push for 30-35.
    
    for n in range(40, 25, -1):
        if time.time() > deadline - 2:
            break
            
        found = False
        restarts = 0
        max_restarts = 30  # Reduced to allow more time per restart
        
        while restarts < max_restarts and not found:
            restarts += 1
            if time.time() > deadline - 2:
                break
                
            # Use deterministic construction for some seeds, random for others
            if restarts % 5 == 0:
                # Try a more structured construction
                # Use a blow-up or other deterministic method
                # For now, use random with a specific seed pattern
                seed = restarts * 12345 + n * 67890
            else:
                seed = restarts * 7 + n
            
            rng = random.Random(seed)
            adj = _random_maximal_triangle_free(n, seed=seed)
            
            # Local search
            adj, current_alpha = _local_search_reduce_alpha(adj, n, rng, 
                                                           max_steps=3000,
                                                           deadline=deadline - 2)
            
            # If we already have alpha < 10, check exactly
            if current_alpha < 10:
                if not _has_independent_set_10(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: Return a known valid graph if we didn't find a better one
    # The seed n=26 graph is valid.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    return {'n': n_seed, 'edges': edges_seed}
