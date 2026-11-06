import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try n=22, 21, 20 in order
    for n in [22, 21, 20]:
        if time.time() > deadline:
            break
        result = find_valid_graph(n, deadline)
        if result is not None:
            return result
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def find_valid_graph(n, deadline):
    """Find a graph on n vertices with no regular induced subgraph of size >= 6."""
    best_penalty = float('inf')
    best_adj = None
    
    rng = random.Random()
    
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Initialize: union of random perfect matchings for even n, or near-perfect for odd
        adj = init_from_matchings(n, rng)
        
        penalty = count_violations_fast(adj, n)
        
        if penalty == 0:
            return format_graph(adj, n)
        
        # Local search with simulated annealing
        current_adj = adj[:]
        current_penalty = penalty
        temp = 2.0
        cooling = 0.997
        min_temp = 0.01
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Pick random edge to flip
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            edge_exists = bool(current_adj[u] & (1 << v))
            if edge_exists:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_penalty = count_violations_fast(current_adj, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty == 0:
                    return format_graph(current_adj, n)
            else:
                delta = new_penalty - current_penalty
                if temp > min_temp and rng.random() < pow(2.718, -delta / temp):
                    pass  # Accept worse solution
                else:
                    # Revert
                    if edge_exists:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                    else:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
            
            temp *= cooling
        
        if current_penalty < best_penalty:
            best_penalty = current_penalty
            best_adj = current_adj[:]
        
        # Diversify for next restart
        if best_adj is not None and restart < 29:
            adj = best_adj[:]
            # Flip a few random edges to diversify
            for _ in range(3):
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
    
    return None

def init_from_matchings(n, rng):
    """Initialize graph as union of random perfect matchings."""
    adj = [0] * n
    
    if n % 2 == 0:
        # Generate a random 1-factorization of K_n
        # For even n, we can create n-1 perfect matchings
        # Simple approach: for each round r, pair vertex i with vertex (i + r) % n
        # But this gives a specific factorization. Let's randomize.
        
        # Create random permutation for each matching
        vertices = list(range(n))
        for r in range(n - 1):
            # Shuffle vertices for this matching
            shuffled = vertices[:]
            rng.shuffle(shuffled)
            # Pair them up: (shuffled[0], shuffled[1]), (shuffled[2], shuffled[3]), ...
            for i in range(0, n, 2):
                u = shuffled[i]
                v = shuffled[i + 1]
                # With some probability, include this edge
                if rng.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    else:
        # For odd n, use near-perfect matchings
        # Leave one vertex out each time
        vertices = list(range(n))
        for r in range(n - 1):
            # Choose which vertex to leave out
            left_out = r % n
            shuffled = [v for v in vertices if v != left_out]
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled), 2):
                if i + 1 < len(shuffled):
                    u = shuffled[i]
                    v = shuffled[i + 1]
                    if rng.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    
    return adj

def count_violations_fast(adj, n):
    """Count regular induced subgraphs of size >= 6. Focus on size 6 first."""
    penalty = 0
    vertices = list(range(n))
    
    # Check size 6
    for combo in itertools.combinations(vertices, 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        d0 = (adj[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            penalty += 1
    
    # If no size-6 violations, check larger sizes
    if penalty == 0:
        for k in range(7, n + 1):
            for combo in itertools.combinations(vertices, k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                d0 = (adj[combo[0]] & mask).bit_count()
                is_regular = True
                for v in combo[1:]:
                    if (adj[v] & mask).bit_count() != d0:
                        is_regular = False
                        break
                if is_regular:
                    penalty += 1
                    break  # No need to count more for this k
            if penalty > 0:
                break
    
    return penalty

def format_graph(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
