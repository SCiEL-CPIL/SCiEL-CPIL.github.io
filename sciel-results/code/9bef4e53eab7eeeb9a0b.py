"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Combines algebraic construction (Paley-type) with fast bitmask validation
and local search for n=21 extension.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 135  # 15s safety margin before 150s kill
    
    best_n = 0
    best_edges = []
    
    # First: try to find a valid n=20 graph quickly
    # Use a mix of algebraic constructions and random search
    
    # Try Paley-type graphs and other structured constructions for n=20
    found_20 = False
    for seed in range(200):
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(seed)
        adj = _generate_candidate(20, rng, seed)
        
        if _is_valid_fast(adj, 20, deadline):
            best_n = 20
            best_edges = _adj_to_edges(adj, 20)
            found_20 = True
            break
    
    # If we didn't find n=20 with structured approach, try pure random
    if not found_20:
        for seed in range(1000):
            if time.time() > deadline - 30:
                break
            
            rng = random.Random(10000 + seed)
            adj = [0] * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if _is_valid_fast(adj, 20, deadline):
                best_n = 20
                best_edges = _adj_to_edges(adj, 20)
                found_20 = True
                break
    
    # If we have n=20, try to extend to n=21
    if best_n == 20 and time.time() < deadline - 20:
        result = _try_extend_to_21(best_edges, deadline)
        if result is not None:
            return result
    
    # Fallback: try smaller n if n=20 failed
    if best_n < 20:
        for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
            if time.time() > deadline - 5:
                break
            
            for seed in range(200):
                if time.time() > deadline - 5:
                    break
                
                rng = random.Random(seed + target_n * 1000)
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if rng.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if _is_valid_fast(adj, target_n, deadline):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = _adj_to_edges(adj, target_n)
                    break
            
            if best_n >= target_n:
                break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Last resort: trivially valid small graph
    return {'n': 5, 'edges': []}


def _generate_candidate(n, rng, seed):
    """
    Generate a candidate graph using a mix of structured and random approaches.
    For n=20, try different construction strategies.
    """
    strategy = seed % 4
    
    if strategy == 0:
        # Pure random
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    elif strategy == 1:
        # Paley-type: use quadratic residues mod p where p is prime >= n
        # Find smallest prime >= n
        p = _next_prime(n)
        adj = [0] * n
        # Compute quadratic residues mod p
        qr = set()
        for x in range(p):
            qr.add((x * x) % p)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Edge if (j - i) mod p is a quadratic residue
                diff = (j - i) % p
                if diff in qr and diff != 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    elif strategy == 2:
        # Turan-type: partition into groups, connect between groups randomly
        adj = [0] * n
        group_size = max(1, n // 4)
        groups = []
        for i in range(n):
            group_idx = i // group_size
            if not groups or len(groups[-1]) >= group_size:
                groups.append([])
            groups[-1].append(i)
        
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                for u in groups[gi]:
                    for v in groups[gj]:
                        if rng.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj
    
    else:
        # Random with degree constraint: aim for roughly regular
        adj = [0] * n
        target_deg = n // 2
        for i in range(n):
            # Randomly pick ~target_deg neighbors
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            for j in candidates[:target_deg]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj


def _next_prime(n):
    """Find the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while True:
        if all(candidate % i != 0 for i in range(2, int(candidate**0.5) + 1)):
            return candidate
        candidate += 2


def _is_valid_fast(adj, n, deadline):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Returns True if valid, False otherwise.
    Uses early termination and bit_count for speed.
    """
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            # Time check
            if time.time() > deadline:
                return False
            
            # Compute mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check regularity
            v0 = subset[0]
            d0 = (adj[v0] & mask).bit_count()
            
            is_regular = True
            for v in subset[1:]:
                dv = (adj[v] & mask).bit_count()
                if dv != d0:
                    is_regular = False
                    break
            
            if is_regular:
                return False
    
    return True


def _adj_to_edges(adj, n):
    """Convert adjacency bitmask list to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges


def _try_extend_to_21(base_edges, deadline):
    """
    Try to extend a valid n=20 graph to n=21 by adding a vertex
    and finding appropriate edges.
    Uses local search.
    """
    n_old = 20
    n_new = 21
    rng = random.Random(999)
    
    # Build adjacency from base edges
    adj = [0] * n_old
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Start: add vertex 20 with random edges to existing vertices
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random edges from new vertex to old
        new_adj = [0] * n_new
        # Copy old adjacency
        for i in range(n_old):
            new_adj[i] = adj[i]
        
        # Random connections from vertex 20 to others
        for i in range(n_old):
            if rng.random() < 0.5:
                new_adj[i] |= (1 << n_old)
                new_adj[n_old] |= (1 << i)
        
        if _is_valid_fast(new_adj, n_new, deadline):
            # Success! Convert to edge list
            edges = _adj_to_edges(new_adj, n_new)
            return {'n': n_new, 'edges': edges}
        
        # If not valid, try local search: flip edges involving new vertex
        # Find a violation and adjust
        if attempt % 10 == 0:
            # Occasionally try a fresh random assignment
            continue
        
        # Flip a random edge involving vertex 20
        v = rng.randint(0, n_old - 1)
        if new_adj[n_old] & (1 << v):
            new_adj[n_old] &= ~(1 << v)
            new_adj[v] &= ~(1 << n_old)
        else:
            new_adj[n_old] |= (1 << v)
            new_adj[v] |= (1 << n_old)
    
    # Alternative: generate fresh random n=21 graphs
    if time.time() < deadline:
        for seed in range(300):
            if time.time() > deadline:
                break
            r = random.Random(seed + 7777)
            adj21 = [0] * n_new
            for i in range(n_new):
                for j in range(i + 1, n_new):
                    if r.random() < 0.5:
                        adj21[i] |= (1 << j)
                        adj21[j] |= (1 << i)
            
            if _is_valid_fast(adj21, n_new, deadline):
                return {'n': n_new, 'edges': _adj_to_edges(adj21, n_new)}
    
    return None
