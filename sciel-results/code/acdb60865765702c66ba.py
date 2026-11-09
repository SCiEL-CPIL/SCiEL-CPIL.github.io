import time
import random
import math

N = 40
DEADLINE = 115  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    adj: list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We look for a path u-a-b-c-d-v
    # a is in N(u), d is in N(v)
    # b is in N(a), c is in N(d)
    # b-c is an edge
    # u, v, a, b, c, d are distinct
    
    forbidden_uv = (1 << u) | (1 << v)
    
    # Iterate over a in N(u)
    a_mask = nu
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        
        if a == v: continue
        
        # Iterate over d in N(v), d != u, d != a
        d_mask = nv & ~(1 << u) & ~(1 << a)
        
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            
            # Forbidden vertices for b and c: u, v, a, d
            forbidden = forbidden_uv | (1 << a) | (1 << d)
            
            # b in N(a) \ forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            
            # c in N(d) \ forbidden
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue
            
            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & c_mask:
                    return True
    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _k5_seed_adj():
    """
    Build a C6-free seed from 8 disjoint K5's (40 vertices).
    Returns adj list.
    """
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _greedy_fill(adj, all_edges, rng):
    """
    Add edges in random order if C6-free.
    Returns the updated adj and count.
    """
    current = adj[:]
    count = _count_edges(current)
    edges_to_try = all_edges[:]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if (current[u] >> v) & 1:
            continue
        if _creates_c6(current, u, v):
            continue
        _add_edge(current, u, v)
        count += 1
    return current, count

def _ils_search(start_adj, deadline, rng):
    """
    Iterated Local Search.
    """
    all_edges = _make_all_edges()
    current = start_adj[:]
    current_count = _count_edges(current)
    best_adj = current[:]
    best_count = current_count
    
    # Initial greedy fill
    current, current_count = _greedy_fill(current, all_edges, rng)
    if current_count > best_count:
        best_count = current_count
        best_adj = current[:]
        
    temperature = 5.0
    min_temperature = 0.1
    cooling_rate = 0.99
    
    # Pre-compute list of existing edges for perturbation
    def get_existing_edges(adj):
        edges = []
        for i in range(N):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append((i, j))
                m ^= lsb
        return edges
        
    current_edges = get_existing_edges(current)
    
    while time.time() < deadline:
        # Perturb: remove k random edges
        k = rng.randint(3, 10)
        if len(current_edges) < k:
            k = len(current_edges)
            
        # Select k random edges to remove
        to_remove = rng.sample(current_edges, k)
        
        perturbed = current[:]
        perturbed_count = current_count
        for u, v in to_remove:
            _remove_edge(perturbed, u, v)
            perturbed_count -= 1
            
        # Repair: Greedy fill
        new_adj, new_count = _greedy_fill(perturbed, all_edges, rng)
        
        # Acceptance criterion
        if new_count >= current_count:
            current = new_adj
            current_count = new_count
            current_edges = get_existing_edges(current)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = current[:]
        else:
            # Metropolis acceptance
            delta = current_count - new_count
            # If delta is small, accept with higher prob
            if rng.random() < math.exp(-delta / temperature):
                current = new_adj
                current_count = new_count
                current_edges = get_existing_edges(current)
                
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current[:]
                    
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature

    return best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    
    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + ILS
    seed = _k5_seed_adj()
    seed, c = _ils_search(seed, deadline, rng)
    refresh_best(seed)

    # Phase 2: Random restarts
    # Try a few different random seeds if time permits
    # But ILS is usually robust. Let's do a few quick restarts with different perturbations.
    
    # Actually, let's just run ILS for the remaining time on the best found so far, 
    # or restart if stuck.
    
    # A better strategy: Run multiple independent ILS instances if time allows, 
    # but ILS is sequential. 
    # Let's just let the single ILS run for the full time. 
    # The _ils_search function already loops until deadline.
    
    # Wait, the current code calls _local_search which has its own deadline loop.
    # My _ils_search also has a deadline loop.
    # I should ensure I don't exceed the deadline in construct().
    
    # Let's structure construct to manage the deadline globally.
    
    # Re-do construct to manage time properly.
    
    return None # Placeholder

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    
    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    all_edges = _make_all_edges()
    
    # Strategy: Run several iterations of (Seed -> Greedy -> ILS)
    # Or just one long ILS.
    
    # Start with K5 seed
    seed = _k5_seed_adj()
    # Quick greedy fill
    seed, c = _greedy_fill(seed, all_edges, rng)
    refresh_best(seed)
    
    # Now run ILS on this seed
    # The ILS function will run until its internal deadline.
    # We pass the global deadline.
    
    # However, ILS might get stuck. 
    # Let's add a "restart" mechanism inside the main loop if no improvement for X seconds.
    
    # For simplicity and robustness, let's just run the ILS.
    
    final_adj, final_count = _ils_search(seed, deadline, rng)
    refresh_best(final_adj)

    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
