import time
import random
import numpy as np

N = 40
DEADLINE = 115  # Wall-clock seconds (safety margin under 120s/150s limit)

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    A C6 is formed if there is a path of length 5 between u and v:
    u - a - b - c - d - v
    with all vertices distinct.
    """
    nu = adj[u]
    nv = adj[v]

    # Iterate a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # Iterate d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            # b in N(a), c in N(d), b,c distinct from {u,v,a,d}
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Any b in b_mask connected to any c in c_mask?
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

def _systematic_polish(adj, deadline):
    """
    Iteratively add any valid missing edge until no more can be added.
    Returns the number of edges added.
    """
    all_edges = _make_all_edges()
    added_any = True
    total_added = 0
    while added_any and time.time() < deadline:
        added_any = False
        # Shuffle to avoid bias towards low indices
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            added_any = True
            total_added += 1
    return total_added

def _k5_seed_adj(rng):
    """
    Build a C6-free seed from 8 disjoint K5's.
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

def _random_greedy_seed(adj, rng, deadline):
    """
    Add random edges greedily.
    """
    all_edges = _make_all_edges()
    rng.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _destroy_repair(adj, rng, deadline, destroy_size=10):
    """
    Remove 'destroy_size' random edges, then greedily repair by adding edges.
    Returns True if the resulting graph has more edges than the original.
    """
    original_count = _count_edges(adj)
    
    # Destroy
    current_edges = []
    for i in range(N):
        m = adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                current_edges.append((i, j))
            m ^= lsb
            
    if len(current_edges) < destroy_size:
        return False

    edges_to_remove = rng.sample(current_edges, destroy_size)
    for u, v in edges_to_remove:
        _remove_edge(adj, u, v)
        
    # Repair
    _systematic_polish(adj, deadline)
    
    new_count = _count_edges(adj)
    if new_count > original_count:
        return True
    else:
        # Revert if not improved (optional, but helps keep state clean if we want to backtrack)
        # However, ILS usually accepts moves that don't strictly improve if they are neutral, 
        # but here we only return True if improved. If not improved, the caller can decide to revert.
        # For simplicity, let's revert if not improved to maintain a high-quality baseline.
        # Re-adding the removed edges is tricky if they were re-added in repair.
        # So, a better strategy: Keep the new state if it's not worse, or revert if worse.
        # Actually, standard ILS accepts worse moves with probability. 
        # Let's just return the count difference and let the caller decide.
        return False

def _construct():
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

    # Phase 1: Multiple Seeds
    # Seed 1: K5 blocks
    adj1 = _k5_seed_adj(rng)
    _systematic_polish(adj1, deadline)
    refresh_best(adj1)
    
    # Seed 2: Random Greedy
    adj2 = [0] * N
    _random_greedy_seed(adj2, rng, deadline)
    _systematic_polish(adj2, deadline)
    refresh_best(adj2)
    
    # Seed 3: Another Random Greedy
    adj3 = [0] * N
    _random_greedy_seed(adj3, rng, deadline)
    _systematic_polish(adj3, deadline)
    refresh_best(adj3)
    
    # Phase 2: Iterated Local Search (Destroy and Repair)
    # We will run ILS on the best graph found so far.
    current_adj = best_adj[:]
    
    # We can also run ILS on other good seeds if time permits, but focusing on the best is usually effective.
    # Let's run ILS on the current best.
    
    while time.time() < deadline - 5:  # Leave 5s for final polish and safety
        # Destroy
        # Vary destroy size slightly
        destroy_size = rng.randint(5, 15)
        
        # Get current edges
        current_edges = []
        for i in range(N):
            m = current_adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    current_edges.append((i, j))
                m ^= lsb
        
        if len(current_edges) < destroy_size:
            break
            
        edges_to_remove = rng.sample(current_edges, destroy_size)
        
        # Save state before destroy to allow revert if needed
        # But we will just repair and check if it's better.
        # If not better, we can revert to the previous best state.
        
        for u, v in edges_to_remove:
            _remove_edge(current_adj, u, v)
            
        # Repair
        _systematic_polish(current_adj, deadline)
        
        new_count = _count_edges(current_adj)
        
        if new_count > best_count:
            refresh_best(current_adj)
        else:
            # If not improved, revert to the best known state to avoid drifting to a worse local optimum
            # This makes it a "best-improvement" ILS variant, which is safe.
            # Alternatively, we could accept neutral moves with probability.
            # Let's revert to best_adj to stay at the peak.
            current_adj = best_adj[:]
            
    # Final Polish
    _systematic_polish(best_adj, deadline)
    refresh_best(best_adj)
    
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

def construct():
    return _construct()

if __name__ == "__main__":
    construct()
