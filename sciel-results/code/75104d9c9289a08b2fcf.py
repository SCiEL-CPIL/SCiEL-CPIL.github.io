import time
import random
import sys

N = 40
DEADLINE = 110  # Safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    Path: u - a - b - c - d - v
    """
    nu = adj[u]
    nv = adj[v]

    # a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << d)

            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask (b != c)
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                # c must be in c_mask and not b
                if adj[b] & (c_mask & ~(1 << b)):
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

def _greedy_build(adj, order, deadline):
    """
    Adds edges from order if possible. Returns count added.
    """
    count = 0
    for u, v in order:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count

def _k5_seed_adj():
    """
    Create a K5-free seed: partition vertices into 8 groups of 5,
    connect all within each group (K5), plus some cross-group edges.
    K5 is C6-free, and we can add cross edges carefully.
    """
    adj = [0] * N
    # 8 groups of 5 vertices: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39
    for g in range(8):
        base = g * 5
        for i in range(5):
            for j in range(i + 1, 5):
                _add_edge(adj, base + i, base + j)
    return adj

def _anneal_from(adj, deadline, rng):
    """
    Simulated annealing to maximize edges while maintaining C6-free property.
    Moves: 
    1. Add a random edge if valid
    2. Remove a random edge, then try to add 1-2 edges
    3. Swap: remove edge e1, add edge e2 if valid
    
    Energy = -num_edges (we want to minimize energy = maximize edges)
    """
    current_count = _count_edges(adj)
    best_count = current_count
    best_adj = adj[:]
    
    all_edges = _make_all_edges()
    T0 = 50.0
    Tend = 0.5
    cooling_rate = 0.999
    
    T = T0
    steps = 0
    
    while time.time() < deadline and T > Tend:
        steps += 1
        if steps % 10000 == 0:
            # Check if we should cool down
            pass
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random edge
            u, v = rng.choice(all_edges)
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            new_count = current_count + 1
            # Always accept improvements
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
        elif move_type < 0.7:
            # Try a swap: remove one edge, add another
            # Pick a random existing edge to remove
            existing = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    existing.append((u, v))
            if not existing:
                continue
            
            u1, v1 = rng.choice(existing)
            _remove_edge(adj, u1, v1)
            
            # Try to add a different edge
            improved = False
            for _ in range(20):
                u2, v2 = rng.choice(all_edges)
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                improved = True
                break
            
            if improved:
                # We're at the same count. Try to add more edges greedily
                for u, v in all_edges:
                    if (adj[u] >> v) & 1:
                        continue
                    if _creates_c6(adj, u, v):
                        continue
                    _add_edge(adj, u, v)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    break  # Just one more step to save time
            else:
                # Revert
                _add_edge(adj, u1, v1)
        
        else:
            # Try to remove an edge and add two (net gain)
            existing = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    existing.append((u, v))
            if not existing:
                continue
            
            u1, v1 = rng.choice(existing)
            _remove_edge(adj, u1, v1)
            
            added = 0
            for _ in range(30):
                u2, v2 = rng.choice(all_edges)
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added += 1
                if added >= 2:
                    break
            
            if added > 1:
                current_count += added - 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert everything we added
                # This is tricky - we need to track what we added
                # For simplicity, just revert the removal and keep what we added
                # Actually, let's just accept if added >= 1 (neutral or better)
                if added == 1:
                    current_count = current_count  # No change
                else:
                    _add_edge(adj, u1, v1)
                    current_count = current_count + 1  # We removed and didn't add back properly
        
        # Cool down
        T *= cooling_rate
        
        # If we've hit a good count, we can speed up
        if best_count >= 100:
            cooling_rate = 0.998
    
    return best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    all_edges = _make_all_edges()
    
    best_adj = [0] * N
    best_count = 0
    
    # Run multiple iterations with different strategies
    while time.time() < deadline:
        # Phase 1: Try different initial structures
        strategy = rng.random()
        
        if strategy < 0.3:
            # K5-based seed
            adj = _k5_seed_adj()
        elif strategy < 0.6:
            # Random greedy from scratch
            adj = [0] * N
            order = all_edges[:]
            rng.shuffle(order)
            _greedy_build(adj, order, deadline)
        else:
            # Start from best found so far (if we have one)
            if best_count > 0:
                adj = best_adj[:]
            else:
                adj = [0] * N
                order = all_edges[:]
                rng.shuffle(order)
                _greedy_build(adj, order, deadline)
        
        current_count = _count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Phase 2: Simulated annealing
        if time.time() < deadline - 3:
            adj, current_count = _anneal_from(adj, deadline - 2, rng)
            
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
        
        # Early exit if we found a very high count
        if best_count >= 105:
            break
    
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
