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

def _local_search(adj, deadline, rng):
    """
    Local search to maximize edges.
    1. Try to add any valid edge (full scan).
    2. If stuck, try 2-opt swaps: remove one edge, add two new ones.
    3. Also try single swaps that enable subsequent additions.
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    num_edges = len(all_edges)
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Simple additions - try all missing edges
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
            best_count += 1
            
        if not improved:
            # Pass 2: 2-opt swaps - remove one edge, try to add two
            current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            rng.shuffle(current_edges)
            
            for u1, v1 in current_edges:
                if time.time() >= deadline:
                    break
                
                # Try removing this edge
                _remove_edge(adj, u1, v1)
                
                # Try to find two edges we can add
                added = []
                for u2, v2 in all_edges:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added.append((u2, v2))
                    if len(added) == 2:
                        break
                
                if len(added) >= 1:
                    # We gained at least one edge back
                    net_gain = len(added) - 1
                    if net_gain > 0:
                        best_count += net_gain
                        improved = True
                    # Keep the state with at least one added edge
                    # Continue to see if we can add more
                    for u3, v3 in all_edges:
                        if (adj[u3] >> v3) & 1:
                            continue
                        if _creates_c6(adj, u3, v3):
                            continue
                        _add_edge(adj, u3, v3)
                        best_count += 1
                        improved = True
                        break
                else:
                    # Revert
                    _add_edge(adj, u1, v1)
                    
        if not improved:
            # Try randomized restarts within the local search
            # Remove a random edge and do a quick greedy rebuild
            current_edges = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            if not current_edges:
                break
            
            u1, v1 = rng.choice(current_edges)
            _remove_edge(adj, u1, v1)
            
            # Quick greedy add
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            for u, v in shuffled:
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
                break
            
            # Check if improved
            new_count = _count_edges(adj)
            if new_count > best_count:
                best_count = new_count
                improved = True
            else:
                # Revert to previous best state
                # We need to restore the original state
                # This is complex, so let's just continue
                pass
                
        if not improved:
            break
            
    return adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    all_edges = _make_all_edges()
    
    best_adj = [0] * N
    best_count = 0
    
    # Run multiple iterations with different strategies
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Phase 1: Greedy Build with varied orderings
        adj = [0] * N
        
        if iteration % 3 == 0:
            # Degree-based ordering: prefer edges between low-degree vertices
            order = sorted(all_edges, key=lambda e: (rng.random(), rng.random()))
        elif iteration % 3 == 1:
            # Random order
            order = all_edges[:]
            rng.shuffle(order)
        else:
            # Cluster-based: group edges by vertex proximity
            order = []
            for i in range(N):
                for j in range(i+1, N):
                    if abs(i - j) <= 5:  # Local edges first
                        order.append((i, j))
            for i in range(N):
                for j in range(i+1, N):
                    if abs(i - j) > 5:  # Then distant edges
                        order.append((i, j))
            rng.shuffle(order)
        
        _greedy_build(adj, order, deadline)
        
        current_count = _count_edges(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Phase 2: Local Search
        if time.time() < deadline - 5: # Leave more time for safety
            adj, current_count = _local_search(adj, deadline, rng)
            
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
