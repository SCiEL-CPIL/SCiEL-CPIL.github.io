import time
import random
import math

N = 15
DEADLINE = 58.0  # Safety margin

def _check_valid(adj, deg):
    """
    Check if the graph defined by adjacency bitmasks and degrees is valid.
    Valid means: No simple 3-edge path s-u-v-t has deg(s) == deg(t).
    """
    for u in range(N):
        au = adj[u]
        for v in range(u + 1, N):
            if not (au >> v) & 1:
                continue
            av = adj[v]
            # s is neighbor of u, s != v
            # t is neighbor of v, t != u, t != s
            # We iterate s and t
            # Optimization: group by degree?
            # For now, simple nested loop with bitmask iteration
            
            # Get list of neighbors for u (excluding v) and v (excluding u)
            # Using bit manipulation to iterate set bits
            su_mask = au & ~(1 << v)
            tv_mask = av & ~(1 << u)
            
            # Iterate s
            s_mask = su_mask
            while s_mask:
                s_bit = s_mask & -s_mask
                s = s_bit.bit_length() - 1
                s_mask ^= s_bit
                ds = deg[s]
                
                # Iterate t
                t_mask = tv_mask & ~(1 << s) # t != s
                while t_mask:
                    t_bit = t_mask & -t_mask
                    t = t_bit.bit_length() - 1
                    t_mask ^= t_bit
                    if ds == deg[t]:
                        return False
    return True

def _count_edges(adj):
    count = 0
    for i in range(N):
        count += bin(adj[i]).count('1')
    return count // 2

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_adj = [0] * N
    best_deg = [0] * N
    best_edges = 0
    
    # Initialize with a random dense graph
    # Or start with K_15 and remove edges?
    # Start with random p=0.8
    adj = [0] * N
    deg = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < 0.8:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
    
    # Ensure we start with a valid graph if possible, or just start SA
    # If invalid, SA will handle it by rejecting invalid moves
    # But we need a valid starting point to measure "current"
    # Let's try to find a valid start quickly
    # Try the K_1,14 star
    for i in range(N):
        adj[i] = 0
        deg[i] = 0
    for i in range(1, N):
        adj[0] |= (1 << i)
        adj[i] |= (1 << 0)
        deg[0] += 1
        deg[i] += 1
    best_adj = adj[:]
    best_deg = deg[:]
    best_edges = _count_edges(adj)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    current_adj = adj[:]
    current_deg = deg[:]
    current_edges = best_edges
    
    while time.time() - t0 < DEADLINE:
        if T < min_T:
            T = 10.0
            # Random restart
            for i in range(N):
                current_adj[i] = 0
                current_deg[i] = 0
            for u in range(N):
                for v in range(u + 1, N):
                    if rng.random() < 0.7:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_deg[u] += 1
                        current_deg[v] += 1
            current_edges = _count_edges(current_adj)
            if _check_valid(current_adj, current_deg):
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
                    best_deg = current_deg[:]
            continue
            
        # Pick a random edge to flip
        u = rng.randint(0, N - 1)
        v = rng.randint(0, N - 1)
        if u == v:
            continue
            
        # Ensure u < v for consistency? Not strictly necessary for bitmask
        # But let's normalize
        if u > v:
            u, v = v, u
            
        is_present = (current_adj[u] >> v) & 1
        
        # Flip
        current_adj[u] ^= (1 << v)
        current_adj[v] ^= (1 << u)
        
        if is_present:
            current_deg[u] -= 1
            current_deg[v] -= 1
            new_edges = current_edges - 1
        else:
            current_deg[u] += 1
            current_deg[v] += 1
            new_edges = current_edges + 1
            
        # Check validity
        valid = _check_valid(current_adj, current_deg)
        
        if valid:
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = current_adj[:]
                best_deg = current_deg[:]
            
            # Accept move
            delta = new_edges - current_edges
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_edges = new_edges
            else:
                # Reject
                if is_present:
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_deg[u] += 1
                    current_deg[v] += 1
                else:
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                    current_deg[u] -= 1
                    current_deg[v] -= 1
        else:
            # Reject invalid move
            if is_present:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                current_deg[u] += 1
                current_deg[v] += 1
            else:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                current_deg[u] -= 1
                current_deg[v] -= 1
                
        T *= cooling_rate
        
    # Convert best_adj to list of edges
    result = []
    for u in range(N):
        mask = best_adj[u]
        while mask:
            bit = mask & -mask
            v = bit.bit_length() - 1
            if u < v:
                result.append([int(u), int(v)])
            mask ^= bit
            
    return result
