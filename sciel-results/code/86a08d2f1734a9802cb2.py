import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s limit, 5s margin

    # Try n=21 first, then n=20
    for n in [21, 20]:
        if time.time() > deadline:
            break
        
        # Try multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Initialize random graph using bitmasks
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        
            # Local search
            for step in range(3000):
                if time.time() > deadline:
                    break
                    
                # Find violation
                violation = find_violation_bitmask(adj_mask, n)
                if violation is None:
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if (adj_mask[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Flip an edge in the violation set
                S = violation
                u, v = random.sample(S, 2)
                if (adj_mask[u] >> v) & 1:
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                else:
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                    
    # Fallback
    return {'n': 5, 'edges': []}

def find_violation_bitmask(adj_mask, n):
    # Check subsets of size >= 6
    # Optimize: Check size 6 first.
    for k in range(6, n+1):
        for S in itertools.combinations(range(n), k):
            if is_regular_bitmask(S, adj_mask):
                return list(S)
    return None

def is_regular_bitmask(S, adj_mask):
    k = len(S)
    if k < 6:
        return False
    
    # Calculate degrees within S
    degs = []
    # Precompute mask for S
    mask_S = 0
    for v in S:
        mask_S |= (1 << v)
        
    for v in S:
        # Count neighbors of v in S
        # adj_mask[v] has bits for all neighbors
        # We want intersection with mask_S, excluding v itself
        # Since v is in mask_S, we can just count bits in (adj_mask[v] & mask_S)
        # Note: adj_mask[v] does not have bit v set (simple graph)
        deg = bin(adj_mask[v] & mask_S).count('1')
        degs.append(deg)
        
    return len(set(degs)) == 1
