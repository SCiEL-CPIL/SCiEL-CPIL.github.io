import time
import random
import itertools

def construct():
    deadline = time.time() + 140
    
    # Helper to check if a mask induces a regular subgraph
    def is_regular(mask, adj):
        degs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            degs.append((adj[v] & mask).bit_count())
            m ^= lsb
        if not degs:
            return True
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    # Helper to find any violating subset
    def find_violation(adj, n):
        # Check sizes 6 to n
        for k in range(6, n + 1):
            # Use combinations for clarity, might be slower than bit tricks but n is small
            # C(22, 6) is small enough
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular(mask, adj):
                    return mask
        return 0

    for n in range(22, 4, -1):
        # Try to find a valid graph of size n
        # We use random restarts
        while time.time() < deadline:
            # Random initialization
            p = 0.5
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local Search
            # We iterate, finding violations and flipping edges
            # Limit steps to avoid getting stuck in one bad config too long
            max_local_steps = 50
            
            for _ in range(max_local_steps):
                if time.time() > deadline:
                    break
                
                viol_mask = find_violation(adj, n)
                
                if viol_mask == 0:
                    # Valid graph found
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Extract vertices in viol_mask
                vs = []
                m = viol_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    vs.append(v)
                    m ^= lsb
                
                # Flip a random edge within the violating set
                if len(vs) >= 2:
                    u, v = random.sample(vs, 2)
                    # Toggle
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    # Should not happen for k>=6
                    break
            
            # If we exit the local search loop without success, 
            # we restart with a new random graph (handled by outer while)

    # Fallback: return a trivial valid graph
    return {'n': 5, 'edges': []}
