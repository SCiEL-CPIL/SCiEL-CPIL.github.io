import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave margin

    def find_bad_subset(n, adj_bits):
        # adj_bits: list of ints, bit j set if edge (i, j) exists
        for k in range(6, n + 1):
            # Optimization: if k is large, there are fewer subsets, 
            # but the check is more expensive? No, bit_count is O(1) effectively.
            # But generating combinations is the cost.
            for sub in itertools.combinations(range(n), k):
                mask = 0
                for v in sub:
                    mask |= (1 << v)
                
                degs = []
                for v in sub:
                    # Count neighbors of v within the subset
                    degs.append((adj_bits[v] & mask).bit_count())
                
                if len(set(degs)) == 1:
                    return sub
        return None

    def try_search(n, max_restarts=100):
        # Use local search with random restarts
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Init random graph
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Local search
            stuck_counter = 0
            while time.time() < deadline:
                bad_sub = find_bad_subset(n, adj_bits)
                if bad_sub is None:
                    return adj_bits
                
                stuck_counter += 1
                if stuck_counter > 200:
                    # If stuck, break and restart
                    break
                
                # Pick random edge inside bad_sub to flip
                # This changes the induced subgraph
                u_idx, v_idx = random.sample(range(len(bad_sub)), 2)
                u = bad_sub[u_idx]
                v = bad_sub[v_idx]
                
                # Flip edge (u, v)
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
        
        return None

    # Try to find a graph with n=21
    adj = try_search(21)
    if adj is not None:
        edges = []
        n = 21
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Try to find a graph with n=20
    adj = try_search(20)
    if adj is not None:
        edges = []
        n = 20
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Fallback: Return a small valid graph
    # Empty graph on 5 vertices is valid
    return {'n': 5, 'edges': []}
