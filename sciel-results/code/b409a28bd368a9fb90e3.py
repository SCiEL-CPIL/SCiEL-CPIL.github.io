import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    n = 40
    
    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        A new C6 must use the new edge (u, v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a, b, c, d distinct and not u, v.
        
        Using bitmasks for adjacency:
        adj[i] is an integer where bit j is 1 if (i, j) is an edge.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # a in N(u) \ {v}
        # d in N(v) \ {u}
        # a != d
        
        # Iterate over a in N(u)
        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # Iterate over d in N(v) \ {u, a}
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                # We need b in N(a) \ {u, v, d}
                # and c in N(d) \ {u, v, a}
                # such that b != c and (b, c) is an edge.
                
                Na = adj[a]
                Nd = adj[d]
                
                # Mask for b: neighbors of a, excluding u, v, d
                # Note: b != a is guaranteed as no self-loops
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                
                # Mask for c: neighbors of d, excluding u, v, a
                # Note: c != d is guaranteed
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # Check if there exists b in b_mask such that adj[b] intersects c_mask
                # Also need b != c. Since adj[b] doesn't have bit b, if c=b, adj[b]&c_mask won't have bit b set.
                # So we just check if for any b in b_mask, adj[b] & c_mask != 0.
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def run_greedy(seed_val):
        random.seed(seed_val)
        
        # Initialize adjacency bitmasks
        adj = [0] * n
        edges = []
        
        # Start with a sparse seed to avoid trivial C6s?
        # Or start empty? Starting empty allows more edges potentially if order is good.
        # But random order from empty might get stuck.
        # Let's try starting with 8 K5s as in the current program, or just empty.
        # The inspiration 1 started with 8 K5s.
        # Let's try both strategies in different restarts?
        # Or just stick to the one that worked (8 K5s).
        
        # Strategy: Start with 8 K5s
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Generate random order of remaining edges
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        return edges

    best_edges = []
    best_count = 0
    
    # Multiple restarts
    # Estimate time per restart.
    # n=40, edges ~ 100-150.
    # would_create_c6 is fast with bitmasks.
    # Let's run as many as possible.
    
    seed_counter = 42
    while time.time() < deadline:
        if time.time() > deadline - 2:
            break
        edges = run_greedy(seed_counter)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed_counter += 1
        
    return best_edges
