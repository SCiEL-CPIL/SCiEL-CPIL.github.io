import random
import time

def construct():
    """
    Constructs a C6-free graph on 40 vertices with many edges.
    Returns a list of edges [u, v].
    """
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    random.seed(42) # For reproducibility of the search process if needed, but usually we want diversity
    
    def is_c6_free(adj_bits, u, v):
        """
        Checks if adding edge (u, v) creates a C6.
        adj_bits: list of ints representing adjacency bitsets.
        A C6 is created if there is a path of length 5 between u and v.
        Path length 5 from u to v means:
        u -> x1 -> x2 -> x3 -> x4 -> v
        This is equivalent to:
        (N(N(N(u))) & N(N(N(v)))) != 0  (path u->x1->x2->x3 and v->x4->x3->x2... wait)
        
        Let's trace carefully.
        Path: u - a - b - c - d - v
        a is in N(u)
        b is in N(a)
        c is in N(b)
        d is in N(c)
        v is in N(d)
        
        So d is in N(v).
        c is in N(d) => c is in N(N(v)).
        b is in N(c) => b is in N(N(N(v))).
        a is in N(b) => a is in N(N(N(N(v)))).
        u is in N(a) => u is in N(N(N(N(N(v))))).
        
        Wait, checking if u is reachable from v in 5 steps is one way.
        Alternatively, check if there is a common vertex in the 3rd neighborhood of u and 2nd neighborhood of v?
        Path: u - a - b - c - d - v
        c is in N2(u) (u-a-b-c is 3 steps, so c is in N3(u)? No.
        u(0) - a(1) - b(2) - c(3) - d(4) - v(5)
        c is at distance 3 from u.
        c is at distance 2 from v (v-d-c).
        So if N3(u) intersects N2(v), we have a path of length 5.
        Also, the path could be traversed in reverse, so N2(u) intersects N3(v) also creates a C6.
        
        So check: (N3(u) & N2(v)) != 0 or (N2(u) & N3(v)) != 0.
        """
        # Calculate N2 and N3 for u and v on the fly? 
        # It's better to precompute N2 and N3 for all vertices, but they change as we add edges.
        # However, we can compute them quickly for a specific u, v.
        
        # N(u)
        nu = adj_bits[u]
        nv = adj_bits[v]
        
        # N2(u) = union of N(x) for x in N(u)
        # Since n=40, we can just iterate.
        # But bitset operations are fast.
        
        # Helper to get Nk
        def n_k(start, k, adj):
            # Returns bitset of vertices at distance exactly k? Or <= k?
            # "Exactly k" is hard with bitsets without subtracting previous.
            # "At most k" is easier.
            # If we use "at most", we might get false positives (shorter paths).
            # A path of length 3 between u and v doesn't create a C6.
            # We need a path of length EXACTLY 5.
            # If there is a path of length 3, and we add u-v, we get a C4.
            # The graph must be C6-free. It doesn't say C4-free.
            # So adding an edge that creates a C4 is fine, as long as it doesn't create a C6.
            # However, if we have a path of length 3, adding u-v creates a C4.
            # Does it create a C6? No.
            # So we only care about paths of length 5.
            
            # To check for path of length exactly 5:
            # We can compute Reachable_1, Reachable_2, etc.
            # R_1(u) = N(u)
            # R_2(u) = N(R_1(u)) \ {u} ? No, N(R_1(u)) includes u.
            # Actually, for C6 detection, we just need to know if there is ANY walk of length 5.
            # If there is a walk of length 5, it might contain backtracking.
            # A simple path of length 5 is required for a simple cycle C6.
            # If there is a walk of length 5, does it imply a simple path of length 5?
            # Not necessarily. But in a simple graph, if dist(u,v) <= 5 and dist(u,v) is odd?
            # Actually, if there is a path of length 3, and a path of length 2, etc.
            
            # Let's use the intersection method which is standard for this.
            # Check if N3(u) intersects N2(v).
            # N3(u) usually means vertices at distance exactly 3.
            # If we use "distance <= 3", we might intersect with N2(v) due to shorter paths.
            # Example: u-x-y-z-v (len 4). u is dist 3 from z? No.
            
            # Let's compute exact distance sets.
            # S1_u = N(u)
            # S2_u = N(S1_u) \ S1_u \ {u}  (Vertices at dist exactly 2)
            # S3_u = N(S2_u) \ S2_u \ S1_u \ {u}
            
            # Similarly for v.
            
            # This is a bit expensive to do from scratch for every edge check.
            # But n=40 is small.
            
            # Optimization: Precompute N2 and N3 for all nodes?
            # They change.
            
            # Let's implement a fast check.
            
            # Get N(u)
            mask_u = adj[start]
            
            # Get N2(start)
            # Iterate bits in mask_u
            # This is slow in Python loops.
            
            # Alternative:
            # Since n=40, we can represent adj as a list of 40-bit integers.
            # We can compute N2(u) by ORing adj[x] for all x in N(u).
            
            # Let's try a different approach:
            # Keep N2 and N3 matrices updated?
            # Or just compute them on the fly efficiently.
            
            # For n=40, a Python loop over 40 items is ~1us.
            # 10000 checks * 1us = 10ms. Very fast.
            
            # Let's write a helper to get N_k_exact(u, k)
            
            prev = 1 << start
            curr = adj[start]
            for _ in range(k):
                # Next layer
                next_layer = 0
                # Iterate through bits in curr
                m = curr
                while m:
                    lsb = m & -m
                    idx = (lsb.bit_length() - 1)
                    next_layer |= adj[idx]
                    m ^= lsb
                # Remove vertices already seen (to get exact distance)
                next_layer &= ~prev
                prev |= next_layer
                curr = next_layer
            return curr

        # Check intersection
        # N3(u) & N2(v)
        n3_u = n_k(u, 3, adj_bits)
        n2_v = n_k(v, 2, adj_bits)
        
        if n3_u & n2_v:
            return False
            
        n2_u = n_k(u, 2, adj_bits)
        n3_v = n_k(v, 3, adj_bits)
        
        if n2_u & n3_v:
            return False
            
        return True

    def run_greedy(restarts=50):
        nonlocal best_edges, best_count
        start_time_inner = time.time()
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Initialize empty graph
            adj = [0] * n
            edges = []
            
            # Generate random edge order
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            # Optional: Start with a random subgraph to break symmetry?
            # Or just start empty.
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                
                # Check if u and v are already connected
                if (adj[u] >> v) & 1:
                    continue
                
                # Check C6
                if is_c6_free(adj, u, v):
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = [list(e) for e in edges]
                # Update deadline slightly if we found a big improvement? No, keep strict.
            
            # If we found a very good graph, we can stop early?
            # Theoretical max is unknown, but 105 is good.
            if best_count >= 105:
                # Try to push a bit more
                pass 

    run_greedy(restarts=200)
    
    # Final check: ensure best_edges is valid
    # (The greedy algorithm ensures validity)
    
    return best_edges

if __name__ == "__main__":
    # Simple test
    edges = construct()
    print(f"Found {len(edges)} edges")
