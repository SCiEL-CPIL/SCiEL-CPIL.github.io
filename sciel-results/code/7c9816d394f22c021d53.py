import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    
    # Helper to check if adding vertex v with mask M to graph G (size k) creates a regular subgraph of size >= 6
    # G is a list of bitmasks for vertices 0..k-1
    # M is the bitmask of neighbors of v in 0..k-1
    # Returns True if NO new regular subgraph of size >= 6 is created.
    def is_valid_addition(G, k, M):
        # We need to check all subsets S' of {0..k-1} with size j >= 5
        # such that S' U {v} is regular.
        # Condition derived:
        # Let S' be a subset of {0..k-1} of size j.
        # Let N_v = S' & M. d = popcount(N_v).
        # For all u in S', deg_{G[S']}(u) + (1 if u in N_v else 0) must equal d.
        
        # Iterate over all masks S' of size >= 5
        # Optimization: We can iterate over all masks and check popcount.
        # But 2^20 is 1M. 1M * 20 ops = 20M per candidate.
        # If we try 10 candidates, 200M ops. Too slow.
        
        # Better: Iterate over combinations of size 5, 6, ...
        # But generating combinations in Python is slow.
        
        # Alternative: 
        # Precompute degrees for all subsets? No.
        
        # Let's try to limit the check to subsets of size 5 and 6 first?
        # If a regular subgraph of size k>6 exists, it might not contain a regular subgraph of size 6.
        # However, regular subgraphs of size >6 are very rare.
        # The task says "No set of 6 or more". So we MUST check all.
        
        # Is there a faster way?
        # Maybe use NumPy?
        # G is a k x k matrix.
        # For a given S' (mask), we can extract the subgraph.
        
        # Let's stick to bitmasks but optimize the loop.
        # We only check if M is "compatible" with S'.
        
        # Iterate over all submasks of full_mask = (1<<k)-1 with popcount >= 5
        full_mask = (1 << k) - 1
        # This loop is the bottleneck.
        # We can skip if k is small.
        
        # For each S', compute degrees in G[S'].
        # deg[u] = (G[u] & S').bit_count()
        
        # To speed up, we can precompute popcount for all masks up to 2^k?
        # 2^20 = 1M integers. 1M * 8 bytes = 8MB. Acceptable.
        # But we need degrees for each vertex in the subset.
        
        # Let's try a different strategy:
        # Randomly sample subsets? No, must be exhaustive to be valid.
        
        # What if we check only subsets of size 5?
        # If we miss a size 6 regular subgraph, the final graph might be invalid.
        # But the evaluator checks ALL subsets.
        # So we must ensure validity.
        
        # Given the time limit, maybe n=20 is hard with this naive check.
        # However, the seed program scored 1.0 (n=5 or similar?).
        # The prompt says "Graphs with 20 vertices are known".
        # So a valid 20-vertex graph exists.
        
        # Let's try to make the check as fast as possible.
        
        # Precompute popcount for all masks up to 2^k?
        # We can do this once per k.
        popcount_cache = [0] * (1 << k)
        for i in range(1, 1 << k):
            popcount_cache[i] = popcount_cache[i >> 1] + (i & 1)
        
        # Now iterate S'
        # To avoid checking all 2^k masks, we can generate combinations?
        # No, iterating 1..2^k is faster in C-level loops if we use bit tricks, but Python loop is slow.
        
        # Let's try to iterate only over masks with popcount >= 5.
        # We can generate them using Gosper's hack for each size j.
        
        for j in range(5, k + 1):
            # Generate all masks of size j
            # Use Gosper's hack
            mask = (1 << j) - 1
            limit = 1 << k
            while mask < limit:
                # Check regularity for S' = mask
                # N_v = mask & M
                N_v = mask & M
                d = popcount_cache[N_v]
                
                # Check if for all u in mask, deg(u) + (u in N_v) == d
                # deg(u) = popcount(G[u] & mask)
                
                # We can check this by iterating bits in mask
                m = mask
                valid = True
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    deg = popcount_cache[G[u] & mask]
                    # u is in N_v if (N_v & lsb) != 0
                    add = 1 if (N_v & lsb) else 0
                    if deg + add != d:
                        valid = False
                        break
                    m -= lsb
                
                if valid:
                    return False # Found a regular subgraph, invalid addition
                
                # Next mask of same popcount
                c = mask & -mask
                r = mask + c
                mask = (((r ^ mask) >> 2) // c) | r
                
        return True

    def verify_graph(G, n):
        # Full verification for the final graph
        # Same logic, but for all subsets of size >= 6
        # We can reuse the logic by checking if adding a dummy vertex with no edges? No.
        # Just check all subsets of size >= 6 directly.
        popcount_cache = [0] * (1 << n)
        for i in range(1, 1 << n):
            popcount_cache[i] = popcount_cache[i >> 1] + (i & 1)
        
        for mask in range(1, 1 << n):
            if popcount_cache[mask] < 6:
                continue
            # Check if G[mask] is regular
            m = mask
            first_deg = -1
            regular = True
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                deg = popcount_cache[G[u] & mask]
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    regular = False
                    break
                m -= lsb
            if regular:
                return False
        return True

    best_n = 5
    best_graph = {'n': 5, 'edges': []}
    
    # Start with a random 5-vertex graph that is valid (no regular subgraph of size 5? 
    # Actually, we just need no regular subgraph of size >= 6. 
    # A 5-vertex graph trivially has no regular subgraph of size >= 6.
    # So any 5-vertex graph is a valid start.
    # But to maximize growth, maybe start with a specific structure?
    # Let's just start with an empty 5-vertex graph or random.
    
    for attempt in range(20):
        random.seed(attempt)
        G = [0] * 5
        edges = []
        # Initialize 5 vertices randomly? Or empty?
        # Empty is safer? 
        # Let's try random edges for the first 5.
        for i in range(5):
            for j in range(i+1, 5):
                if random.random() < 0.5:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
                    edges.append([i, j])
        
        current_n = 5
        
        # Try to add vertices up to 22
        for target_n in range(6, 23):
            k = current_n
            found_mask = None
            # Try 50 random masks
            for _ in range(50):
                cand_mask = random.getrandbits(k)
                if is_valid_addition(G, k, cand_mask):
                    found_mask = cand_mask
                    break
            
            if found_mask is None:
                break # Failed to add vertex
            
            # Add the vertex
            G.append(found_mask)
            for i in range(k):
                if (found_mask >> i) & 1:
                    G[i] |= (1 << k)
                    edges.append([i, k])
            
            current_n += 1
            
            if current_n > best_n:
                # Verify the whole graph to be sure
                if verify_graph(G, current_n):
                    best_n = current_n
                    best_graph = {'n': current_n, 'edges': edges.copy()}
                    if current_n >= 20:
                        return best_graph
            
            if time.time() - start_time > 110:
                return best_graph
        
    return best_graph
