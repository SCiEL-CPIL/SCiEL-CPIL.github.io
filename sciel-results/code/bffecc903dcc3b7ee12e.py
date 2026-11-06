"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _count_reg_subgraphs(adj, n, k):
    """
    Count the number of induced subgraphs of size k that are regular.
    adj: list of ints, adj[i] is bitmask of neighbors of i.
    """
    count = 0
    vertices = range(n)
    for subset in itertools.combinations(vertices, k):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Check regularity
        # A graph is regular if all degrees are equal.
        # In an induced subgraph on 'subset', degree of v is popcount(adj[v] & mask).
        # Note: adj[v] & mask includes v itself if v is in adj[v]? No, simple graph, no self loops.
        # But adj[v] & mask counts neighbors of v within the subset.
        
        degrees = []
        for v in subset:
            d = (adj[v] & mask).bit_count()
            degrees.append(d)
        
        if len(set(degrees)) == 1:
            count += 1
            
    return count

def _has_violation_full(adj, n):
    """
    Check if any induced subgraph of size >= 6 is regular.
    Returns True if violation exists.
    """
    for k in range(6, n + 1):
        if _count_reg_subgraphs(adj, n, k) > 0:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Target n=20 first, then 19, 18...
    for n in [20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for this n
        max_restarts = 10 if n == 20 else 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Objective: Minimize count of regular subgraphs of size 6
            # This is the most common violation and cheapest to check
            current_cost = _count_reg_subgraphs(adj, n, 6)
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.1
            cooling = 0.99
            max_iter = 500  # Keep low because cost eval is expensive
            
            it = 0
            while it < max_iter and T > T_min and time.time() < deadline - 2:
                it += 1
                
                # Pick random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = _count_reg_subgraphs(adj, n, 6)
                delta = new_cost - current_cost
                
                # Accept if better or with prob exp(-delta/T)
                if delta < 0:
                    current_cost = new_cost
                else:
                    # If delta is 0, accept to diversify
                    if delta == 0 or random.random() < pow(2.71828, -delta / T):
                        current_cost = new_cost
                    else:
                        # Revert
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                T *= cooling
                
                if current_cost == 0:
                    # Candidate for full check
                    if not _has_violation_full(adj, n):
                        # Valid graph!
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        # Break out of SA loop and restarts? 
                        # If we found a valid graph for this n, we can stop searching this n
                        # and try a larger n? No, we are iterating n downwards.
                        # So if we found valid n=20, we return it.
                        # If we found valid n=19, we might want to try n=20 more?
                        # The loop structure tries n=20 first. If it fails, it goes to 19.
                        # So if we find valid n=19, we should probably stop and return 19,
                        # unless we have time to retry n=20.
                        # Given the difficulty, finding any valid graph is good.
                        # Let's break and return the best found.
                        return {'n': best_n, 'edges': best_edges}
                    else:
                        # Failed full check, continue SA to try to fix larger k
                        # Or break and restart? Restarting might be better.
                        break
            
            # If we didn't return in the while loop, continue to next restart

    # Fallback: return best found or default
    if best_n == 5:
         # Try a quick random search for small n just in case? 
         # Or just return 5.
         # Actually, let's try to find a valid graph for n=10-15 quickly if 20-17 failed.
         for n in [15, 14, 13, 12, 11, 10]:
             if time.time() > deadline - 2:
                 break
             for _ in range(5):
                 adj = [0] * n
                 for i in range(n):
                     for j in range(i + 1, n):
                         if random.random() < 0.5:
                             adj[i] |= (1 << j)
                             adj[j] |= (1 << i)
                 if not _has_violation_full(adj, n):
                     edges = []
                     for i in range(n):
                         for j in range(i + 1, n):
                             if adj[i] & (1 << j):
                                 edges.append([i, j])
                     if n > best_n:
                         best_n = n
                         best_edges = edges
                     break
             if best_n > 5:
                 break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
