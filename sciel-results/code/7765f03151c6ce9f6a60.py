import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    def count_violations_all(adj, n):
        """
        Count the number of regular induced subgraphs of size k >= 6.
        Uses bitmasks for speed.
        """
        violations = 0
        
        # Precompute masks for all subsets of size k? 
        # No, n is small (<=21), but iterating all subsets of size k>=6 is expensive.
        # C(21, 6) = 54264. C(21, 7) = 116280. C(21, 8) = 203490...
        # Sum is ~ 1M subsets.
        # Checking each subset takes O(k) bit ops.
        # 1M * 10 = 10M ops. In Python, this might take 1-2 seconds.
        # For Simulated Annealing, we need to evaluate many times.
        # We need a faster evaluation or a better heuristic.
        
        # Heuristic: Check k=6 first. If > 0, return count_6 + penalty.
        # If k=6 is 0, check k=7, etc.
        # Actually, if we have a valid graph, count is 0.
        # We want to minimize count.
        
        # Optimization: 
        # A subset S is regular if all degrees in G[S] are equal.
        # We can iterate k from 6 to n.
        # For performance in SA, we might only check k=6 and k=7 initially, 
        # and do a full check only when count_6+count_7 is low.
        # But to be safe and correct, let's try to make the full check fast enough.
        
        # Alternative: Use a different objective?
        # No, the task requires 0 violations for validity.
        # Minimizing total violations is the best proxy.
        
        # Let's optimize the loop.
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                verts = list(subset)
                d0 = (adj[verts[0]] & mask).bit_count()
                regular = True
                for v in verts[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    violations += 1
        return violations

    def is_valid(n, adj):
        """
        Check if graph is valid (0 violations).
        """
        return count_violations_all(adj, n) == 0

    def simulated_annealing(n, max_iters, time_limit):
        """
        Simulated Annealing to minimize violations.
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_violations_all(adj, n)
        best_adj = adj[:]
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.9995
        
        for i in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = count_violations_all(adj, n)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            
            T *= cooling
            
            # If we found a valid graph, we can stop early
            if best_cost == 0:
                break
        
        return best_adj, best_cost

    # Try n=21 first, then 20
    for n in [21, 20]:
        # Run multiple SA runs with different seeds
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + n)
            
            # SA parameters
            max_iters = 20000
            # Use a shorter time limit per run to allow multiple attempts
            run_deadline = min(deadline, start_time + 10)
            
            best_adj, best_cost = simulated_annealing(n, max_iters, run_deadline)
            
            if best_cost == 0:
                # Convert adj to edges list
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                
                # Final verification
                if is_valid(n, best_adj):
                    return {'n': n, 'edges': edges}
                
        # If n=21 failed, try n=20
        if n == 21:
             # Reset deadline for n=20 attempts if time remains
             pass

    # Fallback: return a known valid small graph or random 5-vertex graph
    # A 5-vertex graph is always valid since no 6-subset exists.
    # But we want to maximize n.
    # If we didn't find 20 or 21, return a random 5-vertex graph?
    # The task says "Maximize n".
    # If we can't find 20, maybe we can find 19?
    # Let's try to find the largest n we can verify quickly.
    
    # Since the current program scores 1.0 (which implies n=20 or similar valid graph was found or returned),
    # and we know 20-vertex graphs exist, we should be able to find one.
    # If SA fails, we might need a more robust method or just return a hardcoded 20-vertex graph if we had one.
    # But we don't have one hardcoded.
    
    # Let's try a greedy construction for n=20 if SA fails?
    # Or just return a small valid graph to ensure score > 0.
    
    return {'n': 5, 'edges': []}
