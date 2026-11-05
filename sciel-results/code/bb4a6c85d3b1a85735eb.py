import random
import time
import numpy as np

def _check_girth_circulant(n, voltages, g):
    """
    Check if the circulant graph defined by 'voltages' on Z_n has girth >= g.
    A cycle of length L exists if there are indices i_1...i_L such that 
    sum(v_{i_j}) = 0 mod n.
    We check for cycles of length 3 to g-1.
    """
    # Precompute sums for small lengths
    # For L=3: v_i + v_j + v_k = 0
    # For L=4: v_i + v_j + v_k + v_l = 0
    # etc.
    
    # Optimization: Use a set of reachable sums for each length
    # reachable[d] = set of sums achievable with d steps
    # Start with {0}
    # reachable[d+1] = { (s + v) % n for s in reachable[d] for v in voltages }
    # If 0 is in reachable[d] for d < g, and d >= 3 (since 2-cycles are prevented by simplicity), then girth < g.
    
    # Note: We must ensure we don't backtrack. 
    # In a circulant graph, backtracking is going i -> j -> i. 
    # The voltage sum for a backtrack is v + (-v) = 0. 
    # Since we check for cycles, we should exclude the immediate reverse.
    # However, the set of sums approach counts walks, not necessarily simple cycles.
    # A walk that backtracks has sum 0 at step 2. 
    # We require no cycle of length < g.
    # If a walk of length L < g returns to 0, it contains a cycle of length <= L.
    # So if 0 is in reachable[L] for L < g, we fail.
    
    # We must be careful: The set of sums includes backtracks.
    # Backtrack: v_i + (-v_i) = 0. This is length 2.
    # We know the graph is simple, so no loops (v_i != 0) and no multi-edges (v_i != -v_j).
    # So no cycles of length 1 or 2.
    # We check L from 3 to g-1.
    
    reachable = {0}
    for d in range(1, g):
        new_reachable = set()
        for s in reachable:
            for v in voltages:
                new_reachable.add((s + v) % n)
        # If we return to 0 at step d, and d >= 3, we have a cycle.
        # We must ensure this cycle is not just a backtrack.
        # Since no length 2 cycles exist, any return to 0 at d >= 3 is a valid cycle.
        if 0 in new_reachable and d >= 3:
            return False
        reachable = new_reachable
        
    return True

def construct(k, g):
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    n_start = max(2 * k, moore)
    if n_start % 2 != 0:
        n_start += 1
        
    max_n = 1200 # Upper bound for search
    
    # We try to find a valid circulant graph (or generalized circulant)
    # For k=5, we need 5 voltages.
    # We iterate n from n_start to max_n.
    
    for n in range(n_start, max_n + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find voltages v1..vk in Z_n
        # Constraints:
        # 1. v_i != 0
        # 2. v_i != -v_j for i != j
        # 3. Girth >= g
        
        attempts = 0
        max_attempts = 5000 # Limit attempts per n
        
        while attempts < max_attempts and time.time() < deadline - 2:
            attempts += 1
            # Generate random voltages
            voltages = []
            valid = True
            for _ in range(k):
                v = rng.randint(1, n - 1)
                # Check uniqueness up to sign
                for u in voltages:
                    if v == u or v == n - u:
                        valid = False
                        break
                if not valid:
                    break
                voltages.append(v)
            
            if not valid:
                continue
                
            if _check_girth_circulant(n, voltages, g):
                # Construct the graph
                edges = []
                for i in range(n):
                    for v in voltages:
                        j = (i + v) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback: Random regular graph search (from original code, but faster)
    # If circulant search fails, try random pairing for small n
    # This is a fallback, might not find a solution for large g
    
    return None, []
