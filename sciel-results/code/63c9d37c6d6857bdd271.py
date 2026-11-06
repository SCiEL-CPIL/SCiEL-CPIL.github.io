import time
import random
import math
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Fast check if the graph defined by adjacency list of bitmasks 'adj' on n vertices 
    contains K2,2,2 as a subgraph.
    adj[i] is a bitmask of neighbors of i.
    """
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            # Need at least 2 more common neighbors to form K2,2,2
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        return True
    return False

def _find_violation_edges(adj, n, max_edges=10):
    """
    Find up to max_edges edges that are part of K2,2,2 subgraphs.
    Returns list of (u, v) tuples where edge (u,v) is in color 0 (adj[i]&adj[j]).
    If no violations in color 0, returns edges from color 1 (complement).
    """
    violations = []
    
    # Check color 0
    for u in range(n):
        if len(violations) >= max_edges:
            break
        for v in range(u + 1, n):
            if len(violations) >= max_edges:
                break
            if not (adj[u] >> v) & 1:
                continue
            common_mask = adj[u] & adj[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            found = False
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = adj[b1] & adj[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        violations.append((u, v))
                        found = True
                        break
                if found:
                    break
            if found:
                continue
                
    if violations:
        return violations
    
    # Check color 1 (complement)
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
        
    for u in range(n):
        if len(violations) >= max_edges:
            break
        for v in range(u + 1, n):
            if len(violations) >= max_edges:
                break
            if not (comp[u] >> v) & 1:
                continue
            common_mask = comp[u] & comp[v]
            common_mask &= ~(1 << u)
            common_mask &= ~(1 << v)
            
            if common_mask.bit_count() < 2:
                continue
                
            cn_list = []
            m = common_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            
            L = len(cn_list)
            found = False
            for i in range(L):
                b1 = cn_list[i]
                for j in range(i + 1, L):
                    b2 = cn_list[j]
                    common_b1_b2 = comp[b1] & comp[b2] & common_mask
                    if common_b1_b2.bit_count() >= 2:
                        # Edge (u,v) is 1 in comp, so 0 in adj
                        violations.append((u, v))
                        found = True
                        break
                if found:
                    break
            if found:
                continue
                
    return violations

def _is_valid_fast(adj, n):
    """Check both color classes are K2,2,2-free using fast check."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _paley_init(n):
    """Initialize a Paley graph on n vertices (if n is a prime congruent to 1 mod 4)."""
    if n < 2:
        return [0] * n
    
    # Find if n is a prime congruent to 1 mod 4
    if n % 4 != 1:
        # For non-Paley-friendly n, use random initialization
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    squares = set()
    for x in range(1, n):
        squares.add(x * x % n)
    
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n_target = 30
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    best_adj = None
    best_n = 29  # Fallback
    
    # Multiple restarts with simulated annealing
    num_restarts = 8
    iterations_per_restart = 8000
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        n = n_target
        adj = [0] * n
        
        # Initialize first 29 vertices with Paley graph
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                
        # Initialize vertex 29 (index 29) with different strategies
        if restart == 0:
            # Random
            for i in range(q):
                if random.random() < 0.5:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        elif restart == 1:
            # All 0
            pass
        elif restart == 2:
            # All 1
            for i in range(q):
                adj[i] |= (1 << 29)
                adj[29] |= (1 << i)
        elif restart == 3:
            # Based on Paley pattern extended
            for i in range(q):
                diff = (29 - i) % q
                if diff in squares:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        else:
            # Random with bias
            bias = 0.3 + 0.4 * (restart - 4) / max(1, num_restarts - 4)
            for i in range(q):
                if random.random() < bias:
                    adj[i] |= (1 << 29)
                    adj[29] |= (1 << i)
        
        # Simulated annealing
        temperature = 1.0
        cooling_rate = 0.9995
        best_local_adj = adj[:]
        best_local_violations = None
        
        # First, try to find any valid solution quickly
        for iteration in range(iterations_per_restart):
            if time.time() > deadline - 5:
                break
            
            violations = _find_violation_edges(adj, n, max_edges=5)
            if not violations:
                # Valid coloring found!
                best_adj = adj[:]
                best_n = n
                break
            
            # Choose a random violating edge
            u, v = random.choice(violations)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Cool down
            temperature *= cooling_rate
            
            # Occasionally restart with a small perturbation if stuck
            if iteration % 1000 == 999 and temperature < 0.1:
                # Add some randomness
                for _ in range(3):
                    ru = random.randint(0, n - 1)
                    rv = random.randint(0, n - 1)
                    if ru != rv:
                        if (adj[ru] >> rv) & 1:
                            adj[ru] &= ~(1 << rv)
                            adj[rv] &= ~(1 << ru)
                        else:
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                temperature = 0.5  # Reheat slightly
        
        if best_n == 30:
            break
    
    if best_n == 30:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
        
    # Fallback to n=29
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': n, 'colors': colors}
