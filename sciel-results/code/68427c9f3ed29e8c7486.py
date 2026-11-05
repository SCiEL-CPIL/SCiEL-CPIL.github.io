import time
import random
import itertools
import numpy as np

def _paley_17():
    """Return the Paley graph on 17 vertices as a bitmask adjacency list."""
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_k4(adj):
    n = len(adj)
    count = 0
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            if not (adj[i] >> j) & 1: continue
            for k in range(j + 1, n - 1):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1): continue
                # i,j,k form a triangle
                # Check l > k
                # Candidates for l are neighbors of i, j, k
                # mask = adj[i] & adj[j] & adj[k] & ~(1<<i) & ~(1<<j) & ~(1<<k)
                # We only care about l > k to avoid double counting?
                # No, we are counting specific sets {i,j,k,l}.
                # To count each K4 once, we can enforce i<j<k<l.
                # The loop above enforces i<j<k.
                # So we need l > k.
                mask = adj[i] & adj[j] & adj[k]
                # Remove i, j, k
                mask &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                # Only consider bits > k
                # mask >>= (k+1) ? No, we need to count bits.
                # We can just iterate l from k+1 to n-1
                for l in range(k + 1, n):
                    if (mask >> l) & 1:
                        count += 1
    return count

def _count_ind6(adj):
    n = len(adj)
    count = 0
    # Iterate all 6-combinations
    for combo in itertools.combinations(range(n), 6):
        is_ind = True
        # Build mask of the combo
        mask = 0
        for v in combo:
            mask |= (1 << v)
        for v in combo:
            if adj[v] & mask:
                is_ind = False
                break
        if is_ind:
            count += 1
    return count

def _is_valid(adj):
    return _count_k4(adj) == 0 and _count_ind6(adj) == 0

def _sa_find(n, time_budget=5.0):
    """Run Simulated Annealing to find a valid graph on n vertices."""
    start_time = time.time()
    best_cost = float('inf')
    best_adj = None
    
    # Initial random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Calculate initial cost
    c_k4 = _count_k4(adj)
    c_ind = _count_ind6(adj)
    cost = c_k4 + c_ind
    if cost == 0:
        return adj
    
    T = 1.0
    cooling = 0.995
    current_adj = adj[:]
    current_cost = cost
    best_adj = current_adj[:]
    best_cost = cost
    
    while time.time() - start_time < time_budget:
        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(i + 1, n - 1)
        
        # Flip edge
        if (current_adj[i] >> j) & 1:
            # Remove edge
            current_adj[i] &= ~(1 << j)
            current_adj[j] &= ~(1 << i)
        else:
            # Add edge
            current_adj[i] |= (1 << j)
            current_adj[j] |= (1 << i)
        
        # Calculate new cost
        # Optimization: Only recalculate if necessary? 
        # For n<=25, full recalc is fast enough (~0.1ms?)
        # _count_k4 is ~1000 ops. _count_ind6 is ~20000 ops.
        # Total ~20k ops. In Python, maybe 1-2ms.
        # 500 steps/sec. 5s -> 2500 steps.
        
        new_c_k4 = _count_k4(current_adj)
        new_c_ind = _count_ind6(current_adj)
        new_cost = new_c_k4 + new_c_ind
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = current_adj[:]
                if best_cost == 0:
                    return best_adj
        else:
            # Revert
            if (current_adj[i] >> j) & 1:
                # It was removed, so add back
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
            else:
                # It was added, so remove
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_adj if best_cost == 0 else None

def construct():
    deadline = time.time() + 110
    best_n = 17
    best_adj = _paley_17()
    
    # Try to improve n
    for n in range(18, 31):
        if time.time() > deadline - 15:
            break
        
        # Try a few restarts for this n
        found = None
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            adj = _sa_find(n, time_budget=3.0)
            if adj:
                found = adj
                break
        
        if found:
            best_n = n
            best_adj = found
            # If we found a high n, we might want to try higher ones
            # But if we are close to deadline, stop
            if n >= 25:
                # We are doing well, maybe try one more step?
                # But keep safety margin
                pass
        else:
            # If we fail at n, it might be harder for n+1
            # But we can still try, just with less time?
            # Or break if n is small and we fail?
            # Let's continue trying until deadline.
            pass

    # Convert bitmask to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
