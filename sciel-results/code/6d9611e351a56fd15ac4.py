"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_n = 128
    nums = list(range(1, max_n + 1))
    
    # Precompute lcm table
    lcm_table = [[0] * (max_n + 1) for _ in range(max_n + 1)]
    for i in range(1, max_n + 1):
        for j in range(i, max_n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict pairs
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a bad triple
    conflict_pairs = [[] for _ in range(max_n + 1)]
    
    for a in range(1, max_n + 1):
        for b in range(a + 1, max_n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, max_n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bitmasks for conflict pairs for faster checking
    conflict_pairs_bits = [[] for _ in range(max_n + 1)]
    for x in range(1, max_n + 1):
        for a, b in conflict_pairs[x]:
            conflict_pairs_bits[x].append((1 << (a-1), 1 << (b-1)))
    
    def greedy(order, mask=0):
        for x in order:
            ok = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (mask & bit_a) and (mask & bit_b):
                    ok = False
                    break
            if ok:
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(max_n) if (mask >> i) & 1]
    
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # 1. Random Restarts with Greedy
    orders = [
        list(range(1, max_n + 1)),
        list(range(max_n, 0, -1)),
    ]
    
    for order in orders:
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # Random shuffles
    for trial in range(20000):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        m = greedy(order)
        size = bin(m).count("1")
        if size > best_size:
            best_size = size
            best_mask = m
    
    # 2. Local Search (Iterated Local Search)
    improved = True
    current_mask = best_mask
    current_size = best_size
    
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try different perturbation sizes
        for k in [1, 2, 3, 5, 10]:
            if time.time() > deadline - 2.0:
                break
            
            # Pick k random elements to remove
            set_bits = [i+1 for i in range(max_n) if (current_mask >> i) & 1]
            if len(set_bits) < k:
                continue
                
            removed = rng.sample(set_bits, k)
            new_mask = current_mask
            for x in removed:
                new_mask &= ~(1 << (x-1))
            
            # Greedily add back
            candidates = [x for x in range(1, max_n + 1) if not (new_mask & (1 << (x-1)))]
            rng.shuffle(candidates)
            
            m = greedy(candidates, new_mask)
            size = bin(m).count("1")
            
            if size > current_size:
                current_size = size
                current_mask = m
                if size > best_size:
                    best_size = size
                    best_mask = m
                improved = True
                break # Break k loop, restart while loop with new current
    
    # 3. Branch and Bound for exact solution
    # Build adjacency structure for B&B
    # For each vertex, store the set of vertices it conflicts with (in the 3-uniform hypergraph sense)
    # A vertex x conflicts with a pair (a,b) if {x,a,b} is a bad triple
    
    # For B&B, we need to track which pairs are "active" (both endpoints in the current set)
    # When we add vertex x, we need to check if any conflict pair (a,b) for x has both a and b already selected
    
    # Upper bound: use the fact that if we have k remaining vertices, the max we can add is k
    # But we can be smarter: if vertex x is in the remaining set, it can only be added if no 
    # conflict pair (a,b) for x has both a,b in the current set
    
    # Simpler upper bound: count remaining vertices that are "safe" (can potentially be added)
    # A vertex x is safe if there's no pair (a,b) in conflict_pairs[x] where both a and b are in current set
    
    def count_safe_vertices(remaining_mask, current_mask):
        count = 0
        for x in range(1, max_n + 1):
            if not (remaining_mask & (1 << (x-1))):
                continue
            # Check if x can be added
            safe = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (current_mask & bit_a) and (current_mask & bit_b):
                    safe = False
                    break
            if safe:
                count += 1
        return count
    
    # B&B with memoization of visited states (limited)
    # State: (remaining_mask, current_mask) -> best additional size found
    # But this is too large, so we'll use a simple B&B with pruning
    
    def branch_and_bound(remaining_mask, current_mask, current_size, depth_limit=20):
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 1.0:
            return
        
        # Upper bound: current_size + number of safe vertices
        safe_count = count_safe_vertices(remaining_mask, current_mask)
        if current_size + safe_count <= best_size:
            return
        
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Find a vertex to branch on
        # Choose vertex with highest degree in remaining subgraph for better pruning
        best_vertex = -1
        best_degree = -1
        
        for x in range(1, max_n + 1):
            if not (remaining_mask & (1 << (x-1))):
                continue
            # Count how many other remaining vertices conflict with x
            degree = 0
            for bit_a, bit_b in conflict_pairs_bits[x]:
                # If a is in remaining and b is in current, or vice versa
                # Actually, for degree we want to count potential conflicts
                # A vertex y in remaining "conflicts" with x if there exists a pair (a,b) in conflict_pairs[x]
                # where one of a,b is y and the other is in current or remaining
                pass
            
            # Simpler: just pick the first available vertex
            if best_vertex == -1:
                best_vertex = x
                break
        
        if best_vertex == -1:
            return
        
        x = best_vertex
        bit_x = 1 << (x-1)
        
        # Option 1: Include x
        # Check if x can be included
        can_include = True
        for bit_a, bit_b in conflict_pairs_bits[x]:
            if (current_mask & bit_a) and (current_mask & bit_b):
                can_include = False
                break
        
        if can_include:
            new_current = current_mask | bit_x
            new_remaining = remaining_mask & ~bit_x
            branch_and_bound(new_remaining, new_current, current_size + 1, depth_limit - 1)
        
        # Option 2: Exclude x
        new_remaining = remaining_mask & ~bit_x
        branch_and_bound(new_remaining, current_mask, current_size, depth_limit - 1)
    
    # Run B&B from the current best solution's complement
    # Start with remaining = all vertices not in best_mask
    remaining_mask = 0
    for x in range(1, max_n + 1):
        if not (best_mask & (1 << (x-1))):
            remaining_mask |= (1 << (x-1))
    
    # Try to improve using B&B
    branch_and_bound(remaining_mask, best_mask, best_size, depth_limit=30)
    
    return mask_to_set(best_mask)
# EVOLVE-BLOCK-END
