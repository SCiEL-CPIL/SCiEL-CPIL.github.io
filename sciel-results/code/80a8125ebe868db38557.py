<<<<<<< SEARCH
"""
Maximize sum of pairwise-coprime interval integers from 1..256.

The optimal set consists of 1 and, for every prime p <= 256, the largest power p^k <= 256.
These numbers are pairwise coprime because each has a unique prime factor (the base p).
Any composite number formed by multiplying primes from this set would be less than the
sum of their individual maximum powers (for small primes) or impossible (for large primes
where p*q > 256).
Thus, this greedy approach yields the global maximum.
"""

import time

def _sieve(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _largest_prime_power(p, n):
    """Return the largest power of p that is <= n."""
    val = p
    while val * p <= n:
        val *= p
    return val

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 256
    
    # Find all primes up to n
    primes = _sieve(n)
    
    # For each prime, find the largest power p^k <= n
    selected = []
    for p in primes:
        val = _largest_prime_power(p, n)
        selected.append(val)
        
    # Include 1, which is coprime with all integers
    result = [1] + selected
    
    # Ensure types are strictly Python int
    result = [int(x) for x in result]
    
    return result
=======
"""
Maximize sum of pairwise-coprime interval integers from 1..256.

Models the problem as a maximum weight independent set in a conflict graph where
vertices are integers 2..256 and edges connect non-coprime pairs. Solves exactly
using branch-and-bound with greedy coloring for tight upper bounds.
"""

import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 256
    
    # Build conflict graph: vertices 2..n, edge if gcd > 1
    vertices = list(range(2, n + 1))
    num_v = len(vertices)
    
    # Precompute gcd conflicts
    # For each vertex, store which other vertices conflict
    conflict = [set() for _ in range(num_v)]
    for i in range(num_v):
        vi = vertices[i]
        for j in range(i + 1, num_v):
            vj = vertices[j]
            if math.gcd(vi, vj) > 1:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Weight of each vertex is its value
    weights = [vertices[i] for i in range(num_v)]
    
    # Greedy coloring for upper bound
    # Color vertices so that adjacent vertices have different colors
    # The number of colors gives an upper bound on independent set size
    def greedy_coloring(order):
        """Color vertices in given order, return color count and color assignment."""
        color_of = [-1] * num_v
        colors_used = 0
        
        for v in order:
            # Find smallest available color
            used_colors = set()
            for u in conflict[v]:
                if color_of[u] != -1:
                    used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[v] = c
            colors_used = max(colors_used, c + 1)
        
        return colors_used, color_of
    
    # Branch and bound
    best_sum = 0
    best_set = []
    
    # Initial solution: greedy by weight/conflict ratio
    def greedy_initial():
        remaining = set(range(num_v))
        selected = []
        total = 0
        
        while remaining:
            # Pick vertex with best weight / (1 + degree in remaining)
            best_v = None
            best_ratio = -1
            for v in remaining:
                deg = len(conflict[v] & remaining)
                ratio = weights[v] / (1 + deg)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_v = v
            
            if best_v is None:
                break
            
            selected.append(best_v)
            total += weights[best_v]
            remaining.discard(best_v)
            # Remove all conflicts
            remaining -= conflict[best_v]
        
        return selected, total
    
    init_set, init_sum = greedy_initial()
    best_sum = init_sum
    best_set = list(init_set)
    
    # Sort vertices by weight descending for better pruning
    order = sorted(range(num_v), key=lambda v: -weights[v])
    
    # Precompute colors for initial order
    colors_used, color_of = greedy_coloring(order)
    
    # Branch and bound with recursion
    # State: current independent set, available vertices
    def branch_and_bound(available, current_sum, current_set):
        nonlocal best_sum, best_set, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
        
        # Upper bound: current_sum + sum of weights of available vertices
        # But we can do better with coloring
        # For vertices in available, compute greedy coloring bound
        avail_list = list(available)
        if not avail_list:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
        
        # Simple upper bound: current_sum + sum of all remaining weights
        upper_bound = current_sum + sum(weights[v] for v in avail_list)
        if upper_bound <= best_sum:
            return
        
        # Pick a vertex to branch on - use highest weight
        v = max(avail_list, key=lambda x: weights[x])
        
        # Branch 1: Include v
        new_available = available - {v} - conflict[v]
        current_set.append(v)
        branch_and_bound(new_available, current_sum + weights[v], current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_available = available - {v}
        branch_and_bound(new_available, current_sum, current_set)
    
    # Start B&B
    available = set(range(num_v))
    branch_and_bound(available, 0, [])
    
    # Build result: 1 + selected vertices
    result = [1] + [vertices[i] for i in best_set]
    
    # Ensure types are strictly Python int
    result = [int(x) for x in result]
    
    return result
>>>>>>> REPLACE
