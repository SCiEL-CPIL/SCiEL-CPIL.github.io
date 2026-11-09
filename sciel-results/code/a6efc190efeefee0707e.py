"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def _factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _solve_mwis():
    """
    Solve maximum weight independent set on the conflict graph.
    Nodes: integers 1..511, weight = 1/(512-a).
    Edge between i,j if gcd(i,j) > 1.
    Uses branch-and-bound with fractional knapsack upper bound.
    """
    N = 511
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute weights and factors
    weights = [Fraction(1, 512 - a) for a in range(1, N + 1)]
    factors = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        factors[a] = _factorize(a)
    
    # Build conflict graph using bitmasks
    # Node a conflicts with node b if gcd(a,b) > 1
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if factors[a] & factors[b]:
                # Set conflict bits
                conflict[a] |= (1 << b)
                conflict[b] |= (1 << a)
    
    # Sort nodes by weight descending for better pruning
    # We'll work with original indices but process in weight order
    node_order = sorted(range(1, N + 1), key=lambda x: -weights[x])
    
    best_solution = [False] * (N + 1)
    best_weight = Fraction(0)
    
    def branch_and_bound(candidates, current_set, current_weight):
        """
        candidates: list of candidate nodes (indices 1..511) that can still be added
        current_set: set of nodes already chosen
        current_weight: Fraction sum of weights so far
        """
        nonlocal best_weight, best_solution
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = [i in current_set for i in range(N + 1)]
            return
        
        # Upper bound: current_weight + sum of all remaining candidate weights
        # This is a loose bound; we can improve with greedy fractional knapsack
        remaining_weight = sum(weights[c] for c in candidates)
        if current_weight + remaining_weight <= best_weight:
            return
        
        # Pick the highest-weight candidate to branch on
        node = candidates[0]
        remaining = candidates[1:]
        
        # Branch 1: include node
        # Filter remaining to those not conflicting with node
        new_candidates = []
        for c in remaining:
            if not (conflict[node] >> c) & 1:
                new_candidates.append(c)
        
        new_set = set(current_set)
        new_set.add(node)
        branch_and_bound(new_candidates, new_set, current_weight + weights[node])
        
        # Branch 2: exclude node
        branch_and_bound(remaining, current_set, current_weight)
    
    branch_and_bound(node_order, set(), Fraction(0))
    
    # Extract solution
    result = [a for a in range(1, N + 1) if best_solution[a]]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    try:
        result = _solve_mwis()
        if result and len(result) > 0:
            return result
    except Exception:
        pass
    
    # Fallback: greedy solution
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    return chosen
# EVOLVE-BLOCK-END
