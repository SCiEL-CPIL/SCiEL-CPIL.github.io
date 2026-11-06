def construct():
    import time
    import math
    from collections import defaultdict

    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    N = 511
    # We consider candidates 1..511. 
    # Note: 1 is coprime to everything. w[1] = 1/511.
    # We want to maximize sum 1/(512-a).
    # This is equivalent to maximizing sum w[a] where w[a] = 1/(512-a).
    
    # 1. Pre-compute prime factors for each number to determine conflicts.
    # Two numbers conflict if they share a prime factor.
    # We can represent the "state" of selected numbers by the set of prime factors used.
    # However, the number of primes up to 511 is 97. A bitmask of 97 bits fits in a Python int.
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, p := i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    p_index = {p: i for i, p in enumerate(primes)}
    
    # For each number a in 1..511, compute its prime factor mask.
    # If a=1, mask is 0.
    # If a>1, mask has bits set for all prime factors of a.
    masks = [0] * (N + 1)
    weights = [0.0] * (N + 1)
    
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        if a == 1:
            masks[a] = 0
        else:
            x = a
            m = 0
            for p in primes:
                if p * p > x:
                    break
                if x % p == 0:
                    m |= (1 << p_index[p])
                    while x % p == 0:
                        x //= p
            if x > 1:
                m |= (1 << p_index[x])
            masks[a] = m

    # Sort candidates by weight descending to help branch and bound pruning.
    # We will decide inclusion in this order.
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda a: -weights[a])
    
    # Pre-compute suffix sums of weights for upper bound calculation.
    # suffix_sum[i] = sum of weights of candidates from i to end.
    # This provides a loose but fast upper bound if we ignore conflicts.
    # A tighter bound considers the mask conflicts.
    
    # Branch and Bound State:
    # current_index: next candidate to consider
    # current_mask: bitmask of prime factors used by selected candidates
    # current_score: sum of weights of selected candidates
    
    best_score = 0.0
    best_selection = []
    
    # Global time check
    def time_exhausted():
        return time.time() >= deadline

    # To make the bound tighter, we can pre-compute for each candidate index i,
    # the sum of weights of all subsequent candidates j > i that are compatible with a given mask?
    # That's too expensive (511 * 2^97).
    
    # Instead, we use a standard recursive B&B.
    # Pruning condition:
    # 1. If time is up, stop.
    # 2. If current_score + sum(weights[i:]) <= best_score, prune.
    #    (This is a valid upper bound because we can't add more weight than the sum of all remaining weights).
    #    To make it tighter, we can check if any remaining candidate is compatible.
    #    But sum of ALL remaining is a valid upper bound.
    #    We can improve this: 
    #    Upper Bound = current_score + sum(w[j] for j in range(i, N) if (masks[j] & current_mask) == 0)
    #    This is still O(N) per node. With 511 nodes, this might be slow if the tree is deep.
    #    However, the depth is at most 511.
    #    Let's try to optimize the bound calculation.
    
    # Alternative: Use a greedy heuristic to find a good initial solution to set best_score high early.
    # Greedy: Iterate candidates in weight order, add if compatible.
    greedy_mask = 0
    greedy_score = 0.0
    greedy_sel = []
    for a in candidates:
        if (masks[a] & greedy_mask) == 0:
            greedy_mask |= masks[a]
            greedy_score += weights[a]
            greedy_sel.append(a)
    
    best_score = greedy_score
    best_selection = list(greedy_sel)
    
    # Now run B&B to try to beat greedy_score.
    
    # We need to be careful with recursion depth. 511 is safe for Python default stack size (usually 1000).
    # But we can implement iteratively or just use recursion with sys.setrecursionlimit.
    import sys
    sys.setrecursionlimit(10000)
    
    # To speed up the bound calculation, we can pre-compute a list of (weight, mask) for candidates.
    # And maybe sort them such that high weights come first (already done).
    
    # Let's define the recursive function.
    # To avoid overhead of list slicing, we pass indices.
    
    # Optimization: 
    # If the sum of all remaining weights is less than the gap (best_score - current_score), prune.
    # We can pre-compute suffix sums of weights.
    suffix_w = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix_w[i] = suffix_w[i + 1] + weights[candidates[i]]
    
    # Even better bound:
    # For a given current_mask, the maximum additional weight is the sum of weights of compatible candidates.
    # Calculating this exactly at every node is O(N).
    # Is N=511 small enough? 
    # If the tree has 10^6 nodes, 10^6 * 511 = 5 * 10^8 operations. In Python, this is too slow (seconds to minutes).
    # We need a faster bound or a different approach.
    
    # Alternative Strategy: 
    # Since the graph is defined by prime factors, and primes are independent "resources",
    # this is a variant of the Knapsack problem with multiple constraints (one per prime).
    # But the number of primes is 97.
    
    # Let's stick to B&B but optimize the bound.
    # We can use a simpler bound: 
    # UB = current_score + sum(weights[i:]) 
    # This is very fast (O(1) with suffix sums).
    # It might be loose, leading to a larger search tree.
    # However, the weights decrease rapidly (1/1, 1/2, 1/3...). 
    # The greedy solution is likely very close to optimal.
    # If the greedy solution is near-optimal, the pruning `current_score + suffix_w[i] <= best_score` will be very effective.
    
    # Let's test this hypothesis.
    # If the greedy solution is not good enough, we might need to explore.
    
    # To make it more robust, we can add a second phase: local search on the best found set.
    # Or, we can improve the bound by checking compatibility of the *next* few items? No, that's complex.
    
    # Let's try the simple suffix sum bound first. If it's too slow, we'll fail the time limit.
    # But 511 items with exponential worst case is risky.
    # However, the "conflict" structure is dense. Many items conflict.
    # The effective branching factor might be low.
    
    # Let's add a "stronger" bound calculation that is still fast.
    # We can pre-compute for each prime p, the list of candidates divisible by p.
    # But checking compatibility requires checking all primes in current_mask.
    
    # Let's try a different angle: **Integer Linear Programming** using scipy?
    # scipy.optimize.milp is available.
    # Variables x_a in {0,1}.
    # Maximize sum w_a x_a.
    # Subject to: for each prime p, sum(x_a for a divisible by p) <= 1.
    # And sum x_a <= ... (no, just pairwise coprime implies each prime used at most once).
    # Wait, pairwise coprime means if a and b share prime p, they can't both be selected.
    # This is exactly: for each prime p, at most one multiple of p can be selected.
    # This is a set packing problem.
    # Number of constraints: 97 primes.
    # Number of variables: 511.
    # This is a very small MILP!
    # SciPy's HiGHS solver should solve this instantly.
    
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np

    # Build the constraint matrix A and bounds b for: A x <= b
    # For each prime p_i, sum_{a: p_i | a} x_a <= 1
    
    n_vars = N + 1 # indices 1..511, index 0 unused
    # We will map variable index i to number i+1? No, let's use 0..510 for numbers 1..511.
    # var index i corresponds to number i+1.
    
    c = np.zeros(n_vars)
    for i in range(n_vars):
        a = i + 1
        c[i] = -weights[a] # Minimize negative weight
        
    # Constraints:
    # For each prime p, sum of x_a for a divisible by p <= 1.
    A_rows = []
    b_rows = []
    
    for p in primes:
        row = np.zeros(n_vars)
        for i in range(n_vars):
            a = i + 1
            if a % p == 0:
                row[i] = 1.0
        A_rows.append(row)
        b_rows.append(1.0)
        
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    
    # Integrality: all variables are integers (binary)
    integrality = np.ones(n_vars)
    
    constraints = LinearConstraint(A, None, b)
    
    # Solve
    res = milp(c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
    
    if res.status == 0: # Optimal
        # Extract solution
        selected = []
        for i in range(n_vars):
            if res.x[i] > 0.5:
                selected.append(i + 1)
        
        # Verify pairwise coprime just in case
        valid = True
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if math.gcd(selected[i], selected[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = sum(1.0 / (512 - a) for a in selected)
            if score > best_score:
                best_score = score
                best_selection = selected
        else:
            # Fallback to greedy if MILP returned invalid (shouldn't happen)
            pass
    else:
        # If MILP failed or timed out, use greedy
        pass
        
    # If we still have time, we can try to improve the MILP solution with local search?
    # Or just return the MILP solution.
    # Given the size, MILP should be exact and fast.
    
    return best_selection

