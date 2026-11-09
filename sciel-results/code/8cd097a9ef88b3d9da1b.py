"""
Strategy: Exact Maximum Weight Independent Set via Branch and Bound.
The problem is to select a subset of (m, r) pairs such that:
1. m is in [20, 200]
2. m values are distinct
3. For any two selected pairs (m1, r1) and (m2, r2), r1 != r2 (mod gcd(m1, m2))
Maximize sum(1/m).

We model this as a graph where nodes are potential classes (m, r).
Edges connect conflicting pairs.
We want the Maximum Weight Independent Set.
Given the size (181 moduli, avg ~110 residues -> ~20k nodes), a naive B&B is too slow.
However, the constraint "distinct moduli" is strong.
We can iterate over subsets of moduli? No, too many.
Instead, we use a B&B on the set of available moduli, where the "weight" of adding a modulus m is 1/m, but we must choose a residue r.
Actually, the standard MWIS B&B works on nodes.
To make it efficient:
1. Pre-compute conflicts.
2. Use a greedy coloring upper bound for the B&B (standard technique for MWIS).
3. Order nodes by weight (1/m) descending. Smaller m first.

Wait, 20,000 nodes is too large for standard MWIS B&B in Python within 120s if the graph is dense.
But the graph is NOT dense. Two classes conflict only if their residues match mod gcd.
If gcd is 1, they never conflict.
Most pairs of numbers in [20, 200] are coprime or have small gcd.
The "distinct moduli" constraint is handled by the problem structure: we pick at most one r for each m.
So we can view this as: Select a subset M of {20..200}, and for each m in M, assign r_m such that no conflicts.
This is a Constraint Satisfaction Problem (CSP) with objective.

Let's refine the B&B:
State: (available_moduli, current_assignment, current_score)
This is still huge.

Alternative: Since the score is sum(1/m), and 1/m decreases rapidly, the optimal solution likely includes almost all small m.
Let's try a B&B that branches on moduli m from 20 upwards.
For each m, we branch on:
1. Exclude m.
2. Include m with residue r (try valid residues).

To prune:
Upper bound = current_score + sum(1/m for m in available_moduli).
If this <= best_score, prune.

To make it faster:
1. Pre-filter: If a modulus m cannot be added to the current partial assignment with ANY residue, we can prune the "Include" branch immediately, but we still have the "Exclude" branch.
2. Ordering: Process m in increasing order (20, 21, ...). This prioritizes high weights.
3. Residue ordering: For a given m, try residues that conflict with the fewest *future* potential moduli? Or just try 0, 1, ...?
   Actually, the choice of r for m affects which future m' can be included.
   We should try residues that leave the most options open for future m'.
   A good heuristic: For each candidate r, count how many future m' (m' > m) would be blocked for ALL their residues? No, that's expensive.
   Simpler: Just try residues 0 to m-1. The B&B will explore.

To speed up conflict checking:
Maintain a data structure of current assignments.
For a new (m, r), check against all current (m', r').
Conflict if (r - r') % gcd(m, m') == 0.

Optimization:
Precompute gcd(m, m') for all pairs.
In the B&B, maintain a list of active (m', r').
Check is O(N_active). N_active <= 181.
Total nodes in B&B tree: 2^181 is impossible.
But the bound is tight?
Sum(1/m) for m=20..200 is ~2.34.
If we find a solution with score 2.3, the bound for any branch excluding m=20 is 2.34 - 1/20 = 2.29.
If best is 2.3, we prune 2.29.
This suggests the B&B might be very effective if we find a good solution quickly.

We need a fast greedy to get a good initial lower bound (best_score).
Then run B&B to prove optimality or find better.

Greedy:
Iterate m=20..200.
For each m, pick r that minimizes conflicts with *already selected*? No, we are adding sequentially.
Pick r that is valid for current set.
To maximize future options, maybe pick r that is "least popular" mod small gcds?
Or just pick the first valid r.
Let's try multiple greedy variants to get a strong lower bound.

Then B&B:
def solve_bnb():
    best = greedy_result
    def dfs(idx, current_set, current_score):
        nonlocal best
        if idx == len(moduli):
            if current_score > best_score:
                best = current_set[:]
            return
        
        m = moduli[idx]
        
        # Bound: current_score + sum(1/m_i for i in idx..end)
        # Precompute suffix sums of 1/m
        bound = current_score + suffix_sum[idx]
        if bound <= best_score + 1e-12:
            return
            
        # Option 1: Exclude m
        dfs(idx + 1, current_set, current_score)
        
        # Option 2: Include m with some residue r
        # Check all r in 0..m-1
        for r in range(m):
            # Check if r is valid for current_set
            valid = True
            for m2, r2 in current_set:
                g = gcd_table[(m, m2)]
                if (r - r2) % g == 0:
                    valid = False
                    break
            if valid:
                current_set.append([m, r])
                dfs(idx + 1, current_set, current_score + 1.0/m)
                current_set.pop()
                
    dfs(0, [], 0.0)

This is still potentially exponential.
But with the bound, it might work.
The number of moduli is 181.
Depth 181.
Branching factor: 1 (exclude) + up to m (include).
But most includes are invalid or pruned by bound.

Key improvement: **Memoization or Symmetry Breaking?**
No, residues are distinct.

Let's add a "forward checking" heuristic.
Before branching on m, check if m *can* be added to the current set.
If no r works, we MUST exclude m. (Prune the include branch).
This reduces branching.

Also, we can order the residues for m such that we try the "most promising" ones first.
Promising = leaves more options for future?
Hard to compute.
Just try 0, 1, ..., m-1.

Let's implement this with a tight deadline.
"""

import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute gcd table
    gcd_table = {}
    for i in range(20, 201):
        for j in range(20, 201):
            gcd_table[(i, j)] = gcd(i, j)

    # Precompute suffix sums of 1/m for bounding
    # suffix_sum[i] = sum(1/m for m in moduli[i:])
    suffix_sum = [0.0] * (n_mod + 1)
    for i in range(n_mod - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / moduli[i]

    best_set = []
    best_score = 0.0

    def update_best(s, score):
        nonlocal best_set, best_score
        if score > best_score:
            best_score = score
            best_set = [x[:] for x in s]

    # 1. Greedy Initial Solution
    # Try a few greedy strategies to get a high lower bound
    def greedy(strategy_id):
        s = []
        for m in moduli:
            found = False
            # Determine residue order based on strategy
            if strategy_id == 0:
                residues = list(range(m))
            elif strategy_id == 1:
                residues = list(range(m-1, -1, -1))
            else:
                # Randomized? No, keep deterministic for reproducibility in this block
                residues = list(range(m))
            
            for r in residues:
                valid = True
                for m2, r2 in s:
                    g = gcd_table[(m, m2)]
                    if (r - r2) % g == 0:
                        valid = False
                        break
                if valid:
                    s.append([m, r])
                    found = True
                    break
        return s

    # Run a few greedy variants
    for strat in range(3):
        s = greedy(strat)
        score = sum(1.0/m for m, r in s)
        update_best(s, score)
        
    # 2. Branch and Bound
    # We use a stack-based DFS to avoid recursion limit and overhead
    # State: (idx, current_set, current_score)
    # We process moduli in order 20, 21, ...
    
    # To speed up conflict checking, we can maintain a list of (m, r) and check linearly.
    # Since max size is 181, linear check is ~181 ops.
    
    stack = [(0, [], 0.0)]
    
    while stack:
        if time.time() > deadline - 2.0:
            break
            
        idx, current_set, current_score = stack.pop()
        
        if idx == n_mod:
            update_best(current_set, current_score)
            continue
            
        m = moduli[idx]
        
        # Upper bound
        bound = current_score + suffix_sum[idx]
        if bound <= best_score + 1e-12:
            continue
            
        # Option 1: Exclude m
        # Push exclude branch first so include is processed first?
        # Actually, we want to find good solutions fast.
        # Including small m gives high score.
        # So push Exclude, then Include. Stack is LIFO, so Include is popped first.
        stack.append((idx + 1, current_set, current_score))
        
        # Option 2: Include m with valid residue
        # Check if any residue is valid
        # We can pre-check if m is "blocked" by current_set?
        # If for all r, there exists m2 in current_set s.t. r == r2 mod gcd(m, m2), then m is blocked.
        # If blocked, we don't push include branches.
        
        # To be efficient, iterate r and check.
        # We can break early if we find one valid? No, we need to explore all valid r.
        # But we can prune r if it's "dominated"? Hard to define.
        
        # Optimization: If we just want to find *a* valid r to improve score, we might stop?
        # No, B&B explores all.
        
        # Let's add a heuristic: Only try the first few valid residues?
        # No, that makes it heuristic again.
        # But if we have a tight bound, we might not need all.
        
        # Let's stick to trying all valid residues.
        for r in range(m):
            valid = True
            for m2, r2 in current_set:
                g = gcd_table[(m, m2)]
                if (r - r2) % g == 0:
                    valid = False
                    break
            if valid:
                # Push include branch
                # We copy current_set. This is O(N).
                # To save memory/time, we can use a persistent structure or just copy.
                # 181 elements copy is fast.
                new_set = current_set + [[m, r]]
                stack.append((idx + 1, new_set, current_score + 1.0/m))
                
    return best_set

# EVOLVE-BLOCK-END
