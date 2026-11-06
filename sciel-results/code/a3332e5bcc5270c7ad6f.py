"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find a maximum-weight subset of [1, 256] with no non-constant 3-term AP,
    where weight of element x is 1/x.
    
    Uses branch-and-bound on the decision variables (include/exclude each element),
    ordered by decreasing weight (i.e., increasing x is NOT the order; we go by 1/x
    which means smaller x has larger weight, so we process x=1,2,3,... in order of
    decreasing 1/x).
    
    Actually, weight 1/x is decreasing in x, so the heaviest elements are the small ones.
    We process elements in order of decreasing weight: x=1, 2, 3, ..., 256.
    
    For each element, we decide include or exclude. We maintain:
    - current set S (as a frozenset or set)
    - current sum of reciprocals (as Fraction)
    - an upper bound on the remaining possible sum
    
    The upper bound is computed by taking all remaining elements that could potentially
    be added (i.e., don't form a 3-AP with any pair in S or with each other in the
    remaining set). A simple upper bound is the sum of 1/x for all remaining elements
    that don't immediately conflict with S. A tighter bound accounts for conflicts
    among the remaining elements themselves, but that's expensive.
    
    For the initial upper bound, we can use the sum of 1/x for all remaining elements
    that are individually compatible with S (i.e., adding them to S doesn't create a 3-AP).
    This is an upper bound because it ignores conflicts among the remaining elements.
    
    We also need to check: if we include element x, does it form a 3-AP with any two
    elements already in S? A 3-AP is (a, b, c) with a+b=2b, i.e., a+c=2b. So x forms
    a 3-AP with S if there exist a, b in S such that x is the middle term (a+c=2x with
    a,c in S) or x is an endpoint (x+b=2c or x+c=2b with b,c in S, meaning 2b-x in S
    or 2c-x in S).
    
    More precisely, x conflicts with S if:
    1. There exist a, c in S with a + c = 2x (x is the middle)
    2. There exists b in S such that 2b - x is in S and 2b - x != x (x is an endpoint, b is middle)
       i.e., 2b - x in S and 2b - x != x
    3. There exists b in S such that 2x - b is in S and 2x - b != x (x is the other endpoint)
    
    Actually, cases 2 and 3 are symmetric. Let's just check: for any b in S, if (2b - x) in S
    and 2b - x != x, then (2b-x, b, x) is a 3-AP. Also for any a, c in S, if a + c = 2x,
    then (a, x, c) is a 3-AP.
    
    To make this efficient, we can precompute for each x the set of pairs that would form
    a 3-AP with x. But since S changes, we need to check dynamically.
    
    Given the small size (256 elements), and the fact that we want to maximize 1/x sum,
    the greedy approach already gives a reasonable answer. Let's think about what the
    optimal solution looks like.
    
    The reciprocal sum is dominated by small numbers. 1/1 = 1, 1/2 = 0.5, 1/3 ≈ 0.333, etc.
    The greedy algorithm picks 1, then 2 (1,2 don't form a 3-AP since we need 3 elements),
    then 3? 1,2,3 is a 3-AP (1+3=2*2), so 3 is excluded. Then 4? Check 1,2,4: 1+4=5≠4, 
    2+4=6≠2, 1+2=3≠8... wait, need to check all triples. 1,2,4: middle term would be 2.5
    (not integer) for (1,?,4), or (1,2,?) → ?=3, (2,4,?) → ?=6. So 1,2,4 has no 3-AP.
    So greedy picks 4. Then 5? Check with {1,2,4}: (1,3,5)→3 not in S, (2,?,5)→? = 3.5 no,
    (4,?,5)→? = 4.5 no, (1,?,5)→?=3 no, (2,?,5)→ no. Wait, let me be more careful.
    
    A 3-AP (a,b,c) with a<b<c has a+c=2b. So for x=5 and S={1,2,4}:
    - Is there a,c in S with a+c=10? 1+9 no, 2+8 no, 4+6 no.
    - Is there b in S with 2b-5 in S and 2b-5≠5? b=1: 2-5=-3 no. b=2: 4-5=-1 no. b=4: 8-5=3 no.
    - Is there b in S with 2*5-b in S and 2*5-b≠5? b=1: 10-1=9 no. b=2: 10-2=8 no. b=4: 10-4=6 no.
    So 5 can be added. S={1,2,4,5}.
    
    This is getting complex. Let me just implement a branch-and-bound with a time limit.
    """
    start_time = time.time()
    DEADLINE = 80.0  # seconds, leaving 10s margin
    
    n = 256
    # Elements in order of decreasing weight (1/x), so x=1,2,...,256
    elements = list(range(1, n+1))
    weights = [Fraction(1, x) for x in elements]
    
    # Precompute for each x, the list of (a, c) pairs such that a + c = 2x and a != c
    # Also for each x and b, check if 2b-x is valid
    # Actually, let's precompute conflicts differently.
    
    # For a given set S, x is compatible if:
    # - No a, c in S with a + c = 2x (a != c)
    # - No b in S with 2b - x in S and 2b - x != x
    # - No b in S with 2x - b in S and 2x - b != x
    
    # Note: the second and third conditions are equivalent by symmetry.
    # 2b - x in S means (2b-x, b, x) is a 3-AP.
    # 2x - b in S means (b, x, 2x-b) is a 3-AP.
    
    # Let's precompute for each x, the set of "forbidden pairs" in terms of what elements
    # in S would conflict. Actually, it's easier to just check on the fly.
    
    def is_compatible(x, S):
        """Check if x can be added to S without forming a 3-AP."""
        S_set = S  # S is already a set
        # Check if x is the middle term: a + c = 2x for some a, c in S
        for a in S_set:
            c = 2 * x - a
            if c in S_set and c != a:
                return False
        # Check if x is an endpoint: for some b in S, 2b - x in S
        for b in S_set:
            other = 2 * b - x
            if other in S_set and other != x:
                return False
        return True
    
    best_set = set()
    best_weight = Fraction(0)
    
    # Greedy initial solution
    greedy_set = set()
    for x in elements:
        if is_compatible(x, greedy_set):
            greedy_set.add(x)
    greedy_weight = sum(Fraction(1, x) for x in greedy_set)
    best_set = set(greedy_set)
    best_weight = greedy_weight
    
    # Branch and bound
    # State: index into elements, current set S, current weight
    # We process elements in order 1, 2, ..., 256 (decreasing weight)
    
    def upper_bound(idx, S, current_weight):
        """
        Upper bound on the total weight if we make optimal choices from elements[idx:].
        Simple bound: current_weight + sum of 1/x for x in elements[idx:] that are
        individually compatible with S.
        """
        bound = current_weight
        S_set = S
        for i in range(idx, n):
            x = elements[i]
            if is_compatible(x, S_set):
                bound += weights[i]
        return bound
    
    def branch_and_bound(idx, S, current_weight):
        nonlocal best_set, best_weight, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if idx == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(S)
            return
        
        # Upper bound check
        ub = upper_bound(idx, S, current_weight)
        if ub <= best_weight:
            return
        
        x = elements[idx]
        
        # Option 1: Include x if compatible
        if is_compatible(x, S):
            S_new = set(S)
            S_new.add(x)
            branch_and_bound(idx + 1, S_new, current_weight + weights[idx])
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, S, current_weight)
    
    # The full branch-and-bound will be too slow for 256 elements.
    # We need a smarter approach.
    
    # Alternative: Since the weights are 1/x, and small x have much larger weights,
    # the optimal solution will include most small numbers. Let's try a local search
    # starting from the greedy solution.
    
    # Actually, let's think about this more carefully. The problem is NP-hard in general,
    # but for n=256 and the specific structure of 3-APs, we might be able to do well.
    
    # Let's try a different approach: start with the greedy solution, then try to improve
    # by swapping elements or adding/removing elements.
    
    # But first, let's see if we can do a limited branch-and-bound on just the first
    # K elements (the heaviest ones), and greedily fill in the rest.
    
    # The first K elements have weights 1, 1/2, 1/3, ..., 1/K. The sum of the rest is
    # bounded by the harmonic tail, which is small.
    
    # Let's try K=20. The sum of 1/x for x=21..256 is about ln(256) - ln(20) ≈ 5.5 - 3.0 = 2.5.
    # The greedy solution on the first 20 elements will have weight at least 1 + 1/2 + 1/4 + 1/5 + ... 
    # which is already > 3.
    
    # Actually, let me just implement a smarter local search.
    
    # Local search: start from greedy, try all single additions, removals, and swaps.
    
    def try_add(x, S):
        if x in S:
            return False
        if is_compatible(x, S):
            S.add(x)
            return True
        return False
    
    def try_remove(x, S):
        if x in S:
            S.discard(x)
            return True
        return False
    
    def local_search(S, max_iterations=1000):
        S = set(S)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() - start_time < DEADLINE:
            improved = False
            iteration += 1
            # Try additions
            for x in range(1, n+1):
                if x not in S:
                    S_trial = set(S)
                    S_trial.add(x)
                    # Check validity
                    valid = True
                    for a in S_trial:
                        for b in S_trial:
                            if a < b:
                                c = 2*b - a
                                if c in S_trial and c != a and c != b:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        S = S_trial
                        improved = True
                        break
            if improved:
                continue
            # Try swaps: remove x, add y
            for x in list(S):
                for y in range(1, n+1):
                    if y in S:
                        continue
                    S_trial = set(S)
                    S_trial.discard(x)
                    S_trial.add(y)
                    # Check validity
                    valid = True
                    for a in S_trial:
                        for b in S_trial:
                            if a < b:
                                c = 2*b - a
                                if c in S_trial and c != a and c != b:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid and Fraction(1, y) > Fraction(1, x):
                        S = S_trial
                        improved = True
                        break
                if improved:
                    break
        return S
    
    # Run local search from greedy
    improved_set = local_search(best_set)
    improved_weight = sum(Fraction(1, x) for x in improved_set)
    if improved_weight > best_weight:
        best_set = improved_set
        best_weight = improved_weight
    
    # Try local search from other starting points
    # Start with just {1}
    for seed in [{1}, {1, 2}, {1, 2, 4}, {1, 2, 4, 5}]:
        if time.time() - start_time > DEADLINE - 5:
            break
        S_seed = set(seed)
        # Greedily fill in
        for x in elements:
            if x not in S_seed and is_compatible(x, S_seed):
                S_seed.add(x)
        S_ls = local_search(S_seed)
        w = sum(Fraction(1, x) for x in S_ls)
        if w > best_weight:
            best_weight = w
            best_set = S_ls
    
    return list(best_set)

# EVOLVE-BLOCK-END
