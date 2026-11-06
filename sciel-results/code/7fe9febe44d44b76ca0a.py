"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(vals):
        odds = [x for x in vals if x % 2 != 0]
        evens = [x for x in vals if x % 2 == 0]
        
        # Filter evens that are compatible with the fixed odds
        # An even e is compatible if for all o in odds, e+o is not in odds
        # (Since e+o is odd, if it's in the instance, it's in odds)
        # Also 2e is even, so it can't be in odds.
        # Wait, the condition is: no a,b,c in S such that a+b=c.
        # If we add e to S_odds, we check:
        # 1. e + o = c. c must be in S.
        #    If c is odd, c in S_odds. So if e+o in S_odds, conflict.
        #    If c is even, c in E_sub. But we haven't added other evens yet in this check.
        #    However, if we add e, and later add c, then e+o=c is a conflict.
        #    So e is "bad" if e+o is an odd number in the instance.
        #    Because if e+o is in the instance, it's in S_odds (since it's odd).
        #    Then e + o = e+o (which is in S) is a violation.
        
        compatible_evens = []
        odds_set = set(odds)
        for e in evens:
            is_compat = True
            for o in odds:
                if (e + o) in odds_set:
                    is_compat = False
                    break
            if is_compat:
                compatible_evens.append(e)
        
        # Now find max subset of compatible_evens that is sum-free internally
        # Constraint: for any a, b in E_sub, a+b must not be in E_sub.
        # (a+b is even, so it can't be in S_odds)
        
        best_subset = []
        best_size = 0
        
        # Greedy with random restarts
        # Limit restarts based on time
        start_time = time.time()
        max_restarts = 1000
        rng = random.Random(42)
        
        for _ in range(max_restarts):
            if time.time() - start_time > 10: # 10s per instance is plenty
                break
            
            shuffled = compatible_evens[:]
            rng.shuffle(shuffled)
            
            current_subset = []
            current_set = set()
            
            for e in shuffled:
                # Check if e can be added
                # 1. 2e not in current_set
                if 2 * e in current_set:
                    continue
                # 2. For all x in current_set, e+x not in current_set
                conflict = False
                for x in current_subset:
                    if (e + x) in current_set:
                        conflict = True
                        break
                if not conflict:
                    current_subset.append(e)
                    current_set.add(e)
            
            if len(current_subset) > best_size:
                best_size = len(current_subset)
                best_subset = current_subset[:]
        
        return odds + best_subset

    # Generate instances
    instances = [
        instance(60, 801),
        instance(120, 802),
        instance(240, 803)
    ]
    
    results = []
    for vals in instances:
        results.append(solve_instance(vals))
        
    return results
# EVOLVE-BLOCK-END
