"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for i, a in enumerate(S):
            for b in S[i:]:  # a and b can be same element
                if a + b in S_set:
                    return False
        return True
    
    def greedy_sum_free(S, order):
        """Greedy: add elements in given order if it keeps the set sum-free."""
        chosen = []
        chosen_set = set()
        for x in order:
            # Check if adding x violates sum-free property
            valid = True
            # x + y = z or x + z = y or y + x = z for existing y,z in chosen
            # More precisely: for any a,b in chosen_set ∪ {x}, a+b must not be in chosen_set ∪ {x}
            # Since chosen is sum-free, we only need to check pairs involving x:
            # 1. x + x in chosen_set? -> 2x in chosen_set
            # 2. x + y in chosen_set? for y in chosen_set
            # 3. y + x in chosen_set? same as 2
            # 4. x + y = x -> y=0, but 0 not in sets (nonzero)
            # 5. y + z = x for y,z in chosen_set? No wait, that's not a violation.
            # Violation: a+b=c where a,b,c all in final set.
            # If we add x to a sum-free set T, violations are:
            # - x+x = c in T
            # - x+y = c in T for y in T
            # - y+z = x for y,z in T (but y+z=x means x is the sum, so a=y,b=z,c=x)
            # Wait, let me re-check. The condition is: no a+b=c with a,b,c in the set.
            # So if we have T (sum-free) and add x, new violations are:
            # 1. a+b=x where a,b in T
            # 2. x+b=c where b,c in T  
            # 3. a+x=c where a,c in T (same as 2)
            # 4. x+x=c where c in T
            # 5. x+x=x -> x=0, not possible
            # 6. x+b=x -> b=0, not possible
            # 7. a+x=x -> a=0, not possible
            
            # So check:
            # - 2x not in chosen_set
            # - For each y in chosen_set: x+y not in chosen_set
            # - For each y in chosen_set: y+x not in chosen_set (same)
            # - For each pair y,z in chosen_set: y+z != x
            
            if 2 * x in chosen_set:
                valid = False
            else:
                for y in chosen_set:
                    if x + y in chosen_set:
                        valid = False
                        break
                if valid:
                    # Check if x is the sum of two elements in chosen_set
                    # This is O(n^2) per check, might be slow. Let's use a set.
                    for y in chosen_set:
                        if x - y in chosen_set and x - y != y or (x - y == y and 2*y == x):
                            # Actually, we need to check if there exist y,z in chosen_set with y+z=x
                            # We can do: for y in chosen_set, check if x-y in chosen_set
                            # But we need to be careful about y==z case
                            valid = False
                            break
            
            if valid:
                chosen.append(x)
                chosen_set.add(x)
        return chosen
    
    def heuristic_order(S):
        """Order elements to try first: larger absolute values tend to be harder to form as sums."""
        return sorted(S, key=lambda x: -abs(x))
    
    def try_partition(S):
        """Try to find a large sum-free subset using various heuristics."""
        best = []
        
        # Strategy 1: All odd numbers
        odds = [x for x in S if x % 2 != 0]
        if is_sum_free(odds):
            best = odds
        
        # Strategy 2: Greedy with different orderings
        # Try descending by absolute value
        order1 = heuristic_order(S)
        result1 = greedy_sum_free(S, order1)
        if len(result1) > len(best):
            best = result1
        
        # Try ascending by absolute value
        order2 = sorted(S, key=lambda x: abs(x))
        result2 = greedy_sum_free(S, order2)
        if len(result2) > len(best):
            best = result2
        
        # Try positive numbers first
        order3 = sorted([x for x in S if x > 0] + [x for x in S if x < 0], key=lambda x: -x)
        result3 = greedy_sum_free(S, order3)
        if len(result3) > len(best):
            best = result3
        
        # Try negative numbers first
        order4 = sorted([x for x in S if x < 0] + [x for x in S if x > 0], key=lambda x: -x)
        result4 = greedy_sum_free(S, order4)
        if len(result4) > len(best):
            best = result4
        
        return best
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        
        # Start with odd numbers as baseline
        odds = [x for x in S if x % 2 != 0]
        best = odds
        
        # Try greedy heuristics
        candidate = try_partition(S)
        if len(candidate) > len(best):
            best = candidate
        
        # If we have time, try local search: remove one element, add another
        if time.time() < deadline:
            current = set(best)
            current_list = list(best)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing each element and see if we can add a better one
                for i in range(len(current_list)):
                    if time.time() >= deadline:
                        break
                    x = current_list[i]
                    # Temporarily remove x
                    current.discard(x)
                    # Try adding each element not in current
                    for y in S:
                        if y in current:
                            continue
                        # Check if current ∪ {y} is sum-free
                        valid = True
                        if 2 * y in current:
                            valid = False
                        else:
                            for z in current:
                                if y + z in current:
                                    valid = False
                                    break
                            if valid:
                                for z in current:
                                    if y - z in current:
                                        valid = False
                                        break
                        if valid:
                            # Add y, remove x
                            current.add(y)
                            current_list[i] = y
                            improved = True
                            break
                    if improved:
                        break
                    else:
                        # Put x back
                        current.add(x)
                if time.time() >= deadline:
                    break
            
            best = list(current)
        
        results.append(best)
        
        if time.time() > deadline:
            # Return what we have for remaining instances using just odds
            for s, sd in ((60, 801), (120, 802), (240, 803))[len(results):]:
                S_rem = instance(s, sd)
                results.append([x for x in S_rem if x % 2 != 0])
            break
    
    return results
# EVOLVE-BLOCK-END
