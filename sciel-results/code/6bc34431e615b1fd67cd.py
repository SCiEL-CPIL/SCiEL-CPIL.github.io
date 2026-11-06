"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    def is_subsum_free(S):
        """Check if set S is subsum-free."""
        S_set = set(S)
        for x in S_set:
            target = Fraction(1, x)
            others = [Fraction(1, y) for y in S_set if y != x]
            # Check if target can be formed by subset of others
            # Use set of achievable sums
            achievable = {Fraction(0)}
            for r in others:
                new_sums = set()
                for s in achievable:
                    ns = s + r
                    if ns <= target:  # prune: all reciprocals positive
                        new_sums.add(ns)
                achievable |= new_sums
            if target in achievable:
                return False
        return True
    
    def greedy_remove(S):
        """Iteratively remove subsumed elements until stable."""
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in sorted(S):
                if time.time() >= deadline:
                    break
                target = Fraction(1, x)
                others = [Fraction(1, y) for y in S if y != x]
                # Check if target is subset sum of others
                achievable = {Fraction(0)}
                found = False
                for r in others:
                    new_sums = set()
                    for s in achievable:
                        ns = s + r
                        if ns <= target:
                            new_sums.add(ns)
                    achievable |= new_sums
                    if target in achievable:
                        found = True
                        break
                if found:
                    S.remove(x)
                    changed = True
        return S
    
    # Start with all elements and remove subsumed ones
    S = greedy_remove(set(range(1, 25)))
    
    # Try greedy construction from different orderings to see if we can do better
    best = S
    
    # Try adding back removed elements in different orders
    # Also try greedy forward construction
    for order in [range(1, 25), range(24, 0, -1), 
                  [1,3,5,7,9,11,13,15,17,19,21,23,2,4,6,8,10,12,14,16,18,20,22,24],
                  [24,22,20,18,16,14,12,10,8,6,4,2,23,21,19,17,15,13,11,9,7,5,3,1]]:
        if time.time() >= deadline:
            break
        S_greedy = set()
        for x in order:
            if time.time() >= deadline:
                break
            # Check if adding x keeps set subsum-free
            candidate = S_greedy | {x}
            if is_subsum_free(candidate):
                S_greedy = candidate
        if len(S_greedy) > len(best):
            best = S_greedy
    
    return list(best)
# EVOLVE-BLOCK-END
