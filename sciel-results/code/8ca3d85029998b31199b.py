import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial set
    N = 256
    S = set(range(2, N + 1))
    
    # Precompute multiples for fast lookup
    # multiples[k] is a set of x in S such that k divides x
    multiples = [set() for _ in range(N + 1)]
    for x in S:
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                multiples[d].add(x)
                other = x // d
                if other != d:
                    multiples[other].add(x)
    
    def check_conflicts():
        """
        Returns a dict {a: count} for a in S with conflicts.
        Conflict for a: exists b, c in S, b != c, b != a, c != a such that a | b*c.
        """
        counts = {a: 0 for a in S}
        for a in S:
            for b in S:
                if b == a:
                    continue
                g = gcd(a, b)
                a_prime = a // g
                # We need c in multiples[a_prime] such that c != a and c != b
                # If a_prime is 1, multiples[1] is S.
                # We need at least one c in multiples[a_prime] \ {a, b}
                
                # Optimization: if len(multiples[a_prime]) - (1 if a in multiples[a_prime] else 0) - (1 if b in multiples[a_prime] else 0) > 0
                # Actually, just check if the set difference is non-empty.
                # For speed, we can just check sizes.
                # But we need to be sure a and b are removed from consideration.
                
                # A faster check:
                # If a_prime == 1:
                #   We need c in S \ {a, b}. Exists if len(S) > 2.
                # Else:
                #   We need c in multiples[a_prime] \ {a, b}.
                
                if a_prime == 1:
                    if len(S) > 2:
                        counts[a] += 1
                else:
                    # Check if there is any c in multiples[a_prime] that is not a and not b
                    # Since a is always in multiples[a_prime] (a | a), and b might be.
                    # We just need len(multiples[a_prime]) - (1 if a in multiples[a_prime] else 0) - (1 if b in multiples[a_prime] else 0) > 0
                    # Note: a is always in multiples[a_prime] because a_prime | a.
                    # b is in multiples[a_prime] if a_prime | b.
                    
                    # Simpler: just iterate? No, use set logic.
                    # To avoid creating new sets, check:
                    # Is there an element in multiples[a_prime] other than a and b?
                    # If len(multiples[a_prime]) > 1:
                    #   If a in multiples[a_prime]:
                    #       If b in multiples[a_prime]:
                    #           if len(multiples[a_prime]) > 2: conflict
                    #       else:
                    #           if len(multiples[a_prime]) > 1: conflict
                    #   else: # a not in multiples? Impossible.
                    # So:
                    #   if a_prime == 1: handled above.
                    #   else:
                    #       if b in multiples[a_prime]:
                    #           if len(multiples[a_prime]) > 2: counts[a] += 1
                    #       else:
                    #           if len(multiples[a_prime]) > 1: counts[a] += 1
                    #       # Wait, what if a is NOT in multiples? Impossible.
                    #       # What if b is NOT in multiples? Then we just need len > 1 (for a and c).
                    
                    if b in multiples[a_prime]:
                        if len(multiples[a_prime]) > 2:
                            counts[a] += 1
                    else:
                        if len(multiples[a_prime]) > 1:
                            counts[a] += 1
        
        return counts

    best_S = set()
    best_size = 0
    
    # Run multiple randomized greedy removals
    for run in range(50):
        if time.time() > deadline:
            break
            
        S = set(range(2, N + 1))
        
        # Rebuild multiples for this run? Or just use the global one and filter?
        # Since S changes, multiples must change.
        # Rebuilding is O(N^2 log N) approx. 256^2 * 8 ~ 500k ops. Fast.
        multiples = [set() for _ in range(N + 1)]
        for x in S:
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    multiples[d].add(x)
                    other = x // d
                    if other != d:
                        multiples[other].add(x)
        
        while True:
            if time.time() > deadline:
                break
                
            counts = check_conflicts()
            bads = [a for a, c in counts.items() if c > 0]
            
            if not bads:
                break
            
            # Select removal candidate
            # Strategy: Remove the one with highest count, tie-break randomly
            max_count = max(counts[a] for a in bads)
            candidates = [a for a in bads if counts[a] == max_count]
            
            if len(candidates) == 1:
                a_to_remove = candidates[0]
            else:
                a_to_remove = random.choice(candidates)
            
            S.remove(a_to_remove)
            
            # Update multiples
            for d in range(1, int(a_to_remove**0.5) + 1):
                if a_to_remove % d == 0:
                    multiples[d].discard(a_to_remove)
                    other = a_to_remove // d
                    if other != d:
                        multiples[other].discard(a_to_remove)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Early stop if we found a very large set?
        # Theoretical max is likely around 50-60.
        if best_size > 60:
            break

    # Return best_S as a list of ints
    return list(best_S)
