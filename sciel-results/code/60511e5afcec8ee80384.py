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
                
                if a_prime == 1:
                    if len(S) > 2:
                        counts[a] += 1
                else:
                    if b in multiples[a_prime]:
                        if len(multiples[a_prime]) > 2:
                            counts[a] += 1
                    else:
                        if len(multiples[a_prime]) > 1:
                            counts[a] += 1
        
        return counts

    def can_add(candidate, S, multiples):
        """
        Check if candidate can be added to S without creating conflicts.
        """
        if candidate in S:
            return True
            
        # Check 1: For each a in S, does a | candidate * b for some b in S, b != a?
        # This creates a conflict triple (a, candidate, b) where a | candidate * b.
        for a in S:
            g = gcd(a, candidate)
            a_prime = a // g
            
            if a_prime == 1:
                # a | candidate * b is always true for any b.
                # We need b in S, b != a, b != candidate (candidate not in S).
                # So we just need b in S, b != a.
                if len(S) > 1:
                    return False
            else:
                # We need b in S such that a_prime | b and b != a.
                b_set = multiples[a_prime]
                if b_set:
                    if a not in b_set:
                        # Any b in b_set works
                        return False
                    else:
                        # Need another b
                        if len(b_set) > 1:
                            return False
        
        # Check 2: For each b, c in S, b != c, does candidate | b * c?
        # This creates a conflict triple (candidate, b, c) where candidate | b * c.
        for b in S:
            g = gcd(candidate, b)
            c_prime = candidate // g
            
            if c_prime == 1:
                # candidate | b * c is always true for any c.
                # We need c in S, c != b, c != candidate (candidate not in S).
                # So we just need c in S, c != b.
                if len(S) > 1:
                    return False
            else:
                # We need c in S such that c_prime | c and c != b.
                c_set = multiples[c_prime]
                if c_set:
                    if b not in c_set:
                        # Any c in c_set works
                        return False
                    else:
                        # Need another c
                        if len(c_set) > 1:
                            return False
        
        return True

    best_S = set()
    best_size = 0
    
    # Run multiple randomized greedy removals
    for run in range(50):
        if time.time() > deadline:
            break
            
        S = set(range(2, N + 1))
        
        # Rebuild multiples
        multiples = [set() for _ in range(N + 1)]
        for x in S:
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    multiples[d].add(x)
                    other = x // d
                    if other != d:
                        multiples[other].add(x)
        
        removed = []
        
        while True:
            if time.time() > deadline:
                break
                
            counts = check_conflicts()
            bads = [a for a, c in counts.items() if c > 0]
            
            if not bads:
                break
            
            # Select removal candidate
            max_count = max(counts[a] for a in bads)
            candidates = [a for a in bads if counts[a] == max_count]
            
            if len(candidates) == 1:
                a_to_remove = candidates[0]
            else:
                a_to_remove = random.choice(candidates)
            
            S.remove(a_to_remove)
            removed.append(a_to_remove)
            
            # Update multiples
            for d in range(1, int(a_to_remove**0.5) + 1):
                if a_to_remove % d == 0:
                    multiples[d].discard(a_to_remove)
                    other = a_to_remove // d
                    if other != d:
                        multiples[other].discard(a_to_remove)
        
        # Add-back phase
        # Try to add back elements from 'removed'
        # We can shuffle removed to randomize the order
        random.shuffle(removed)
        
        added_any = True
        while added_any:
            if time.time() > deadline:
                break
            added_any = False
            for x in list(removed):
                if x in S:
                    continue
                if can_add(x, S, multiples):
                    S.add(x)
                    removed.remove(x)
                    # Update multiples
                    for d in range(1, int(x**0.5) + 1):
                        if x % d == 0:
                            multiples[d].add(x)
                            other = x // d
                            if other != d:
                                multiples[other].add(x)
                    added_any = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Early stop if we found a very large set?
        if best_size > 60:
            break

    # Return best_S as a list of ints
    return list(best_S)
