"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    
    # Key insight: for any pair of distinct moduli, a valid residue pair always exists.
    # The question is whether a global assignment exists for all 127 moduli.
    # Let's try to find one using greedy + local search.
    
    def check_all(assignment):
        """Verify all pairs are disjoint."""
        keys = list(assignment.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                m1, m2 = keys[i], keys[j]
                d = math.gcd(m1, m2)
                if d > 1 and (assignment[m1] - assignment[m2]) % d == 0:
                    return False
        return True
    
    def greedy_assign(order):
        """Greedy: assign residues in given order, skip if impossible."""
        assignment = {}
        for m in order:
            for r in range(m):
                ok = True
                for n, rn in assignment.items():
                    d = math.gcd(m, n)
                    if d > 1 and (r - rn) % d == 0:
                        ok = False
                        break
                if ok:
                    assignment[m] = r
                    break
        return assignment
    
    def local_search(assignment, max_iters=10000):
        """Try to fix conflicts by reassigning residues."""
        assignment = dict(assignment)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Find a conflict
            keys = list(assignment.keys())
            conflict_found = False
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    m1, m2 = keys[i], keys[j]
                    d = math.gcd(m1, m2)
                    if d > 1 and (assignment[m1] - assignment[m2]) % d == 0:
                        # Conflict: try to fix m1's residue
                        fixed = False
                        for r in range(m1):
                            if (r - assignment[m1]) % m1 == 0:
                                continue  # same residue
                            # Check if r works for m1 against all others
                            ok = True
                            for n, rn in assignment.items():
                                if n == m1:
                                    continue
                                dd = math.gcd(m1, n)
                                if dd > 1 and (r - rn) % dd == 0:
                                    ok = False
                                    break
                            if ok:
                                assignment[m1] = r
                                fixed = True
                                break
                        if fixed:
                            conflict_found = True
                            break
                if conflict_found:
                    break
            if not conflict_found:
                break
        return assignment
    
    best_count = 0
    best_assignment = None
    
    # Try multiple orderings
    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    orderings = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: (-num_distinct_primes(x), x)),
        sorted(moduli, key=lambda x: (num_distinct_primes(x), x)),
        sorted(moduli, key=lambda x: (-x.bit_length(), x)),
        sorted(moduli, key=lambda x: (x.bit_length(), x)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(10):
        o = moduli[:]
        rng.shuffle(o)
        orderings.append(o)
    
    for order in orderings:
        if time.time() > deadline:
            break
        asg = greedy_assign(order)
        count = len(asg)
        
        # Try local search to fix any remaining issues (though greedy shouldn't have any)
        # Actually greedy already ensures no conflicts, so just check count
        if count > best_count:
            best_count = count
            best_assignment = asg
    
    # If we got all 127, great. Otherwise, try harder with different strategies.
    # Try: start with all moduli, use constraint propagation
    
    # Alternative: formulate as exact cover / CSP and solve
    # For now, if we have a good solution, return it
    
    if best_assignment is not None:
        result = [[best_assignment[m], m] for m in sorted(best_assignment.keys())]
        return result
    
    # Fallback
    return [[1, 2]]
# EVOLVE-BLOCK-END

