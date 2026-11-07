"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute divisibility info
    # For each a, we want to know which pairs (b, c) from 2..256 satisfy a | b*c
    # However, we only care about pairs within our current set S.
    # To speed up checking if adding x to S creates a violation:
    # A violation exists if there are a, b, c in S_new such that a | b*c.
    # Since S was valid, the only new violations involve x.
    # So we check:
    # 1. x | a*b for some a,b in S
    # 2. a | x*b for some a,b in S
    # 3. a | b*x for some a,b in S (same as 2)
    # Actually, the condition is: for every distinct a,b,c in S_new, a !| b*c.
    # If S is valid, adding x creates a violation if:
    # - There exist distinct a, b in S such that x | a*b
    # - There exists a in S such that a | x*b for some b in S, b!=a
    # - There exists b in S such that b | x*a for some a in S, a!=b
    # Note: a, b, c must be distinct.
    
    def is_valid_addition(x, S):
        """
        Check if adding x to set S maintains the property:
        For all distinct a, b, c in S U {x}, a does not divide b*c.
        """
        S_list = list(S)
        n = len(S_list)
        
        # Case 1: x is the 'divisor' a.
        # Need to check if x divides b*c for any distinct b, c in S.
        # If n < 2, no pair (b,c) exists.
        if n >= 2:
            for i in range(n):
                for j in range(i+1, n):
                    b = S_list[i]
                    c = S_list[j]
                    if (b * c) % x == 0:
                        return False
        
        # Case 2: x is one of the 'multiplicands' b or c.
        # Let x be b. We need to check if any a in S divides x*c for some c in S, c != a.
        # Distinctness: a != x (x not in S), c != x (x not in S), a != c.
        # So for each a in S, and each c in S such that c != a, check if a | x*c.
        # We can optimize: for each a in S, check if there exists c in S, c != a, such that a | x*c.
        # This is equivalent to: let g = gcd(a, x). Then a/g must divide c.
        # So we need to check if there is a c in S, c != a, such that c is a multiple of a/g.
        
        # Precompute multiples for speed? Or just check.
        # Since N is small, simple check is fine.
        for a in S_list:
            g = __import__('math').gcd(a, x)
            required_divisor = a // g
            # We need c in S, c != a, such that c % required_divisor == 0
            found = False
            for c in S_list:
                if c == a:
                    continue
                if c % required_divisor == 0:
                    found = True
                    break
            if found:
                return False
        
        return True

    def is_valid_set(S):
        """Verify the entire set is valid. Used for final check."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j: continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Strategy:
    # 1. Start with an empty set.
    # 2. Iterate through candidates in a specific order.
    # 3. Try to add each candidate. If it fits, add it.
    # 4. To maximize size, we should prioritize numbers that are "hard" to fit or "safe".
    #    Actually, primes are very safe. Composites are risky.
    #    Let's sort candidates by "conflict potential".
    #    Numbers with many divisors/multiples are risky.
    #    Primes are safest.
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True

    # Heuristic Ordering:
    # Add primes first (safest).
    # Then add numbers with fewest factors?
    # Or just iterate 2..256 and try to add.
    # A greedy approach:
    # Sort candidates. 
    # Try adding in that order.
    # If we get stuck, maybe we should have added a different number earlier?
    # This is a standard Greedy Set Packing / Independent Set problem.
    # Simple greedy might not be optimal.
    
    # Let's use a randomized greedy with multiple restarts.
    # Each restart:
    # 1. Shuffle candidates.
    # 2. Iterate, adding if valid.
    # 3. Record best size.
    
    best_size = 0
    best_set = set()
    
    # We can also try to improve a valid set by swapping.
    # But first, let's see how good greedy is.
    
    num_restarts = 50
    # Estimate time per restart:
    # 255 candidates. For each, check O(S^2) validity.
    # S grows up to ~100. 255 * 100^2 = 2.5M ops per restart.
    # 50 restarts = 125M ops. Might be too slow in Python.
    # Optimization: is_valid_addition is O(S^2) worst case.
    # We can precompute gcd or use the logic above which is O(S^2) but with small constants.
    # Let's reduce restarts or optimize the check.
    
    # Optimization for is_valid_addition:
    # Instead of nested loops for Case 1, we can precompute for each x, the pairs (b,c) that conflict?
    # No, that's 256^3.
    
    # Let's try a smarter greedy:
    # Start with all primes.
    # Then try to add composites.
    
    def greedy_construct(ordering):
        S = set()
        for x in ordering:
            if is_valid_addition(x, S):
                S.add(x)
        return S

    # Ordering 1: Primes first, then others
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Sort composites by "number of factors" or just random?
    # Let's try a few orderings.
    
    orderings = []
    # 1. Primes, then composites sorted by value
    orderings.append(primes + composites)
    # 2. Primes, then composites sorted by -value (larger composites might be safer? No, smaller are more likely to divide)
    orderings.append(primes + sorted(composites, reverse=True))
    # 3. Random shuffles
    for _ in range(20):
        rand_order = candidates[:]
        random.shuffle(rand_order)
        # Ensure primes are prioritized in random order?
        # Maybe just pure random.
        orderings.append(rand_order)
    
    # Run greedy for each ordering
    for i, ordering in enumerate(orderings):
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Now, try to improve best_set using local search (swaps)
    # If we have a valid set of size K, can we swap an element with another to get K+1?
    # Or swap to allow adding a new element?
    
    # Let's try a simple hill climbing on the set.
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Precompute "conflicts" for faster swap checks?
    # A swap (remove u, add v) is valid if:
    # 1. S' = (S - {u}) U {v} is valid.
    # This requires checking all triples in S'.
    # Since S is valid, only triples involving v can be invalid.
    # So we check if v creates violations with S - {u}.
    # This is similar to is_valid_addition(v, S - {u}).
    
    # We can iterate over all u in S and all v not in S.
    # If is_valid_addition(v, S - {u}), then we have a valid set of same size.
    # But we want LARGER size.
    # So we want to find a u, v such that after swapping, we can ADD another w?
    # This is a 2-step lookahead.
    
    # Alternative: 
    # Iterate through candidates not in S.
    # For each candidate v, check if there is a u in S such that (S - {u}) U {v} is valid 
    # AND allows adding some other w?
    # Too complex.
    
    # Let's just do single-step swaps to find a different valid set of same size, 
    # and hope it's a "better" configuration for future additions?
    # Or, more directly:
    # Try to add any x not in S.
    # If it fails, find a u in S that "conflicts" with x.
    # Remove u. Now x might be addable.
    # If x is added, size is same.
    # But maybe removing u allows adding x AND y?
    
    # Let's try a "Repair" strategy.
    # Take best_set.
    # Try to add every x not in S.
    # If x fails, identify the conflicting pairs.
    # Try removing one element from a conflicting pair to see if x can be added.
    # If yes, we have a set of same size.
    # Repeat until no more swaps improve "potential".
    
    # Actually, a better heuristic for MIS is:
    # 1. Find a maximal independent set (greedy).
    # 2. Try to expand it by removing one node and adding two? (Net +1)
    
    # Let's implement a simple expansion search:
    # For each u in S:
    #    S_temp = S - {u}
    #    For each v not in S:
    #        If v can be added to S_temp:
    #            S_new = S_temp U {v}
    #            Now try to add any w not in S_new.
    #            If w can be added:
    #                S_final = S_new U {w}
    #                Size increased by 1.
    #                Update best.
    
    # This is O(S * (N-S) * (N-S)) which might be heavy.
    # S ~ 100, N-S ~ 150. 100 * 150 * 150 = 2.25M checks.
    # Each check is O(S^2) ~ 10k. Total 22.5B ops. Too slow.
    
    # We need to be smarter.
    # Only try v that are "close" to being valid?
    # Or limit the search time.
    
    # Let's just do a random search on the state space of valid sets of size best_size.
    # Move: Swap u in S with v not in S, keeping validity.
    # Goal: Find a state where we can add a new element.
    
    # Given time constraints, the greedy + simple swap improvement is the best bet.
    # Let's add a time-bounded local search.
    
    time_limit_for_local = deadline - 5.0
    
    while time.time() < time_limit_for_local:
        improved = False
        # Try to add any element
        added = False
        for x in candidates:
            if x not in current_set:
                if is_valid_addition(x, current_set):
                    current_set.add(x)
                    current_size += 1
                    added = True
                    break # Restart adding loop? Or continue?
        
        if added:
            continue # Loop again to try adding more
        
        # If no addition possible, try a swap
        # Pick a random u in S and v not in S
        # If swap is valid, perform it.
        # This explores the neighborhood.
        
        # To be more effective, try swaps that "unblock" additions.
        # For each v not in S, check why it can't be added.
        # If it's because of u, try swapping u with something else?
        
        # Simple random walk on valid sets of same size.
        u = random.choice(list(current_set))
        v = random.choice([x for x in candidates if x not in current_set])
        
        S_test = current_set - {u}
        if is_valid_addition(v, S_test):
            current_set = S_test | {v}
            # Check if this allows an addition
            for w in candidates:
                if w not in current_set:
                    if is_valid_addition(w, current_set):
                        current_set.add(w)
                        current_size += 1
                        improved = True
                        break
        
        if time.time() > time_limit_for_local:
            break
            
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)

    return list(best_set)

# EVOLVE-BLOCK-END
