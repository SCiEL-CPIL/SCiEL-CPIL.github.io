"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 75  # Safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
    
    # For each element x, store the pairs (y,z) such that {x,y,z} is a bad triple.
    # Use defaultdict with list for efficiency
    conflicts = defaultdict(list)
    
    # Fast conflict generation:
    # For each pair (a,b) with a<b, compute L = lcm(a,b)
    # Find all c > b such that lcm(a,c) == L and lcm(b,c) == L
    # This is equivalent to: c | L, c > b, and for all prime p:
    #   max(vp(a), vp(c)) == vp(L) and max(vp(b), vp(c)) == vp(L)
    # Since L = lcm(a,b), vp(L) = max(vp(a), vp(b))
    # So we need: max(vp(a), vp(c)) == max(vp(a), vp(b)) and max(vp(b), vp(c)) == max(vp(a), vp(b))
    
    # Precompute prime factorization for numbers 1..N
    # For each number, store its prime factorization as a dict
    def prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Precompute factorizations
    factorizations = [None] * (N + 1)
    for i in range(1, N + 1):
        factorizations[i] = prime_factors(i)
    
    # For each L (LCM value), precompute which numbers divide L
    # and their "compatible" pairs
    # Actually, let's just iterate over all pairs and find compatible c's
    
    # Better approach: For each L, find all divisors of L in [1,N]
    # Then for each triple of divisors (a,b,c) of L, check if L is the LCM of all three pairs
    
    # Even better: iterate a < b, compute L = lcm(a,b), then find all c > b 
    # such that c|L and the LCM conditions hold
    
    # Precompute: for each L, list of numbers in [1,N] that divide L
    # L can be at most lcm(1..128) which is huge, but we only care about L values that are lcm(a,b) for a,b in [1,128]
    
    # Let's use a different approach: for each c, find all pairs (a,b) with a<b<c 
    # such that {a,b,c} is a bad triple
    # For fixed c, we need: lcm(a,c) = lcm(b,c) = lcm(a,b) = L
    # This means a|L, b|L, c|L, and L is the LCM of each pair
    
    # Most efficient: iterate a<b<c but with early termination
    # For fixed a,b, L=lcm(a,b). c must divide L and c>b.
    # Also need lcm(a,c)=L and lcm(b,c)=L
    
    # Let's precompute for each L, the set of numbers in [1,N] that divide L
    # But L values can be large. Instead, for each pair (a,b), 
    # we can enumerate divisors of L that are > b and <= N
    
    def get_divisors_in_range(L, low, high):
        """Get all divisors of L in [low, high]"""
        result = []
        # Only check up to sqrt(L) for efficiency
        # But L can be large, so let's just iterate from low to high
        # Actually, for each c in range(low, high+1), check if c|L
        # This is O(N) per pair, and we have O(N^2) pairs, so O(N^3) total
        # Same as before. Let's think differently.
        
        # Alternative: precompute for each number c, which (a,b) pairs have L = lcm(a,b) 
        # such that c|L and the LCM conditions hold
        
        # Let's just do the direct check but optimize with bitmasks or sets
        for c in range(low, high + 1):
            if L % c == 0:
                # Check if lcm(a,c) == L and lcm(b,c) == L
                # lcm(a,c) = a*c // gcd(a,c)
                # We need a*c // gcd(a,c) == L
                # i.e., c // gcd(a,c) == L // a
                # Let g = gcd(a,c), then c/g * a = L, so a*c/g = L
                # This is equivalent to: a divides L and c divides L and 
                # max(vp(a), vp(c)) = vp(L) for all primes
                # Since we know c|L, we just need to check the LCM condition
                gc = gcd(a, c)
                if a * c // gc == L:
                    gb = gcd(b, c)
                    if b * c // gb == L:
                        result.append(c)
        return result
    
    # Build conflicts by iterating over all pairs
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            L = lcm[a][b]
            # Find all c > b such that c|L, lcm(a,c)=L, lcm(b,c)=L
            # Instead of iterating all c, note that c must be a divisor of L
            # and c > b, c <= N
            # Let's find divisors of L in (b, N]
            # For efficiency, precompute divisors for each possible L? 
            # L values are limited. Let's cache.
            
            # Simple approach: iterate c from b+1 to N
            # This is O(N) per pair, O(N^3) total ~ 2M operations, should be fast enough
            for c in range(b+1, N+1):
                if L % c != 0:
                    continue
                # c divides L, check LCM conditions
                gc = gcd(a, c)
                if a * c // gc != L:
                    continue
                gb = gcd(b, c)
                if b * c // gb != L:
                    continue
                # Found a bad triple (a, b, c)
                conflicts[a].append(frozenset([b, c]))
                conflicts[b].append(frozenset([a, c]))
                conflicts[c].append(frozenset([a, b]))
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        # S is a set of ints
        for pair in conflicts.get(x, []):
            # pair is a frozenset of two elements y, z
            # Check if both y and z are in S
            if pair.issubset(S):
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Heuristic: Start with empty set or small random set
    # We will run multiple restarts.
    
    def greedy_fill(S, order):
        """
        Add elements from 'order' to S if possible.
        Modifies S in place.
        """
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
                return True # Indicates something was added, might want to restart loop? 
                           # Standard greedy just goes through once. 
                           # But adding x might not block future y, but might block z.
                           # Since we check can_add dynamically, one pass is sufficient 
                           # for the given order. However, a newly added element 
                           # doesn't make an already rejected element valid again.
                           # So one pass is correct for fixed order.
        return False

    # Local search helper: try to improve S by removing one element and re-filling
    def local_search(S):
        """
        Attempts to improve S by 1-removal and re-greedy-fill.
        Returns True if S was improved (size increased).
        """
        current_size = len(S)
        if current_size == 0:
            return False
            
        # Get list of elements in S
        S_list = list(S)
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline - 10:
                break
                
            # Remove u
            S.remove(u)
            
            # Try to re-fill with multiple random orders to find a good completion
            best_completion = set(S)
            best_completion_size = len(S)
            
            # Try a few different orderings
            for trial in range(3):
                if time.time() > deadline - 10:
                    break
                S_trial = set(S)
                candidates = [x for x in range(1, N+1) if x not in S_trial]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x, S_trial):
                        S_trial.add(x)
                
                if len(S_trial) > best_completion_size:
                    best_completion_size = len(S_trial)
                    best_completion = set(S_trial)
            
            # Update S with the best completion
            S = set(best_completion)
            
            # Check if we improved
            if len(S) > current_size:
                return True
            else:
                # Size didn't increase, revert to original
                S.add(u)
                
        return False

    # Main loop
    num_restarts = 200 # Reduced due to tighter deadline
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Randomize the starting set or just start empty?
        # Starting empty and filling greedily with random order is a good baseline.
        S = set()
        order = elements[:]
        random.shuffle(order)
        
        greedy_fill(S, order)
        
        # Local Search Phase
        # Repeat local search until no improvement or time limit
        improved = True
        while improved and time.time() < deadline - 10:
            improved = local_search(S)
            
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
    # Final polish: try to add any remaining element to best_set
    if time.time() < deadline - 3:
        for x in elements:
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                
    if not best_set:
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
