"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt

_P = 256
_DEADLINE = 80.0  # safety margin before 90s kill

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    def elapsed():
        return time.time() - start
    
    # Candidates are 2..256
    candidates = list(range(2, _P + 1))
    
    # Precompute: for each a, which pairs (b,c) with b<c, b,c in [2,256] satisfy a | b*c?
    # Actually we need: a does NOT divide b*c for all distinct a,b,c in the set.
    # So conflict: if a | b*c for some distinct b,c, then a cannot coexist with both b and c? 
    # No! The condition is: for EVERY triple (a,b,c) of distinct members, a ∤ b*c.
    # This means: for a given a in the set, and any two other distinct members b,c in the set, a ∤ b*c.
    # So if we have a set S, for each a in S, and for each pair {b,c} ⊂ S\{a} with b≠c, we need a ∤ b*c.
    
    # This is a 3-uniform hypergraph constraint. Let's think about it differently.
    # For a fixed a, define the "forbidden pairs" as all pairs {b,c} (b≠c, b,c≠a) such that a | b*c.
    # Then in the set S, for each a ∈ S, no forbidden pair for a can both be in S\{a}.
    
    # Greedy approach: build set incrementally.
    # Key insight: primes are "safe" in a sense. If a is prime p, then p | b*c iff p|b or p|c.
    # So for prime p, the forbidden pairs are those where at least one of b,c is divisible by p.
    # If S contains only numbers not divisible by p (other than p itself), then no forbidden pair exists for p.
    
    # Strategy: Start with all primes > 128 (since 2*128=256, primes >128 can't divide b*c for b,c ≤ 256 unless b or c is the prime itself... wait, if p>128 and b,c < p, then b*c < p^2 but could p divide b*c? Only if p|b or p|c, but b,c < p and b,c ≥ 2, so p can't divide them. So primes > 128 are always safe!)
    
    # Actually for any prime p, if b,c are both not divisible by p, then p ∤ b*c.
    # So the issue is when b or c is divisible by p.
    
    # Let me try a different approach: use the fact that if all numbers are "large enough" relative to each other...
    
    # Actually, let's just do a good greedy + local search.
    
    def check_valid_add(x, S):
        """Check if x can be added to set S (as a set of ints)."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if a * x % a == 0:  # This is trivially true, wrong check
                    pass
            # For each a in S, check that a ∤ x*b for any b in S\{a}
            # And for each a,b in S, check that a ∤ b*x and b ∤ a*x
        # More systematically:
        # Triple (x, a, b): x ∤ a*b for all a,b in S, a≠b
        # Triple (a, x, b): a ∤ x*b for all a,b in S, a≠b  
        # Triple (a, b, x): a ∤ b*x for all a,b in S, a≠b -- same as above
        
        # So we need:
        # 1. x ∤ a*b for all distinct a,b in S
        # 2. a ∤ x*b for all distinct a,b in S
        # 3. b ∤ a*x for all distinct a,b in S -- same as 2 with a,b swapped
        
        # Condition 1 and 2 are distinct.
        
        # Condition 1: x ∤ a*b for all distinct a,b in S
        # Condition 2: for each a in S, a ∤ x*b for all b in S\{a}
        
        # Check condition 1
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                if (S_list[i] * S_list[j]) % x == 0:
                    return False
        
        # Check condition 2
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
        
        return True
    
    # Greedy: try adding elements in various orders
    def greedy_build(order):
        S = []
        S_set = set()
        for x in order:
            if check_valid_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S
    
    # Various orderings to try
    random.seed(42)
    
    # Order 1: descending (larger numbers first)
    order1 = sorted(candidates, reverse=True)
    
    # Order 2: primes first (primes > 128 are always safe), then composites descending
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, isqrt(n) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Sort primes: large primes first (they're safer)
    primes_sorted = sorted(primes, reverse=True)
    composites_sorted = sorted(composites, reverse=True)
    
    order2 = primes_sorted + composites_sorted
    order3 = composites_sorted + primes_sorted
    order4 = sorted(candidates, key=lambda x: (-is_prime(x), -x))
    
    orders = [order1, order2, order3, order4]
    
    best_S = []
    for order in orders:
        if elapsed() > deadline - 5:
            break
        S = greedy_build(order)
        if len(S) > len(best_S):
            best_S = S[:]
    
    # Local search: try adding, then swapping
    S = best_S[:]
    S_set = set(S)
    candidates_set = set(candidates)
    
    def local_search(S, S_set):
        improved = True
        while improved and elapsed() < deadline - 3:
            improved = False
            
            # Try adding
            for x in candidates:
                if x in S_set:
                    continue
                if check_valid_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for x in candidates:
                if x in S_set:
                    continue
                for y in list(S):
                    S_temp = [v for v in S if v != y]
                    S_temp_set = set(S_temp)
                    if check_valid_add(x, S_temp_set):
                        S = S_temp + [x]
                        S_set = S_temp_set | {x}
                        improved = True
                        break
                if improved:
                    break
        
        return S, S_set
    
    S, S_set = local_search(S, S_set)
    
    # Try randomized restarts if time permits
    best_len = len(S)
    
    remaining = deadline - 3 - elapsed()
    if remaining > 2:
        for attempt in range(10):
            if elapsed() > deadline - 3:
                break
            # Randomized greedy
            random_order = candidates[:]
            random.shuffle(random_order)
            S_rand = greedy_build(random_order)
            if len(S_rand) > best_len:
                S, S_set = local_search(S_rand, set(S_rand))
                best_len = len(S)
    
    # Final validation
    S_final = sorted(S)
    # Verify
    valid = True
    n = len(S_final)
    for i in range(n):
        a = S_final[i]
        for j in range(i+1, n):
            b = S_final[j]
            for k in range(j+1, n):
                c = S_final[k]
                if (b * c) % a == 0:
                    valid = False
                    break
                if (a * c) % b == 0:
                    valid = False
                    break
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid set
        return [int(x) for x in range(2, _P+1) if is_prime(x)]
    
    return [int(x) for x in S_final]

# EVOLVE-BLOCK-END
