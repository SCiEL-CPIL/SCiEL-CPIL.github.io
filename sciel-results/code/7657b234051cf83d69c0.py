import random
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0  # 15s margin
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    all_numbers = list(range(2, 41))
    
    # Pre-compute prime factorization vectors for fast collision checking?
    # Or just use set of products.
    
    def is_valid_set(S):
        """Check if all subset products are distinct."""
        # Optimization: Sort S
        S_sorted = sorted(S)
        prods = {1}
        for x in S_sorted:
            new_prods = []
            for p in prods:
                np = p * x
                if np in prods:
                    return False
                new_prods.append(np)
            prods.update(new_prods)
        return True

    def check_addition(S, x):
        """Check if x can be added to S without collision."""
        # S is a list, x is an int
        # We need to check if any product of subset of S equals x * product of subset of S
        # This is equivalent to checking if x * (subset of S) intersects (subset of S)
        # Faster: Just generate products of S and see if x * p in prods for any p in prods?
        # No, that's not right.
        # If S is valid, prods(S) has 2^|S| elements.
        # If we add x, we need prods(S) and x * prods(S) to be disjoint.
        # So we just need to check if any p in prods(S) has x * p in prods(S).
        
        # To be fast, we can maintain the set of products.
        # But for a single check, we can just do:
        # For each p in prods(S), if x * p in prods(S), return False.
        # This requires prods(S).
        
        # Since we are doing local search, we should maintain prods.
        # For now, this function is for the greedy step where S is small.
        
        # Actually, a simpler check:
        # If S is valid, the set of products P is known.
        # We need P and x*P to be disjoint.
        # So check if any x*p is in P.
        
        # We don't have P here. Let's assume this is called when S is small enough
        # or we pass P.
        # For the main loop, we will maintain P.
        
        pass

    best_result = []
    best_size = 0
    
    # Strategy 1: Greedy with Random Restart
    # Start with all primes, remove some, add composites.
    
    # Hardcoded good start: 17 elements
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (10)
    # Composites: 4, 16, 32, 9, 27, 25, 12 (7) -> Total 17
    # Let's verify this specific set:
    # S = [4, 5, 7, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37]
    # Check 12: 12 = 2^2 * 3.
    # Products involving 12:
    # 12 * 5 = 60
    # 12 * 4 = 48
    # 12 * 9 = 108
    # 12 * 16 = 192
    # 12 * 25 = 300
    # 12 * 27 = 324
    # 12 * 32 = 384
    # None of these are in S.
    # Are any products of other elements equal to 12?
    # 4 * 3 (3 not in S)
    # 2 * 6 (2,6 not in S)
    # So 12 is safe.
    
    # Check collisions among composites:
    # 4, 16, 32 (Powers of 2: 2^2, 2^4, 2^5). Distinct.
    # 9, 27 (Powers of 3: 3^2, 3^3). Distinct.
    # 25 (5^2).
    # 12 (2^2 * 3).
    
    # Cross products:
    # 4 * 9 = 36 (Not in S)
    # 4 * 27 = 108 (Not in S)
    # 16 * 9 = 144
    # 32 * 9 = 288
    # 4 * 25 = 100
    # 9 * 25 = 225
    # 12 * 4 = 48
    # 12 * 9 = 108
    # 12 * 16 = 192
    # 12 * 25 = 300
    # 12 * 27 = 324
    # 12 * 32 = 384
    
    # Any product of two composites equal to another composite?
    # 4 * 9 = 36.
    # 4 * 12 = 48.
    # 9 * 12 = 108.
    # 16 * 12 = 192.
    # 25 * 4 = 100.
    # 27 * 4 = 108.
    # 32 * 12 = 384.
    
    # Any product of a composite and a prime equal to another element?
    # 4 * 5 = 20 (No)
    # 4 * 7 = 28 (No)
    # 9 * 5 = 45 (No)
    # 12 * 5 = 60 (No)
    # 16 * 5 = 80 (No)
    # 25 * 5 = 125 (No)
    # 27 * 5 = 135 (No)
    # 32 * 5 = 160 (No)
    # 4 * 7 = 28
    # 9 * 7 = 63
    # 12 * 7 = 84
    # 16 * 7 = 112
    # 25 * 7 = 175
    # 27 * 7 = 189
    # 32 * 7 = 224
    
    # It seems this set of 17 is valid.
    
    candidate_17 = [4, 5, 7, 9, 11, 12, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37]
    if is_valid_set(candidate_17):
        best_result = candidate_17
        best_size = 17
    
    # Try to find 18
    # Try adding one more number to candidate_17
    candidates_to_add = [x for x in range(2, 41) if x not in candidate_17]
    for c in candidates_to_add:
        new_set = candidate_17 + [c]
        if is_valid_set(new_set):
            best_result = new_set
            best_size = 18
            break
    
    if best_size >= 18:
        return best_result

    # If 18 not found, try randomized search for 17/18
    # Use a faster validity check for local search
    # Maintain set of products
    
    def try_add(S, P, x):
        """Try to add x to S (list) with product set P (set). Returns (new_S, new_P) or None."""
        # Check if x * p in P for any p in P
        # If yes, collision.
        for p in P:
            if x * p in P:
                return None
        # Also check if x is in P? 
        # If x is in P, then {x} collides with some subset.
        # But S is valid, so x (if not in S) cannot be in P?
        # Wait, P contains products of subsets of S.
        # If x is in P, then x = prod(A) for some A subset S.
        # Then {x} collides with A.
        # So we must check if x in P.
        if x in P:
            return None
        
        # If no collision, add x
        new_P = set(P)
        for p in P:
            new_P.add(x * p)
        return S + [x], new_P

    # Randomized Local Search
    random.seed(42)
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        # Start with a random subset of primes and composites
        # Bias towards the known good structure
        # Remove 0, 1, or 2 primes from the full prime list
        primes_copy = primes[:]
        random.shuffle(primes_copy)
        num_primes_to_remove = random.randint(0, 2)
        selected_primes = primes_copy[num_primes_to_remove:]
        
        # Add composites greedily
        S = selected_primes[:]
        P = {1}
        for x in S:
            # S is just primes, so no collisions among themselves
            # But we need to build P
            new_P = set(P)
            for p in P:
                new_P.add(x * p)
            P = new_P
        
        # Try to add composites
        composites = [x for x in range(2, 41) if x not in S and not any(x % p == 0 for p in primes if p < x)] # This filter is wrong, just use all composites
        composites = [x for x in range(2, 41) if x not in S]
        random.shuffle(composites)
        
        for c in composites:
            if time.time() > deadline:
                break
            res = try_add(S, P, c)
            if res:
                S, P = res
        
        if len(S) > best_size:
            best_size = len(S)
            best_result = S[:]
            if best_size == 18:
                break

    return best_result

# Run once to verify
# print(construct())
