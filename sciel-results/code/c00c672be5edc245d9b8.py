import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for overhead/return
    
    # Seed solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    def get_cardinality(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        return len(sums)
        
    def is_pairwise_coprime(nums):
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(nums[i], nums[j]) > 1:
                    return False
        return True

    # Evaluate seed
    seed_sums = set(a + b for a in best_A for b in best_B)
    if len(seed_sums) == 9 and is_pairwise_coprime(list(seed_sums)):
        return best_A, best_B # Actually returns lists, so [best_A, best_B]
        
    best_card = get_cardinality(best_A, best_B)
    # Check if seed is valid pairwise coprime for its cardinality? 
    # The task requires "All DISTINCT values in A+B must be pairwise coprime".
    # So if seed has 5 sums and they are pairwise coprime, it's a valid candidate with score 5.
    if is_pairwise_coprime(list(seed_sums)):
        pass # best_card is valid
    else:
        # If seed is invalid, best_card should be 0 or we need to find a valid one.
        # But seed is known to be valid (primes).
        pass

    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    # Random Search
    while time.time() < deadline:
        # Try both parity configurations
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
            
        A.sort()
        B.sort()
        
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
                
        card = len(sums)
        if card < best_card:
            continue
            
        # If card is 9, we must check coprimality.
        # If card < 9, we can also check coprimality to see if it's a better valid set.
        # But maximizing cardinality is the primary goal.
        # If we find a valid set with card > best_card, we update.
        
        if card > best_card:
            if is_pairwise_coprime(list(sums)):
                best_card = card
                best_A = A
                best_B = B
                if best_card == 9:
                    return [best_A, best_B]
        elif card == best_card:
            # Optional: try to find a better set with same cardinality?
            # No, score is just cardinality. 
            # But if current best is invalid? 
            # We only update best if it's valid.
            # So if best_card is 5 and valid, and we find another 5, we don't care.
            # If best_card is 0 (invalid seed?), we need to find any valid.
            # Seed is valid.
            pass

    return [best_A, best_B]
