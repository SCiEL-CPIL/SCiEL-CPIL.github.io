import math
import random
import time

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # 1. Generate all pairwise coprime triples in [1, 128]
    # Optimization: Precompute primes or just use gcd
    # A number in [1, 128] is either prime or composite.
    # If a number is composite, it has prime factors.
    # Two numbers are coprime if they share no prime factors.
    
    # Let's just generate the list first.
    candidates = []
    for a in range(1, 129):
        for b in range(a+1, 129):
            if math.gcd(a, b) != 1: continue
            for c in range(b+1, 129):
                if math.gcd(a, c) == 1 and math.gcd(b, c) == 1:
                    candidates.append([a, b, c])
                    
    # Sort candidates? Maybe not necessary.
    # Randomize to find good solutions quickly?
    # Or systematic?
    # Let's try systematic with early exit if max size 9 found.
    
    best_A = None
    best_B = None
    best_size = 0
    
    n = len(candidates)
    # If n is large, n^2 is huge.
    # Let's estimate n.
    # Probability of 3 random numbers being pairwise coprime is roughly (6/pi^2)^3 ~ 0.16?
    # 350,000 * 0.16 ~ 56,000 candidates.
    # 56,000^2 = 3.1 * 10^9. Too slow for Python.
    
    # We need a faster search.
    # Random search is better.
    # The knowledge says random search found 1.8 (size 9).
    # So random search is effective.
    
    random.shuffle(candidates)
    
    # Try random pairs from candidates
    # If we find size 9, we stop.
    
    # To be safe, we can also check specific structured pairs.
    # But random should work.
    
    # Let's do a few thousand random pairs first.
    # Then if not found, expand?
    
    # Actually, we can just iterate through random pairs until deadline.
    # Or iterate through all if n is small enough?
    # 56k is too many for full n^2.
    
    # Strategy:
    # 1. Random sampling of pairs from candidates.
    # 2. Keep track of best size.
    # 3. If best size == 9, return.
    
    # How many random pairs to try?
    # Let's try 10,000 pairs.
    # If not found, try more?
    # Or just keep going until deadline.
    
    # Let's use a counter and break if deadline exceeded.
    
    # To increase chance, we can bias towards numbers with fewer prime factors?
    # Primes are best.
    # Let's just use random.
    
    # Optimization: Precompute distinct sums and check coprimality efficiently.
    
    for _ in range(1000000): # Max iterations, but break on time
        if time.time() > deadline:
            break
            
        idx1 = random.randint(0, n-1)
        idx2 = random.randint(0, n-1)
        
        A = candidates[idx1]
        B = candidates[idx2]
        
        # Compute distinct sums
        sums = set()
        for a in A:
            for b in B:
                sums.add(a+b)
                
        if len(sums) <= best_size:
            continue
            
        # Check pairwise coprimality of sums
        sum_list = list(sums)
        valid = True
        for i in range(len(sum_list)):
            for j in range(i+1, len(sum_list)):
                if math.gcd(sum_list[i], sum_list[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            best_size = len(sums)
            best_A = A
            best_B = B
            if best_size == 9:
                return [best_A, best_B]
                
    # If we didn't find 9, return best found
    if best_A is None:
        # Fallback to seed
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
