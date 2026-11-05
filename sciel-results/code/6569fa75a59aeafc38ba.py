"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Precompute primes up to 256 (max sum = 128+128) using sieve
    max_sum = 256
    sieve = [True] * (max_sum + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, max_sum+1, i):
                sieve[j] = False
    prime_set = set(i for i in range(2, max_sum+1) if sieve[i])
    
    # We want 9 distinct pairwise coprime sums.
    # The most robust way is to have all 9 sums be distinct primes.
    # This requires that for all a in A and b in B, a+b is prime.
    # Since primes > 2 are odd, a+b must be odd.
    # This implies one set is all even and the other is all odd.
    
    # Strategy:
    # 1. Iterate through possible A sets (sorted)
    # 2. For each A, find the intersection of valid B's (b such that a+b is prime for all a in A)
    # 3. If intersection has >= 3 elements, pick 3 B's that yield distinct sums.
    #    Note: If A = {a1, a2, a3} and B = {b1, b2, b3}, sums are a_i + b_j.
    #    Distinctness condition: a_i - a_k != b_j - b_l for any i!=k, j!=l.
    
    def check_distinct_sums(A, B):
        sums = set()
        for a in A:
            for b in B:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Precompute valid_b_for_a for all a in [1, 128]
    # valid_b_for_a[a] is a set of b in [1, 128] such that a+b is prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = set()
        for b in range(1, 129):
            if (a + b) in prime_set:
                valid_b_for_a[a].add(b)
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Search for 9 distinct prime sums
    # To optimize, we can restrict A to odd or even numbers?
    # If A is all odd, B must be all even.
    # If A is all even, B must be all odd.
    # Let's try both cases.
    
    for A_parity in [0, 1]: # 0: Even A, 1: Odd A
        B_parity = 1 - A_parity
        
        # Generate candidate A values with correct parity
        A_candidates = list(range(1 + A_parity, 129, 2))
        
        # We need 3 distinct A's.
        # To keep the search space manageable, we can limit the range or use heuristics.
        # However, with valid_b_for_a precomputed, the inner loop is fast.
        
        # Let's iterate through combinations of A
        # We can prune: if intersection of valid_b for a1, a2 is small, skip.
        
        # To ensure we find a solution quickly, we might want to iterate A's in a specific order.
        # Small A's often have more valid B's?
        
        found = False
        for i1 in range(len(A_candidates)):
            if time.time() > deadline:
                break
            a1 = A_candidates[i1]
            vb1 = valid_b_for_a[a1]
            
            for i2 in range(i1 + 1, len(A_candidates)):
                if time.time() > deadline:
                    break
                a2 = A_candidates[i2]
                vb12 = vb1 & valid_b_for_a[a2]
                if len(vb12) < 3:
                    continue
                
                for i3 in range(i2 + 1, len(A_candidates)):
                    if time.time() > deadline:
                        break
                    a3 = A_candidates[i3]
                    valid_b = vb12 & valid_b_for_a[a3]
                    if len(valid_b) < 3:
                        continue
                    
                    # We have a set of valid B's. We need to pick 3 distinct ones.
                    # Since all a+b are prime, they are pairwise coprime.
                    # We just need distinct sums.
                    b_list = sorted(valid_b)
                    
                    # Try to find 3 B's that yield distinct sums.
                    # Since A has distinct elements, if we pick B's with distinct differences?
                    # Actually, just checking distinct sums is O(9).
                    # We can just pick the first 3 B's from the sorted list?
                    # Not necessarily. a1+b1 == a2+b2 is possible.
                    # Example: A={1, 3}, B={2, 4}. 1+4=5, 3+2=5.
                    # So we need to check.
                    
                    # Heuristic: Consecutive B's in the sorted list are unlikely to cause collisions
                    # unless the differences in A match differences in B.
                    # Let's try a few combinations of B's from the valid set.
                    
                    # We can iterate through combinations of 3 from b_list.
                    # But b_list can be large.
                    # Let's just try the first few triples from b_list.
                    # If b_list is sorted, b[i], b[i+1], b[i+2] is a good candidate.
                    
                    for idx in range(len(b_list) - 2):
                        B = (b_list[idx], b_list[idx+1], b_list[idx+2])
                        if check_distinct_sums((a1, a2, a3), B):
                            best_A = [a1, a2, a3]
                            best_B = list(B)
                            best_count = 9
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            return [best_A, best_B]
    
    # Fallback if no 9-prime solution found (should not happen given density of primes)
    # But let's have a robust fallback.
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
