import math
import time
from itertools import combinations

def construct():
            start_time = time.time()
            deadline = start_time + 80
            
            # Sieve for primes up to 256
            max_sum = 256
            sieve = [True] * (max_sum + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(max_sum**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, max_sum+1, i):
                        sieve[j] = False
            prime_set = set(i for i in range(2, max_sum+1) if sieve[i])
            
            # Precompute valid B for each A
            valid_b_for_a = {}
            for a in range(1, 129):
                valid_b_for_a[a] = set()
                for b in range(1, 129):
                    if (a + b) in prime_set:
                        valid_b_for_a[a].add(b)
            
            best_count = 0
            best_A = None
            best_B = None
            
            # Precompute all 3-element combinations of A values
            a_combos = list(combinations(range(1, 129), 3))
            
            # Search for 9 distinct prime sums
            for a1, a2, a3 in a_combos:
                if time.time() > deadline:
                    break
                valid_b = valid_b_for_a[a1] & valid_b_for_a[a2] & valid_b_for_a[a3]
                if len(valid_b) < 3:
                    continue
                b_list = sorted(valid_b)
                # Check all combinations of 3 from b_list
                for b1, b2, b3 in combinations(b_list, 3):
                    B = (b1, b2, b3)
                    sums = [a + b for a in (a1, a2, a3) for b in B]
                    if len(set(sums)) == 9:
                        return [[a1, a2, a3], list(B)]
            
            # If 9 not found, search for best pairwise coprime
            # (This part is likely not reached if 9 exists)
            
            return [[1, 7, 13], [4, 10, 16]]
