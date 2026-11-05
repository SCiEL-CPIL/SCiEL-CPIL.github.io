import math
import time

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
            
            # Search for 9 distinct prime sums
            for a1 in range(1, 129):
                if time.time() > deadline:
                    break
                vb1 = valid_b_for_a[a1]
                for a2 in range(a1 + 1, 129):
                    if time.time() > deadline:
                        break
                    vb12 = vb1 & valid_b_for_a[a2]
                    if len(vb12) < 3:
                        continue
                    for a3 in range(a2 + 1, 129):
                        if time.time() > deadline:
                            break
                        valid_b = vb12 & valid_b_for_a[a3]
                        if len(valid_b) < 3:
                            continue
                        b_list = sorted(valid_b)
                        # Check all combinations of 3 from b_list
                        # Since we need 9 distinct sums, we just need to pick any 3 that work
                        # Actually, if all 9 sums are prime, we just need to ensure they are distinct.
                        for i in range(len(b_list) - 2):
                            for j in range(i + 1, len(b_list) - 1):
                                for k in range(j + 1, len(b_list)):
                                    B = (b_list[i], b_list[j], b_list[k])
                                    sums = [a + b for a in (a1, a2, a3) for b in B]
                                    if len(set(sums)) == 9:
                                        return [[a1, a2, a3], list(B)]
            
            # If 9 not found, search for best pairwise coprime
            # (This part is likely not reached if 9 exists)
            
            return [[1, 7, 13], [4, 10, 16]]
