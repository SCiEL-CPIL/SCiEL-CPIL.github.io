import itertools
import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s

    def is_pairwise_coprime(nums):
        nums = list(nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) == 0:
            return -1
        if not is_pairwise_coprime(sums):
            return -1
        return len(sums)

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = evaluate(best_A, best_B) # 5

    # Strategy: Search for prime grid where all 9 sums are distinct primes
    # This guarantees pairwise coprimality and score 9
    
    # Generate primes up to 256 (max sum is 128+128=256)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(256)
    prime_set = set(primes)
    
    # For a 3x3 grid of primes p[i][j] = a[i] + b[j], we need:
    # p[i][j] - p[i][0] = b[j] - b[0] for all i
    # So the differences must be consistent across rows
    
    # Search over prime triples for row 0, then check if other rows can be formed
    # p[0][0], p[0][1], p[0][2] are primes
    # b[1] - b[0] = p[0][1] - p[0][0]
    # b[2] - b[0] = p[0][2] - p[0][0]
    # Then a[i] = p[i][0] - b[0] for i=0,1,2
    # And we need p[i][j] = a[i] + b[j] to be prime for all i,j
    
    found = False
    for p00 in primes:
        if time.time() > deadline - 5:
            break
        if found:
            break
        for p01 in primes:
            if p01 <= p00:
                continue
            if time.time() > deadline - 5:
                break
            if found:
                break
            for p02 in primes:
                if p02 <= p01:
                    continue
                if time.time() > deadline - 5:
                    break
                if found:
                    break
                    
                # Differences for B
                db1 = p01 - p00
                db2 = p02 - p00
                
                # Try different values of b0
                # a[0] = p00 - b0, must be in [1, 128]
                # b[1] = b0 + db1, must be in [1, 128]
                # b[2] = b0 + db2, must be in [1, 128]
                
                # b0 must satisfy:
                # 1 <= p00 - b0 <= 128  =>  p00 - 128 <= b0 <= p00 - 1
                # 1 <= b0 + db1 <= 128  =>  1 - db1 <= b0 <= 128 - db1
                # 1 <= b0 + db2 <= 128  =>  1 - db2 <= b0 <= 128 - db2
                
                b0_min = max(1, p00 - 128, 1 - db1, 1 - db2)
                b0_max = min(p00 - 1, 128 - db1, 128 - db2)
                
                if b0_min > b0_max:
                    continue
                
                for b0 in range(b0_min, b0_max + 1):
                    if time.time() > deadline - 5:
                        break
                    if found:
                        break
                    
                    b1 = b0 + db1
                    b2 = b0 + db2
                    a0 = p00 - b0
                    
                    # Check all values are in [1, 128] and distinct
                    B_vals = [b0, b1, b2]
                    if any(x < 1 or x > 128 for x in B_vals):
                        continue
                    if len(set(B_vals)) != 3:
                        continue
                    
                    # Now search for a1, a2 such that all 9 sums are distinct primes
                    # a1 = p10 - b0 for some prime p10
                    # a2 = p20 - b0 for some prime p20
                    # Need p10, p20 to be primes
                    # And all 9 sums must be distinct primes
                    
                    # p10 must be prime, and a1 = p10 - b0 in [1, 128]
                    # So p10 in [b0+1, b0+128] and prime
                    # Similarly for p20
                    
                    # For efficiency, iterate over possible p10, p20
                    # But we also need all other sums to be prime and distinct
                    
                    # Let's try a smaller search: iterate over a1, a2 in [1,128]
                    # and check if all 9 sums are distinct primes
                    
                    # But that's 128*128 = 16384 per (p00,p01,p02,b0) which is too slow
                    # Instead, note that a1 = p10 - b0 where p10 is prime
                    # So we can iterate over primes p10 in [b0+1, b0+128]
                    
                    p10_candidates = [p for p in primes if b0 + 1 <= p <= b0 + 128]
                    p20_candidates = [p for p in primes if b0 + 1 <= p <= b0 + 128]
                    
                    for p10 in p10_candidates:
                        if time.time() > deadline - 5:
                            break
                        if found:
                            break
                        
                        a1 = p10 - b0
                        if a1 == a0:
                            continue
                        
                        for p20 in p20_candidates:
                            if time.time() > deadline - 5:
                                break
                            if found:
                                break
                            
                            a2 = p20 - b0
                            if a2 == a0 or a2 == a1:
                                continue
                            
                            A_vals = [a0, a1, a2]
                            if any(x < 1 or x > 128 for x in A_vals):
                                continue
                            if len(set(A_vals)) != 3:
                                continue
                            
                            # Check all 9 sums are distinct primes
                            sums = set()
                            valid = True
                            for i, a in enumerate(A_vals):
                                for j, b in enumerate(B_vals):
                                    s = a + b
                                    if s not in prime_set:
                                        valid = False
                                        break
                                    if s in sums:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            
                            if valid and len(sums) == 9:
                                best_A = A_vals
                                best_B = B_vals
                                best_score = 9
                                found = True
                                break

    # If we found score 9, return it
    if best_score == 9:
        return [list(best_A), list(best_B)]
    
    # Fallback: try local search for score 8 or better
    def random_set():
        return sorted(random.sample(range(1, 129), 3))
    
    def local_search():
        A = random_set()
        B = random_set()
        curr_score = evaluate(A, B)
        
        for step in range(5000):
            if time.time() > deadline - 2:
                return A, B, curr_score
                
            if curr_score == 9:
                return A, B, curr_score
                
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old = A[idx]
                A[idx] = random.randint(1, 128)
                while len(set(A)) < 3:
                    A[idx] = random.randint(1, 128)
                A.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    A[idx] = old
                    A.sort()
                else:
                    curr_score = new_score
            else:
                idx = random.randint(0, 2)
                old = B[idx]
                B[idx] = random.randint(1, 128)
                while len(set(B)) < 3:
                    B[idx] = random.randint(1, 128)
                B.sort()
                new_score = evaluate(A, B)
                if new_score < curr_score:
                    B[idx] = old
                    B.sort()
                else:
                    curr_score = new_score
                    
        return A, B, curr_score

    for _ in range(20):
        if time.time() > deadline - 2:
            break
        A_ls, B_ls, score_ls = local_search()
        if score_ls > best_score:
            best_score = score_ls
            best_A = A_ls
            best_B = B_ls
            if best_score == 9:
                break

    return [list(best_A), list(best_B)]
