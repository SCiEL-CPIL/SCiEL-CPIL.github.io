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
    best_score = evaluate(best_A, best_B)  # 5

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

    # Precompute for each b, the set of a's such that a+b is prime
    prime_a_for_b = [set() for _ in range(129)]
    for b in range(1, 129):
        for a in range(1, 129):
            s = a + b
            if 2 <= s <= 256 and s in prime_set:
                prime_a_for_b[b].add(a)

    found = False

    # Strategy: iterate over b1 < b2, find V12 = a's where a+b1, a+b2 are prime
    # Then for each b3, find a's in V12 where a+b3 is also prime
    # Pick 3 such a's and check if all 9 sums are distinct primes
    
    # Optimization: only consider b values that have at least 3 a's making prime sums
    valid_b = [b for b in range(1, 129) if len(prime_a_for_b[b]) >= 3]
    
    for b1 in valid_b:
        if time.time() > deadline - 15:
            break
        for b2 in valid_b:
            if b2 <= b1:
                continue
            if time.time() > deadline - 10:
                break
            
            V12 = prime_a_for_b[b1] & prime_a_for_b[b2]
            if len(V12) < 3:
                continue
                
            for b3 in valid_b:
                if b3 == b1 or b3 == b2:
                    continue
                if time.time() > deadline - 5:
                    break
                
                C_b3 = V12 & prime_a_for_b[b3]
                if len(C_b3) < 3:
                    continue
                
                # Try to find 3 a's from C_b3 such that all 9 sums are distinct primes
                c_list = sorted(C_b3)
                
                if len(c_list) <= 12:
                    comb_iter = list(itertools.combinations(c_list, 3))
                else:
                    comb_iter = [tuple(sorted(random.sample(c_list, 3))) for _ in range(15)]
                
                for a1, a2, a3 in comb_iter:
                    A = [a1, a2, a3]
                    B = [b1, b2, b3]
                    
                    sums = set()
                    all_prime = True
                    for a in A:
                        for b in B:
                            s = a + b
                            sums.add(s)
                            if s not in prime_set:
                                all_prime = False
                                break
                        if not all_prime:
                            break
                    
                    if all_prime and len(sums) == 9:
                        best_A = A
                        best_B = B
                        best_score = 9
                        found = True
                        break
                        
                if found:
                    break
                    
            if found:
                break
        if found:
            break

    if best_score == 9:
        return [list(best_A), list(best_B)]

    # Fallback: local search with prime-focused initialization
    def random_prime_set():
        # Pick 3 numbers that tend to produce prime sums
        candidates = random.sample(range(1, 129), 20)
        # Score by how many prime sums they produce
        scored = []
        for x in candidates:
            cnt = len(prime_a_for_b[x]) if x <= 128 else 0
            scored.append((cnt, x))
        scored.sort(reverse=True)
        return sorted([x for _, x in scored[:3]])

    def local_search():
        A = random_prime_set()
        B = random_prime_set()
        curr_score = evaluate(A, B)

        for step in range(3000):
            if time.time() > deadline - 3:
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

    for _ in range(15):
        if time.time() > deadline - 3:
            break
        A_ls, B_ls, score_ls = local_search()
        if score_ls > best_score:
            best_score = score_ls
            best_A = A_ls
            best_B = B_ls
            if best_score == 9:
                break

    return [list(best_A), list(best_B)]
