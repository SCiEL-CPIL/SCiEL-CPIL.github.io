import random
import math
import itertools

def construct():
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_pairwise_coprime(sums):
        # sums is a list of distinct values
        n = len(sums)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(sums[i], sums[j]) > 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if not sums:
            return 0
        if not is_pairwise_coprime(list(sums)):
            return 0
        return len(sums)

    # Precompute primes up to 256 (max sum 128+128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False

    def find_solution_prime():
        # Search for A, B such that all a+b are distinct primes
        # This guarantees pairwise coprime and max cardinality 9
        
        # Try small values first
        range_limit = 40
        for A in itertools.combinations(range(1, range_limit + 1), 3):
            # Find all b in [1, 128] such that a+b is prime for all a in A
            valid_bs = []
            for b in range(1, 129):
                ok = True
                for a in A:
                    s = a + b
                    if s > max_val or not is_prime[s]:
                        ok = False
                        break
                if ok:
                    valid_bs.append(b)
            
            if len(valid_bs) < 3:
                continue
            
            # Try combinations of 3 distinct b's from valid_bs
            # We need the 9 sums to be distinct
            for B in itertools.combinations(valid_bs, 3):
                sum_set = set()
                distinct = True
                for a in A:
                    for b in B:
                        s = a + b
                        if s in sum_set:
                            distinct = False
                            break
                        sum_set.add(s)
                    if not distinct:
                        break
                
                if distinct and len(sum_set) == 9:
                    # All sums are primes, so they are pairwise coprime
                    # Verify just in case
                    if is_pairwise_coprime(list(sum_set)):
                        return list(A), list(B)
        return None

    # Try the prime search
    result = find_solution_prime()
    if result:
        A, B = result
        # Verify
        if evaluate(A, B) == 9:
            return [sorted(A), sorted(B)]

    # Fallback to random search if prime search fails (unlikely)
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    s = evaluate(best_A, best_B)
    if s > best_score:
        best_score = s

    def random_set(size, low, high):
        return random.sample(range(low, high+1), size)

    max_iters = 5000
    current_A = random_set(3, 1, 100)
    current_B = random_set(3, 1, 100)
    current_score = evaluate(current_A, current_B)
    
    if current_score > best_score:
        best_score = current_score
        best_A = current_A
        best_B = current_B

    for _ in range(max_iters):
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            current_A[idx] = random.randint(1, 128)
            new_score = evaluate(current_A, current_B)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = current_A
                    best_B = current_B
            else:
                current_A[idx] = old_val
        else:
            idx = random.randint(0, 2)
            old_val = current_B[idx]
            current_B[idx] = random.randint(1, 128)
            new_score = evaluate(current_A, current_B)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = current_A
                    best_B = current_B
            else:
                current_B[idx] = old_val

        if _ % 1000 == 0:
            current_A = random_set(3, 1, 100)
            current_B = random_set(3, 1, 100)
            current_score = evaluate(current_A, current_B)
            if current_score > best_score:
                best_score = current_score
                best_A = current_A
                best_B = current_B

    return [sorted(best_A), sorted(best_B)]

# EVOLVE-BLOCK-START
# The code above is the logic. I will wrap it properly.
# EVOLVE-BLOCK-END
