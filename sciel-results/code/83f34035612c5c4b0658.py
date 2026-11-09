"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = sorted(set(vals))
    n = len(distinct)
    if n <= 1:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _sumset(A, B):
    """Return the set of all sums a+b for a in A, b in B."""
    return set(a + b for a in A for b in B)

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct ints in [1,128], all distinct sums pairwise coprime."""
    if len(A) != 3 or len(set(A)) != 3:
        return False
    if len(B) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    sums = _sumset(A, B)
    return _pairwise_coprime(list(sums))

def _score(A, B):
    """Return the number of distinct sums if valid, else 0."""
    if not _is_valid(A, B):
        return 0
    return len(_sumset(A, B))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Best known solution from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # Generate primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(256)
    primes_set = set(primes)
    small_primes = [p for p in primes if p <= 128]
    
    random.seed(42)
    
    def random_set(size, low, high):
        """Generate a random set of 'size' distinct integers in [low, high]."""
        return random.sample(range(low, high + 1), size)
    
    # Phase 1: Prime grid search - enforce all 9 sums are distinct primes
    # This guarantees pairwise coprimality and maximizes cardinality to 9
    for b0 in range(1, 50):
        if time.time() > deadline:
            break
        for b1 in range(b0 + 1, 50):
            if time.time() > deadline:
                break
            for b2 in range(b1 + 1, 50):
                if time.time() > deadline:
                    break
                B = [b0, b1, b2]
                
                valid_A_candidates = []
                for a in range(1, 129):
                    s0 = a + b0
                    s1 = a + b1
                    s2 = a + b2
                    if s0 <= 256 and s1 <= 256 and s2 <= 256:
                        if s0 in primes_set and s1 in primes_set and s2 in primes_set:
                            valid_A_candidates.append((a, s0, s1, s2))
                
                n_cand = len(valid_A_candidates)
                for i in range(n_cand):
                    a0, p00, p01, p02 = valid_A_candidates[i]
                    for j in range(i + 1, n_cand):
                        a1, p10, p11, p12 = valid_A_candidates[j]
                        if a1 == a0:
                            continue
                        for k in range(j + 1, n_cand):
                            a2, p20, p21, p22 = valid_A_candidates[k]
                            if a2 == a0 or a2 == a1:
                                continue
                            
                            all_sums = [p00, p01, p02, p10, p11, p12, p20, p21, p22]
                            if len(set(all_sums)) == 9:
                                A = [a0, a1, a2]
                                s = 9
                                if s > best_score:
                                    best_score = s
                                    best_A = A[:]
                                    best_B = B[:]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 2: Random search with local improvement
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        strategy = iteration % 4
        if strategy == 0:
            A = random_set(3, 1, 25)
            B = random_set(3, 1, 25)
        elif strategy == 1:
            A = [1] + random_set(2, 2, 40)
            random.shuffle(A)
            B = random_set(3, 1, 40)
        elif strategy == 2:
            A = random_set(3, 1, 20)
            B = random_set(3, 1, 50)
        else:
            A = random.sample(small_primes, 3)
            B = random.sample(small_primes, 3)
        
        s = _score(A, B)
        if s > best_score:
            best_score = s
            best_A = A[:]
            best_B = B[:]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 3: Local search from best
    def local_search(A, B, max_iter=30000):
        best_A_local = A[:]
        best_B_local = B[:]
        best_s = _score(best_A_local, best_B_local)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for idx in range(3):
                old_val = best_A_local[idx]
                new_val = random.randint(1, 128)
                if new_val not in best_A_local:
                    best_A_local[idx] = new_val
                    s = _score(best_A_local, best_B_local)
                    if s > best_s:
                        best_s = s
                        improved = True
                    else:
                        best_A_local[idx] = old_val
            
            for idx in range(3):
                old_val = best_B_local[idx]
                new_val = random.randint(1, 128)
                if new_val not in best_B_local:
                    best_B_local[idx] = new_val
                    s = _score(best_A_local, best_B_local)
                    if s > best_s:
                        best_s = s
                        improved = True
                    else:
                        best_B_local[idx] = old_val
            
            if not improved:
                break
        
        return best_A_local, best_B_local, best_s
    
    A_ls, B_ls, s_ls = local_search(best_A, best_B)
    if s_ls > best_score:
        best_score = s_ls
        best_A = A_ls[:]
        best_B = B_ls[:]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 4: More random restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        
        A = random_set(3, 1, 60)
        B = random_set(3, 1, 60)
        A_ls, B_ls, s_ls = local_search(A, B, max_iter=10000)
        if s_ls > best_score:
            best_score = s_ls
            best_A = A_ls[:]
            best_B = B_ls[:]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
