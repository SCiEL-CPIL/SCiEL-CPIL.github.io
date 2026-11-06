import math
import time
import itertools

def _is_prime_sieve(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return is_prime

def _pairwise_coprime(vals):
    """Check if all pairs in vals are coprime."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """
    Return two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B.
    """
    t_start = time.time()
    DEADLINE = 85  # Safety margin before 90s kill
    
    # Max sum is 128 + 128 = 256
    MAX_SUM = 256
    is_prime = _is_prime_sieve(MAX_SUM)
    
    # Strategy:
    # 1. We want 9 distinct sums. This requires disjoint difference sets.
    # 2. We want pairwise coprime sums. Making them all prime is a strong sufficient condition.
    # 3. To make sums odd (and thus potentially prime > 2), we can pick A as odds and B as evens.
    
    # Candidate elements
    # A: distinct odds in [1, 128]
    # B: distinct evens in [1, 128]
    
    # We will search in increasing magnitude to find a small, simple solution first.
    # This is likely to be fast.
    
    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_score = 0
    
    def get_score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Initial fallback score
    best_score = get_score(best_A, best_B)
    
    # Search for a solution with 9 distinct prime sums
    # Iterate through combinations of A and B
    # To ensure speed, we can prune based on time.
    
    # We can optimize by checking difference sets first.
    # For A = {a1, a2, a3}, diffs are {a2-a1, a3-a2, a3-a1}
    # For B = {b1, b2, b3}, diffs are {b2-b1, b3-b2, b3-b1}
    # If diff sets intersect, sums are not all distinct.
    
    found_max = False
    
    # Limit search space to smaller numbers first for speed
    # A solution is expected to exist with small numbers.
    # Let's search A from first 20 odds, B from first 20 evens.
    # C(20,3) * C(20,3) = 1140 * 1140 ~ 1.3M checks. Very fast in Python if optimized.
    
    search_range = 20
    odds_search = odds[:search_range]
    evens_search = evens[:search_range]
    
    # Precompute difference sets for all combinations to speed up?
    # Or just compute on the fly. On the fly is fine for 1.3M.
    
    for A in itertools.combinations(odds_search, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = set()
        a_diffs.add(A[1] - A[0])
        a_diffs.add(A[2] - A[1])
        a_diffs.add(A[2] - A[0])
        
        for B in itertools.combinations(evens_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = set()
            b_diffs.add(B[1] - B[0])
            b_diffs.add(B[2] - B[1])
            b_diffs.add(B[2] - B[0])
            
            # Check disjointness
            if a_diffs & b_diffs:
                continue
            
            # Check if all 9 sums are prime
            # Since A are odds and B are evens, sums are odd.
            # If all are prime, they are distinct (checked by diff sets) and pairwise coprime.
            all_prime = True
            for a in A:
                for b in B:
                    s = a + b
                    if not is_prime[s]:
                        all_prime = False
                        break
                if not all_prime:
                    break
            
            if all_prime:
                return [A, B]
    
    # If not found in small range, expand search or fall back to best pairwise coprime set
    # Given the density of primes, it's highly unlikely we miss it in the first 20.
    # But let's do a broader check for pairwise coprime sums if we didn't find 9 primes.
    
    # We can also try mixed parity if needed, but odd/even is most efficient for primes.
    
    # Fallback: Just return the best found so far (which might be < 9)
    # However, the problem asks to MAXIMIZE cardinality.
    # The search above is exhaustive for the first 20 elements.
    # If it fails, we should try a wider range or a different strategy.
    
    # Let's try a slightly wider range just in case, with time check.
    if time.time() - t_start < DEADLINE - 10:
        search_range = 40
        odds_search = odds[:search_range]
        evens_search = evens[:search_range]
        
        for A in itertools.combinations(odds_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            A = list(A)
            a_diffs = set()
            a_diffs.add(A[1] - A[0])
            a_diffs.add(A[2] - A[1])
            a_diffs.add(A[2] - A[0])
            
            for B in itertools.combinations(evens_search, 3):
                if time.time() - t_start > DEADLINE:
                    break
                B = list(B)
                b_diffs = set()
                b_diffs.add(B[1] - B[0])
                b_diffs.add(B[2] - B[1])
                b_diffs.add(B[2] - B[0])
                
                if a_diffs & b_diffs:
                    continue
                
                # Check pairwise coprime directly, not just prime, to be more general
                # But prime is a subset. Let's stick to prime for speed first.
                all_prime = True
                for a in A:
                    for b in B:
                        s = a + b
                        if not is_prime[s]:
                            all_prime = False
                            break
                    if not all_prime:
                        break
                
                if all_prime:
                    return [A, B]
    
    # If we still haven't found 9, we should find the best pairwise coprime set we can.
    # But 9 is the max. The search for 9 primes is very likely to succeed.
    # If it fails, there might be a configuration with 9 distinct sums that are pairwise coprime but not all prime.
    # Example: 1, 5, 7, 9, 11, 13, 17, 19, 23. 9 is not prime, but gcd(9, others) might be 1.
    
    # Let's do a final check for pairwise coprime in a small range if prime search failed.
    if time.time() - t_start < DEADLINE - 5:
        search_range = 20
        odds_search = odds[:search_range]
        evens_search = evens[:search_range]
        
        for A in itertools.combinations(odds_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            A = list(A)
            a_diffs = set()
            a_diffs.add(A[1] - A[0])
            a_diffs.add(A[2] - A[1])
            a_diffs.add(A[2] - A[0])
            
            for B in itertools.combinations(evens_search, 3):
                if time.time() - t_start > DEADLINE:
                    break
                B = list(B)
                b_diffs = set()
                b_diffs.add(B[1] - B[0])
                b_diffs.add(B[2] - B[1])
                b_diffs.add(B[2] - B[0])
                
                if a_diffs & b_diffs:
                    continue
                
                sums = set()
                for a in A:
                    for b in B:
                        sums.add(a + b)
                
                if len(sums) == 9 and _pairwise_coprime(list(sums)):
                    return [A, B]
    
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {A}")
    print(f"B: {B}")
    sums = set(a + b for a in A for b in B)
    print(f"Distinct Sums: {sorted(sums)}")
    print(f"Count: {len(sums)}")
    print(f"Pairwise Coprime: {_pairwise_coprime(list(sums))}")
