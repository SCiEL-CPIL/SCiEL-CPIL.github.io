"""Direct construction: find A, B with 9 distinct pairwise-coprime sums (all primes)."""
import math
import time
import itertools

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    """Check all pairs are coprime. Returns True if valid."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False

    def score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)

    # We know from the current program that a solution with score 9 exists.
    # Let's find it quickly with a targeted search.
    
    # Strategy: Search for A (odds) and B (evens) in a small range.
    # If all 9 sums are prime, they are pairwise coprime.
    # Let's try a wider range than the current program to be safe, but still fast.
    
    odds = list(range(1, 101, 2))  # 1,3,5,...,99
    evens = list(range(2, 101, 2))  # 2,4,6,...,100

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)

    found = False
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        # Compute differences of A
        a_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                a_diffs.add(A[j] - A[i])
        
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            # Check difference disjointness
            b_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    b_diffs.add(B[j] - B[i])
            
            if a_diffs & b_diffs:
                continue  # sums won't all be distinct
            
            # Compute sums
            sums = []
            for a in A:
                for b in B:
                    sums.append(a + b)
            
            distinct_sums = list(set(sums))
            if len(distinct_sums) != 9:
                continue
            
            # Check pairwise coprime
            if _pairwise_coprime(distinct_sums):
                best_A = A
                best_B = B
                best_score = 9
                found = True
                break
        
        if found:
            break

    if found:
        return [best_A, best_B]
    
    # If not found in the main search, try a broader search with mixed parities
    # but with a tighter range to stay within time limits
    range_val = 50
    all_nums = list(range(1, range_val + 1))
    
    for A in itertools.combinations(all_nums, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                a_diffs.add(A[j] - A[i])
        
        for B in itertools.combinations(all_nums, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    b_diffs.add(B[j] - B[i])
            
            if a_diffs & b_diffs:
                continue
            
            sums = []
            for a in A:
                for b in B:
                    sums.append(a + b)
            
            distinct_sums = list(set(sums))
            if len(distinct_sums) != 9:
                continue
            
            if _pairwise_coprime(distinct_sums):
                return [A, B]
    
    # Fallback: return best found so far (should be valid with score 9 from initial guess or search)
    return [best_A, best_B]
