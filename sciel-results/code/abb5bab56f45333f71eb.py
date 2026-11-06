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

    # Direct construction: use small odd A, small even B so all sums are odd.
    # Small odd numbers are more likely to be prime.
    # A = {1, 3, 5}, B = {2, 4, 6} -> sums: 3,5,7,5,7,9,7,9,11 -> distinct {3,5,7,9,11} -> 9 not prime, fail
    # Try A = {1, 3, 7}, B = {2, 4, 8} -> 3,5,9,5,7,11,9,11,15 -> distinct {3,5,7,9,11,15} -> 9,15 not coprime with others, fail
    # 
    # Strategy: search small ranges for 9 distinct primes.
    # A from small odds, B from small evens (or vice versa) to keep sums small and odd.
    
    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0
    
    def score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    best_score = score(best_A, best_B)
    
    # Search: A from odds, B from evens (or mixed) in small range
    # To get 9 distinct sums, we need the 3x3 grid of a+b to have all distinct values.
    # This happens when A-B has distinct differences and B-A has distinct differences.
    # Specifically, a1+b1 = a2+b2 iff a1-a2 = b2-b1.
    # So we need all pairwise differences in A to be distinct from all pairwise differences in B (in magnitude).
    
    # Let's search systematically.
    # A: 3 distinct from [1..N]
    # B: 3 distinct from [1..N]
    
    # First, try to find a solution with all 9 sums prime.
    # This is the strongest guarantee of pairwise coprimality.
    
    # For efficiency, precompute which (a,b) pairs give prime sums.
    # Then search for 3x3 subgrid where all 9 are prime.
    
    # But first, let's just do a bounded search with early termination.
    
    # Search space: A and B from [1..50] first, expand if needed.
    # Number of 3-subsets from 50 is C(50,3) = 19600, squared is ~384M - too large.
    # Need smarter approach.
    
    # Key insight: for 9 distinct sums, we need the difference sets to not overlap.
    # Let A = {a1, a2, a3} with a1 < a2 < a3
    # Let B = {b1, b2, b3} with b1 < b2 < b3
    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # These are distinct iff (a2-a1) != (b2-b1), (a3-a1) != (b3-b1), (a3-a2) != (b3-b2)
    # and no cross equalities: a2-a1 != b3-b2, a3-a1 != b3-b2, a3-a2 != b2-b1, etc.
    # More precisely, all 9 are distinct iff the set of differences {a2-a1, a3-a2, a3-a1} 
    # is disjoint from {b2-b1, b3-b2, b3-b1}.
    
    # So let's pick A and B with disjoint difference sets, then check if sums are pairwise coprime.
    
    # Common difference set: arithmetic progression has one difference.
    # A = {1, 1+d1, 1+2*d1}, differences: {d1, 2*d1}
    # B = {1, 1+e1, 1+2*e1}, differences: {e1, 2*e1}
    # Disjoint if d1, 2*d1 not in {e1, 2*e1}
    
    # Let's try A = {1, 1+p, 1+2p} for prime p, B = {1, 1+q, 1+2q} for prime q
    # with p, 2p not equal to q, 2q.
    
    # But sums will be: 2, 2+p+q, 2+2q, 2+p, 2+2p+q, 2+3q, 2+2p, 2+3p+q, 2+4q
    # Wait, that's not right. Let me recompute.
    # A = {1, 1+p, 1+2p}, B = {1, 1+q, 1+2q}
    # Sums:
    # 1+1=2, 1+(1+q)=2+q, 1+(1+2q)=2+2q
    # (1+p)+1=2+p, (1+p)+(1+q)=2+p+q, (1+p)+(1+2q)=2+p+2q
    # (1+2p)+1=2+2p, (1+2p)+(1+q)=2+2p+q, (1+2p)+(1+2q)=2+2p+2q
    
    # 2 is even. If we want all odd (for prime chance), make A all odd, B all even.
    
    # A = {a, a+p, a+2p} (odd, so a odd, p even... wait, p even makes a+p odd, a+2p odd)
    # Actually if a is odd and p is even, then a, a+p, a+2p are all odd.
    # B = {b, b+q, b+2q} (even, so b even, q even)
    
    # Let me just do a direct search with pruning.
    
    # Search A from small odds, B from small evens.
    # Prune: check distinctness early.
    
    odds = list(range(1, 81, 2))  # 1,3,5,...,79
    evens = list(range(2, 81, 2))  # 2,4,6,...,80
    
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
            
            # Compute sums and check if all are prime (strongest guarantee)
            all_prime = True
            for a in A:
                for b in B:
                    if not is_prime[a + b]:
                        all_prime = False
                        break
                if not all_prime:
                    break
            
            if all_prime:
                best_A = A
                best_B = B
                best_score = 9
                found = True
                break
            
            # If not all prime, still check pairwise coprime
            sums = []
            for a in A:
                for b in B:
                    sums.append(a + b)
            
            distinct_sums = list(set(sums))
            if len(distinct_sums) != 9:
                continue  # Shouldn't happen if diff sets disjoint, but just in case
            
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
    
    # If not found in small range, try wider range with all combinations (not just odd/even)
    # But this is expensive. Let's try a different direct construction.
    
    # Direct construction using known prime tuples.
    # We want 9 distinct primes. Let's try to construct A and B such that sums are primes.
    
    # Try specific candidates:
    candidates_A = [
        [1, 3, 7],
        [1, 3, 11],
        [1, 3, 13],
        [1, 5, 7],
        [1, 5, 13],
        [1, 7, 13],
        [1, 7, 19],
        [1, 11, 13],
        [3, 5, 7],
        [3, 5, 11],
    ]
    candidates_B = [
        [2, 4, 8],
        [2, 4, 10],
        [2, 4, 12],
        [2, 4, 14],
        [2, 6, 8],
        [2, 6, 10],
        [2, 6, 14],
        [2, 8, 10],
        [4, 6, 8],
        [4, 6, 10],
    ]
    
    for A in candidates_A:
        if time.time() - t_start > DEADLINE:
            break
        for B in candidates_B:
            if time.time() - t_start > DEADLINE:
                break
            sums = [a + b for a in A for b in B]
            distinct_sums = list(set(sums))
            if len(distinct_sums) == 9 and _pairwise_coprime(distinct_sums):
                return [A, B]
    
    # Fallback: return best found so far
    return [best_A, best_B]
