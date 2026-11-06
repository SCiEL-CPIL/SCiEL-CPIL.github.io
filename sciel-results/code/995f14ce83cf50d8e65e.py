import itertools
import math

def construct():
    # Precompute primes up to 256 (128+128)
    max_sum = 256
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(max_sum)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
    
    # We search for A in [1, 50] to keep sums small and prime density high
    # But we should probably go a bit wider if needed. 50 seems safe.
    # To be safe and thorough, let's go up to 60.
    range_a = range(1, 61)
    
    best_score = 0
    best_A = [1, 5, 7]
    best_B = [6, 36, 66]
    
    # Check the known solution first
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Check pairwise coprime
        lst = list(sums)
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(lst[i], lst[j]) > 1:
                    return 0
        return len(sums)

    s = evaluate(best_A, best_B)
    if s > best_score:
        best_score = s
        # If we found 9, we can return immediately as it's the max possible
        if best_score == 9:
            return [sorted(best_A), sorted(best_B)]

    # Systematic search for 9 distinct primes
    # Iterate through combinations of A
    for A in itertools.combinations(range_a, 3):
        A = list(A)
        # Find all valid b's for this A
        valid_bs = []
        for b in range(1, 129):
            s1 = A[0] + b
            s2 = A[1] + b
            s3 = A[2] + b
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                valid_bs.append(b)
        
        if len(valid_bs) < 3:
            continue
            
        # We need to pick 3 distinct b's from valid_bs
        # such that the 9 sums are distinct
        # This is equivalent to: no difference b_j - b_k equals any difference A_i - A_m
        A_diffs = set()
        for i in range(3):
            for j in range(3):
                if i != j:
                    A_diffs.add(A[i] - A[j])
        
        found = False
        for B in itertools.combinations(valid_bs, 3):
            B = list(B)
            # Check if any difference in B is in A_diffs
            ok = True
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if B[i] - B[j] in A_diffs:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                # Verify distinctness and coprimality (primes are coprime)
                # Since all sums are prime and distinct, they are pairwise coprime
                # Just to be absolutely sure, we can run evaluate
                score = evaluate(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [sorted(best_A), sorted(best_B)]
                found = True
                # We can break here if we found a valid one, as we just want max cardinality
                # But maybe a different A has a "nicer" B? Doesn't matter for score.
                break

    # If we didn't find 9, we might have found 8 or less in the loop above
    # But the loop above only updates if score > best_score.
    # If best_score is still 0 or low, we might need a fallback.
    # However, the known solution gives 9, so we should return early.
    
    return [sorted(best_A), sorted(best_B)]
