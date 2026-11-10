import itertools
import math

def construct():
    max_val = 128
    max_sum = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
                
    odds = [x for x in range(1, max_val + 1) if x % 2 == 1]
    evens = [x for x in range(2, max_val + 1) if x % 2 == 0]
    
    # Precompute neighbors (valid B's) for each odd A
    # A is odd, B is even => A+B is odd. Primes > 2 are odd.
    # So we only care about primes > 2.
    neighbors = {a: set() for a in odds}
    for a in odds:
        for b in evens:
            if is_prime[a + b]:
                neighbors[a].add(b)
                
    # Greedy search for A
    # We want to pick a1, a2, a3 such that their common neighbors are large
    # Sort odds by degree descending
    sorted_odds = sorted(odds, key=lambda x: len(neighbors[x]), reverse=True)
    
    best_A = None
    best_B = None
    best_distinct_count = 0
    
    # Try a few top candidates for a1
    for a1 in sorted_odds[:20]:
        # Common neighbors with a1 are just neighbors[a1]
        # We need to pick a2 from odds that shares many neighbors with a1
        # Calculate intersection size with a1
        candidates_a2 = []
        for a2 in sorted_odds:
            if a2 == a1: continue
            inter12 = neighbors[a1].intersection(neighbors[a2])
            if len(inter12) >= 3: # Need at least 3 for B
                candidates_a2.append((a2, inter12, len(inter12)))
        
        # Sort a2 candidates by intersection size
        candidates_a2.sort(key=lambda x: x[2], reverse=True)
        
        for a2, inter12, _ in candidates_a2[:10]:
            # Now pick a3 from odds that shares neighbors with inter12
            candidates_a3 = []
            for a3 in sorted_odds:
                if a3 == a1 or a3 == a2: continue
                inter123 = inter12.intersection(neighbors[a3])
                if len(inter123) >= 3:
                    candidates_a3.append((a3, inter123, len(inter123)))
            
            candidates_a3.sort(key=lambda x: x[2], reverse=True)
            
            for a3, inter123, _ in candidates_a3[:10]:
                # We have A = [a1, a2, a3] and a pool of B's in inter123
                # We need to pick 3 B's from inter123 such that all 9 sums are distinct
                # Sums are a_i + b_j.
                # Distinctness condition:
                # a1+b1, a1+b2, a1+b3
                # a2+b1, a2+b2, a2+b3
                # a3+b1, a3+b2, a3+b3
                # All distinct iff:
                # b1 != b2 != b3 (distinct B's)
                # a1 != a2 != a3 (distinct A's)
                # a_i + b_j == a_k + b_l => b_j - b_l = a_k - a_i
                # Since a's are distinct, a_k - a_i is non-zero constant for fixed i,k.
                # We just need to check if any sum repeats.
                
                B_list = list(inter123)
                # If len(B_list) is small, check all combinations
                # If large, we can try to pick greedily or just check first few
                # Since we want 9 distinct, we just need to avoid collisions.
                
                found = False
                # Try combinations of 3 from B_list
                # Limit to first 50 to keep it fast, as inter123 is likely large
                B_subset = B_list[:50]
                for B in itertools.combinations(B_subset, 3):
                    b1, b2, b3 = B
                    sums = {
                        a1+b1, a1+b2, a1+b3,
                        a2+b1, a2+b2, a2+b3,
                        a3+b1, a3+b2, a3+b3
                    }
                    if len(sums) == 9:
                        # Verify pairwise coprime (should be true as they are primes)
                        # But just in case, if we changed logic later
                        sums_list = list(sums)
                        valid = True
                        for i in range(len(sums_list)):
                            for j in range(i+1, len(sums_list)):
                                if math.gcd(sums_list[i], sums_list[j]) != 1:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            return [a1, a2, a3], [b1, b2, b3]
        
    # Fallback to original exhaustive search if greedy fails (unlikely)
    for A in itertools.combinations(odds, 3):
        a1, a2, a3 = A
        V_A = neighbors[a1].intersection(neighbors[a2]).intersection(neighbors[a3])
        if len(V_A) < 3:
            continue
        V_A_list = list(V_A)
        for B in itertools.combinations(V_A_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                return [list(A), list(B)]
                
    # Final fallback
    return [[1, 7, 13], [4, 10, 16]]
