import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit

    # Precompute primes up to 256 (max sum 128+128)
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(256**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Fallback solution (known to work or good heuristic)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5

    # We want to find A, B such that all 9 sums are distinct and pairwise coprime.
    # Distinct primes are a sufficient condition.
    # To ensure sums are odd primes, A and B must have opposite parity.
    
    # Case 1: A odd, B even
    # Case 2: A even, B odd
    
    # We'll iterate over A values. To speed up, we can restrict A to specific parities.
    # Let's try both cases.
    
    for parity in [0, 1]: # 0 for A even, 1 for A odd
        # Determine parity of A elements
        a_parity = parity
        # Determine parity of B elements
        b_parity = 1 - parity
        
        # Generate candidate A values with correct parity in [1, 128]
        # We only need to check a subset. Small values are more likely to yield prime sums?
        # Actually, larger values might avoid small prime factors?
        # Let's stick to 1..60 for A to keep search space manageable.
        
        a_candidates = list(range(1 + a_parity, 61, 2))
        
        for i in range(len(a_candidates)):
            if time.time() > deadline:
                return [best_A, best_B]
            a1 = a_candidates[i]
            for j in range(i+1, len(a_candidates)):
                a2 = a_candidates[j]
                for k in range(j+1, len(a_candidates)):
                    a3 = a_candidates[k]
                    A = (a1, a2, a3)
                    
                    # Find valid Bs
                    valid_bs = []
                    b_start = 1 + b_parity
                    for b in range(b_start, 129, 2):
                        if is_prime[a1+b] and is_prime[a2+b] and is_prime[a3+b]:
                            valid_bs.append(b)
                    
                    if len(valid_bs) < 3:
                        continue
                    
                    # Check combinations of valid Bs
                    # We want 9 distinct sums.
                    # This happens if no a_i + b_j == a_k + b_l for (i,j) != (k,l).
                    # i.e., a_i - a_k != b_l - b_j.
                    
                    for b1, b2, b3 in combinations(valid_bs, 3):
                        B = (b1, b2, b3)
                        sums = set()
                        for a in A:
                            for b in B:
                                sums.add(a+b)
                        
                        if len(sums) > best_count:
                            best_count = len(sums)
                            best_A = list(A)
                            best_B = list(B)
                            
                            if best_count == 9:
                                # Verify pairwise coprime
                                unique_sums = list(sums)
                                coprime = True
                                for x in range(len(unique_sums)):
                                    for y in range(x+1, len(unique_sums)):
                                        if gcd(unique_sums[x], unique_sums[y]) != 1:
                                            coprime = False
                                            break
                                    if not coprime:
                                        break
                                if coprime:
                                    return [best_A, best_B]
    
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick self-test
    A, B = construct()
    print(f"Found A={A}, B={B}")
    sums = set(a+b for a in A for b in B)
    print(f"Distinct sums: {len(sums)}, Values: {sorted(sums)}")
