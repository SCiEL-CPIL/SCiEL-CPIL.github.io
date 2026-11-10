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
    
    primes_set = set(i for i in range(2, 257) if is_prime[i])

    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5 # Fallback from current best knowledge

    # Check if a set of values is pairwise coprime
    def is_pairwise_coprime(values):
        vals = list(set(values))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    # Strategy: Find A, B such that all 9 sums are distinct and pairwise coprime.
    # Iterate A from small values.
    # For each A, find valid Bs.
    
    # Optimization: 
    # If A contains even and odd, parity constraints apply.
    # Best chance: All sums odd primes.
    # Case 1: A odd, B even.
    # Case 2: A even, B odd.
    
    # Let's just iterate all A in a reasonable range and check.
    # Range 1..60 is likely sufficient.
    
    for a1 in range(1, 61):
        if time.time() > deadline:
            return [best_A, best_B]
        for a2 in range(a1+1, 61):
            if time.time() > deadline:
                return [best_A, best_B]
            for a3 in range(a2+1, 61):
                if time.time() > deadline:
                    return [best_A, best_B]
                
                A = (a1, a2, a3)
                
                # Find all b in [1, 128] such that a1+b, a2+b, a3+b are pairwise coprime
                valid_bs = []
                for b in range(1, 129):
                    s1, s2, s3 = a1+b, a2+b, a3+b
                    # Check pairwise coprime for this triple
                    if gcd(s1, s2) == 1 and gcd(s1, s3) == 1 and gcd(s2, s3) == 1:
                        valid_bs.append(b)
                
                if len(valid_bs) < 3:
                    continue
                
                # We need to pick 3 bs such that the 9 sums are distinct and pairwise coprime.
                # Sums are a_i + b_j.
                # If a_i are distinct and b_j are distinct, sums can still collide.
                # Collision: a_i + b_j == a_k + b_l  => a_i - a_k == b_l - b_j
                # To maximize distinctness, we should pick b's that avoid these differences.
                
                # Try combinations of valid_bs
                for b1, b2, b3 in combinations(valid_bs, 3):
                    B = (b1, b2, b3)
                    sums = set()
                    for a in A:
                        for b in B:
                            sums.add(a+b)
                    
                    if len(sums) > best_count:
                        if is_pairwise_coprime(sums):
                            best_count = len(sums)
                            best_A = list(A)
                            best_B = list(B)
                            
                            if best_count == 9:
                                return [best_A, best_B]
    
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick self-test
    A, B = construct()
    print(f"Found A={A}, B={B}")
    sums = set(a+b for a in A for b in B)
    print(f"Distinct sums: {len(sums)}, Values: {sorted(sums)}")
