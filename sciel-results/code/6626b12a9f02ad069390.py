"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt

def construct():
    # Generate powerful numbers up to a large limit
    # A powerful number has the form a^2 * b^3 where b is squarefree
    # Actually, every powerful number can be written as a^2 * b^3 with b squarefree.
    # Let's generate all powerful numbers <= L.
    
    L = 10**9  # We can go up to 10^9 for powerful numbers, since we need a+b=N <= 10^18
    # Wait, if a and b are powerful and a+b=N, with N up to 10^18, then a and b can be up to 10^18.
    # But we need to generate powerful numbers. How many powerful numbers are there up to 10^18?
    # Powerful numbers are of the form a^2 * b^3. The count up to X is roughly O(X^(2/5)) or similar.
    # Actually, the number of powerful numbers up to X is about c * X^(1/2) * log(X)^(1/3) ... 
    # More precisely, it's known to be ~ c * X^(2/5) for some constant? No.
    # Let's think: powerful numbers are n such that for every prime p|n, p^2|n.
    # They are of the form a^2 * b^3 where b is squarefree.
    # The count of powerful numbers <= X is asymptotic to zeta(3/2)/zeta(3) * sqrt(X) ... no.
    # Actually, it's known that the number of powerful numbers <= X is ~ c * X^(1/2) * (log X)^(1/3)? 
    # Let me just generate them. For X = 10^9, there are about 10^9^(1/2) = 31623 powerful numbers? 
    # Actually, the density is about 1/sqrt(x) times a constant. So up to 10^9, we have ~ 30k powerful numbers.
    # Up to 10^18, we would have ~ 10^9 powerful numbers, which is too many.
    # So we need to be clever. We can't generate all powerful numbers up to 10^18.
    
    # Strategy: Generate powerful numbers up to some limit L_pow.
    # For a given N, we want to count pairs (a, b) with a+b=N, a<=b, a,b powerful.
    # If we generate all powerful numbers up to L_pow, and if N <= 2*L_pow, we can check all pairs.
    # But N can be up to 10^18. So we need powerful numbers up to 10^18.
    # That's too many to generate explicitly.
    
    # Alternative approach: For each powerful number a, b = N-a must also be powerful.
    # We can generate powerful numbers up to 10^9 (about 30k of them).
    # Then for N up to 2*10^9 = 2*10^9, we can find all pairs.
    # But we want to maximize the number of pairs for some N <= 10^18.
    # The number of powerful numbers up to X grows like X^(1/2) * polylog(X).
    # So the number of representations of N as a sum of two powerful numbers is at most ~ sqrt(N) * polylog(N).
    # To maximize this, we want N to be large, but we can only generate powerful numbers up to some limit.
    
    # Let's generate powerful numbers up to L = 10^9.
    # There are about 30k such numbers.
    # Then we can compute all sums a+b where a,b are powerful, a<=b, and count how many times each sum occurs.
    # The maximum sum would be 2*10^9.
    # We use np.bincount to count the sums.
    
    # Generate powerful numbers up to L
    def generate_powerful(L):
        # Powerful numbers are of the form a^2 * b^3 where b is squarefree
        # Actually, every powerful number can be uniquely written as a^2 * b^3 with b squarefree.
        # Let's iterate over b (squarefree) and a.
        # b^3 <= L => b <= L^(1/3)
        # For each b, a^2 <= L / b^3 => a <= sqrt(L / b^3)
        
        max_b = int(L ** (1/3)) + 1
        powerful = set()
        for b in range(1, max_b + 1):
            # Check if b is squarefree
            is_sqfree = True
            for p in range(2, int(b ** 0.5) + 1):
                if b % (p * p) == 0:
                    is_sqfree = False
                    break
            if not is_sqfree:
                continue
            max_a = isqrt(L // (b ** 3))
            for a in range(1, max_a + 1):
                val = a * a * b ** 3
                if val <= L:
                    powerful.add(val)
        return sorted(powerful)
    
    L = 10**9
    powerful = generate_powerful(L)
    M = len(powerful)
    
    # Use vectorized approach to count sums
    # We want to count, for each sum s, the number of pairs (i, j) with i <= j and powerful[i] + powerful[j] = s.
    # We can use np.add.outer and then bincount, but that's O(M^2) memory.
    # For M = 30k, M^2 = 9*10^8, which is about 7.2 GB for int64. Too much.
    
    # Alternative: Use the fact that we only need the maximum count.
    # We can iterate over each powerful number a, and for each a, check which b = N - a are powerful.
    # But we want to find N that maximizes the count.
    
    # Let's use a different approach: for each a in powerful, for each b in powerful with b >= a,
    # compute s = a + b, and count.
    # We can do this in chunks to avoid memory issues.
    
    # Actually, let's just use a dictionary to count sums, processing in chunks.
    from collections import defaultdict
    sum_counts = defaultdict(int)
    
    # Process in chunks to avoid O(M^2) memory
    chunk_size = 1000
    for i_start in range(0, M, chunk_size):
        i_end = min(i_start + chunk_size, M)
        for i in range(i_start, i_end):
            a = powerful[i]
            # For each a, b must be >= a and a + b <= 2*L
            # We can binary search for the range of b values
            # b >= a and b <= 2*L - a
            # Find indices j such that powerful[j] >= a and powerful[j] <= 2*L - a
            j_start = i  # since powerful is sorted and b >= a means j >= i
            # Find j_end: largest j such that powerful[j] <= 2*L - a
            max_b_val = 2 * L - a
            # Binary search for j_end
            lo, hi = j_start, M - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if powerful[mid] <= max_b_val:
                    lo = mid + 1
                else:
                    hi = mid - 1
            j_end = lo  # j_end is the first index where powerful[j_end] > max_b_val, so valid j are [j_start, j_end-1]
            for j in range(j_start, j_end):
                s = a + powerful[j]
                sum_counts[s] += 1
    
    # Find the N with the maximum number of pairs
    if not sum_counts:
        # Fallback: just return a simple case
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_counts, key=sum_counts.get)
    best_count = sum_counts[best_N]
    
    # Generate the pairs for best_N
    pairs = []
    for a in powerful:
        b = best_N - a
        if b < a:
            continue
        if b > 2 * L:
            continue
        # Check if b is powerful
        # We can check by seeing if b is in the powerful set
        # But we only generated powerful numbers up to L, so b might be > L
        # If b > L, we can't check it. So we should only consider b <= L.
        if b > L:
            continue
        # Check if b is in powerful
        # Use binary search
        lo, hi = 0, M - 1
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            if powerful[mid] == b:
                found = True
                break
            elif powerful[mid] < b:
                lo = mid + 1
            else:
                hi = mid - 1
        if found:
            pairs.append([a, b])
    
    # Remove duplicates and sort
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [[int(p[0]), int(p[1])] for p in pairs]
    
    return {'N': int(best_N), 'pairs': pairs}
# EVOLVE-BLOCK-END
