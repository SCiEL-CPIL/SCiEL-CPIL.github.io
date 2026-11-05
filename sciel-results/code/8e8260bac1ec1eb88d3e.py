import time
from math import isqrt

def construct():
    start = time.time()
    deadline = start + 140.0  # 20s safety margin

    # Generate powerful numbers up to 10^18 efficiently.
    # A powerful number n = p1^e1 * p2^e2 * ... with ei >= 2 for all i.
    # Equivalently, n = a^2 * b^3 where b is square-free (in the standard parametrization).
    # But we need to be careful: every prime factor appears with exponent >= 2.
    # The standard way: n = product of p_i^(2*k_i) where k_i >= 1.
    # Actually, the cleanest: generate all powerful numbers <= 10^18 by iterating
    # over square-full numbers. Let's use the fact that n = a^2 * b^3 with b square-free
    # is NOT quite right for "powerful" in the sense that every prime appears squared.
    # 
    # Powerful number: every prime factor p of n has p^2 | n.
    # Such numbers are of the form n = a^2 * b^3 where b is square-free? No.
    # Actually, if n is powerful, write n = product p_i^(e_i) with e_i >= 2.
    # We can write e_i = 2 + 3*k_i + r_i where r_i in {0,1,2}? That's messy.
    # 
    # Better approach: iterate over possible "square parts" and "cube parts".
    # Let n = s^2 * c where s is a positive integer and c is a powerful number with all
    # exponents odd? No, that's not right either.
    #
    # Cleanest enumeration: 
    # For n powerful <= 10^18, we have n = product p_i^(e_i), e_i >= 2.
    # Let's enumerate by the largest prime factor. But that's complex.
    #
    # Alternative: generate all powerful numbers by:
    # n = a^2 * b^3 where b is square-free. This is a known parametrization?
    # Let me verify: if n is powerful, write n = product p_i^(e_i), e_i >= 2.
    # Let e_i = 2 + 3*k_i + r_i? No. Let's just use: n = m^2 * k where k is powerful
    # and all exponents in k are odd? 
    #
    # Actually, the standard enumeration: iterate over all c (the "cube part") up to
    # (10^18)^(1/3) ≈ 10^6, where c is any positive integer, and for each c,
    # the square part s must satisfy s^2 * c <= 10^18 and s^2 * c is powerful.
    # This is getting complicated.
    #
    # Let me use a simpler but valid approach: 
    # Generate powerful numbers by iterating over all possible "squareful" forms.
    # A powerful number is of the form product p_i^(2 + 3*t_i) or product p_i^(2+t_i*3) ...
    #
    # Practical approach: iterate over all a from 1 to 10^9, and for each a,
    # compute a^2. Then multiply by cubes of square-free numbers b where b^3 * a^2 <= 10^18.
    # But we need a^2 * b^3 to be powerful, which requires that for each prime p in b,
    # the exponent in b^3 plus the exponent in a^2 must be >= 2, which is always true
    # since b contributes at least 3 and a contributes at least 0. Wait, but if p divides
    # b, then p^3 divides b^3, so p^3 divides n. If p divides a but not b, then p^2
    # divides a^2, so p^2 divides n. If p divides both, p^(2*ea + 3*eb) divides n, which
    # is >= p^2. So YES, n = a^2 * b^3 where b is square-free is powerful. But is every
    # powerful number of this form? 
    #
    # Claim: n is powerful iff n = a^2 * b^3 for some positive integers a, b with b square-free.
    # Proof: If n = a^2 * b^3 with b square-free, then for each prime p:
    #   - if p | b: v_p(n) = 2*v_p(a) + 3*v_p(b) >= 3 >= 2
    #   - if p | a but p ∤ b: v_p(n) = 2*v_p(a) >= 2
    #   - if p ∤ ab: v_p(n) = 0
    # So n is powerful. Conversely, if n is powerful, write n = product p_i^(e_i), e_i >= 2.
    # Let b = product of p_i where e_i is odd (so v_p(b) = 1 for these p_i). Then b is square-free.
    # Let a = product of p_i^((e_i - 1)/2) for e_i odd, and p_i^(e_i/2) for e_i even.
    # Then a^2 * b^3 = product p_i^(2*((e_i-1)/2) + 3*1) for e_i odd = product p_i^(e_i-1+3) = p_i^(e_i+2)? That's wrong.
    #
    # Let me redo: if e_i is odd, e_i >= 3. Let b contain p_i once. Then v_p(b^3) = 3.
    # We need v_p(a^2) = e_i - 3, so e_i - 3 must be even and non-negative, i.e., e_i is odd and >= 3.
    # v_p(a) = (e_i - 3)/2. If e_i is even, v_p(a) = e_i/2, v_p(b) = 0.
    # So yes, every powerful number can be written as a^2 * b^3 with b square-free.
    #
    # So we enumerate: for b square-free up to 10^6, and for a such that a^2 * b^3 <= 10^18.

    # Step 1: Generate square-free numbers up to 10^6
    MAX_B = 10**6
    is_square_free = [True] * (MAX_B + 1)
    is_square_free[0] = False
    for p in range(2, isqrt(MAX_B) + 1):
        p2 = p * p
        for j in range(p2, MAX_B + 1, p2):
            is_square_free[j] = False
    
    square_free = [i for i in range(2, MAX_B + 1) if is_square_free[i]]
    
    # Step 2: Generate all powerful numbers up to 10^18
    powerful_set = set()
    powerful_set.add(1)
    
    for b in square_free:
        b3 = b ** 3
        if b3 > 10**18:
            break
        max_a = isqrt(10**18 // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val > 10**18:
                break
            powerful_set.add(val)
    
    powerful_list = sorted(powerful_set)
    powerful_set_set = set(powerful_list)
    
    # Step 3: Search for N with many representations
    # We'll look for N that is sum of two powerful numbers in many ways.
    # Use convolution-like approach: for each powerful number a, and each powerful number b >= a,
    # if a + b <= 10^18, increment count[a+b].
    
    # But the set of powerful numbers is large. Let's estimate: 
    # For b=1: a up to 10^9, so ~10^9 numbers? No, a is up to isqrt(10^18) = 10^9.
    # That's too many. We need a smarter approach.
    
    # Alternative: for a given N, the number of representations is the number of powerful a
    # such that a <= N/2 and N-a is powerful.
    
    # Let's search over candidate N values. Good candidates might be large and have many
    # powerful divisors or be near the center of the range.
    
    # Strategy: sample candidate N values and count representations.
    # To count efficiently, we can iterate over powerful numbers a up to N/2 and check if N-a is powerful.
    
    # Let's first get a list of powerful numbers up to 10^18/2 = 5*10^17 for the smaller element.
    
    # The powerful numbers are sparse. Let's count how many we have.
    n_powerful = len(powerful_list)
    
    # For each candidate N, count = sum(1 for a in powerful_list if a <= N//2 and (N-a) in powerful_set_set)
    
    # To find good N, we can try N values that are sums of pairs of powerful numbers.
    # Let's generate candidate N's by summing pairs of powerful numbers.
    
    # But that's O(n^2) which is too much if n is large.
    
    # Let's try a different approach: pick random powerful numbers and sum them.
    import random
    random.seed(42)
    
    candidate_Ns = set()
    sample_size = min(n_powerful, 10000)
    indices = list(range(n_powerful))
    
    for _ in range(50000):
        if time.time() > deadline - 30:
            break
        i = random.randint(0, n_powerful - 1)
        j = random.randint(0, n_powerful - 1)
        if i <= j:
            s = powerful_list[i] + powerful_list[j]
            if s <= 10**18:
                candidate_Ns.add(s)
    
    # Also try some structured candidates
    for _ in range(10000):
        if time.time() > deadline - 30:
            break
        i = random.randint(0, n_powerful - 1)
        s = powerful_list[i] + powerful_list[i]
        if s <= 10**18:
            candidate_Ns.add(s)
    
    # Evaluate top candidates
    best_N = 2
    best_pairs = []
    best_count = 0
    
    # Sort candidates and evaluate the most promising ones
    sorted_candidates = sorted(candidate_Ns, reverse=True)[:2000]
    
    for N in sorted_candidates:
        if time.time() > deadline - 10:
            break
        pairs = []
        for a in powerful_list:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set_set:
                pairs.append([a, b])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N
            best_pairs = pairs
    
    # If we haven't found many pairs, do a more targeted search
    if best_count < 10:
        # Try to find N by meeting in the middle
        # For powerful numbers in a certain range, compute all pair sums
        if time.time() < deadline - 15:
            # Use a subset of powerful numbers in a denser region
            mid = n_powerful // 2
            subset_start = max(0, mid - 5000)
            subset_end = min(n_powerful, mid + 5000)
            subset = powerful_list[subset_start:subset_end]
            
            from collections import defaultdict
            sum_counts = defaultdict(list)
            
            for i in range(len(subset)):
                if time.time() > deadline - 10:
                    break
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s <= 10**18:
                        sum_counts[s].append((subset[i], subset[j]))
            
            for N, pairs in sum_counts.items():
                if len(pairs) > best_count:
                    best_count = len(pairs)
                    best_N = N
                    best_pairs = pairs
    
    return {'N': best_N, 'pairs': best_pairs}
