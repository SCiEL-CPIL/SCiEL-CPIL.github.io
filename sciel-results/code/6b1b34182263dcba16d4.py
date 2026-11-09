import math
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pairwise_coprime(lst):
        n = len(lst)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    # Precompute primes up to 256
    primes = set()
    for n in range(2, 257):
        is_p = True
        for d in range(2, int(n**0.5)+1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.add(n)
    
    # Also include 1 (gcd(1, anything) = 1) and small prime powers
    # For pairwise coprime, we can use: primes, and products of disjoint prime sets
    # But simplest: focus on primes first, then add 1, 4, 9, 25, 49, 121, etc.
    
    # Candidate pool: numbers in [2, 256] that are "coprime-friendly"
    # A number is coprime-friendly if it's prime or a prime power
    coprime_friendly = set()
    for n in range(2, 257):
        # Check if n is a prime power
        for p in primes:
            if n % p != 0:
                continue
            temp = n
            while temp % p == 0:
                temp //= p
            if temp == 1:
                coprime_friendly.add(n)
                break
        else:
            # Check if it's a product of exactly 2 distinct primes (semiprime with coprime factors)
            # Actually for pairwise coprime, we just need each number to have disjoint prime factor sets
            # So any number works as long as no two share a prime factor
            # Let's just include all numbers and check pairwise coprimality
            pass
    
    # Better approach: enumerate all numbers 2..256 and use pairwise coprime check
    all_candidates = list(range(2, 257))
    
    # Strategy: 
    # 1. Try to find 9 distinct pairwise coprime numbers that form a 3x3 sum grid
    # 2. A 3x3 sum grid has the property: s[i][j] = base + a_i + b_j
    #    where a_0 = b_0 = 0
    # 3. So we need: base, base+a1, base+a2, base+b1, base+b2, base+a1+b1, base+a1+b2, base+a2+b1, base+a2+b2
    #    all pairwise coprime
    # 4. Equivalently, pick 3 "row offsets" {0, a1, a2} and 3 "col offsets" {0, b1, b2}
    #    and a base, such that all 9 sums are pairwise coprime
    
    # Key structural insight: if we fix the 9 target values (pairwise coprime), 
    # we can check if they fit a 3x3 grid by checking:
    #   s[i][j] - s[i][0] = s[0][j] - s[0][0] for all i,j
    #   i.e., the "differences along rows" are the same for all rows
    #   and the "differences along columns" are the same for all columns
    
    def can_form_grid(nine_vals):
        """Check if 9 values can be arranged in a 3x3 grid where each cell = row_offset + col_offset + base"""
        # Sort and try all permutations? No, too expensive.
        # Instead: in a 3x3 grid s[i][j] = c + r_i + c_j, the structure is:
        # s[i][j] - s[i][0] = c_j - c_0 (independent of i)
        # s[i][0] - s[0][0] = r_i - r_0 (independent of j)
        # So: (s[i][j] - s[i][0]) must equal (s[0][j] - s[0][0]) for all i,j
        # And: (s[i][0] - s[0][0]) must equal (s[i][j] - s[0][j]) for all i,j
        # 
        # Try all 9! = 362880 permutations? Too slow.
        # Better: fix s[0][0] (9 choices), then s[0][1], s[0][2] (8*7=56 choices for first row)
        # Then determine r_1 = s[1][0]-s[0][0], r_2 = s[2][0]-s[0][0]
        # c_1 = s[0][1]-s[0][0], c_2 = s[0][2]-s[0][0]
        # Then s[1][1] should be s[0][0] + r_1 + c_1, s[1][2] = s[0][0]+r_1+c_2, etc.
        # Check if these computed values are in the set
        
        vals_set = set(nine_vals)
        if len(vals_set) != 9:
            return None
        
        # Try each element as s[0][0]
        for s00 in nine_vals:
            remaining = [v for v in nine_vals if v != s00]
            # Try each pair as s[0][1], s[0][2]
            for i in range(len(remaining)):
                s01 = remaining[i]
                rest = remaining[:i] + remaining[i+1:]
                for j in range(len(rest)):
                    s02 = rest[j]
                    rest2 = rest[:j] + rest[j+1:]
                    # Now s[1][0] and s[2][0] must be in rest2
                    # r_1 = s[1][0] - s[0][0], r_2 = s[2][0] - s[0][0]
                    # c_1 = s[0][1] - s[0][0], c_2 = s[0][2] - s[0][0]
                    c1 = s01 - s00
                    c2 = s02 - s00
                    # Try each pair in rest2 as s[1][0], s[2][0]
                    for k in range(len(rest2)):
                        s10 = rest2[k]
                        rest3 = rest2[:k] + rest2[k+1:]
                        for l in range(len(rest3)):
                            s20 = rest3[l]
                            rest4 = rest3[:l] + rest3[l+1:]
                            r1 = s10 - s00
                            r2 = s20 - s00
                            # Compute expected values
                            s11_exp = s00 + r1 + c1
                            s12_exp = s00 + r1 + c2
                            s21_exp = s00 + r2 + c1
                            s22_exp = s00 + r2 + c2
                            # Check if these are in rest4
                            expected = {s11_exp, s12_exp, s21_exp, s22_exp}
                            if expected == set(rest4):
                                # Found a valid grid!
                                # s[0][0]=s00, s[0][1]=s01, s[0][2]=s02
                                # s[1][0]=s10, s[1][1]=s11_exp, s[1][2]=s12_exp
                                # s[2][0]=s20, s[2][1]=s21_exp, s[2][2]=s22_exp
                                grid = [
                                    [s00, s01, s02],
                                    [s10, s11_exp, s12_exp],
                                    [s20, s21_exp, s22_exp]
                                ]
                                return grid
        return None
    
    # Now search for 9 pairwise coprime numbers that can form a grid
    # To make this tractable, we'll:
    # 1. Generate candidate 9-tuples of pairwise coprime numbers
    # 2. Check if they can form a grid
    
    # But 9-tuples from 255 numbers is C(255,9) which is huge.
    # Instead, use the grid structure to guide the search.
    
    # Alternative: directly search over (base, r1, r2, c1, c2) and check if all 9 are pairwise coprime
    # This is what the original code does, but let's make it smarter.
    
    # Even smarter: use the fact that for pairwise coprime, each number should have unique prime factors.
    # If we ensure all 9 are prime, they're automatically pairwise coprime.
    
    # Let's do a focused search for 9 prime sums in a grid
    # base + r_i + c_j must all be prime
    # r_0=0, c_0=0, r_1,r_2,c_1,c_2 > 0 (assuming sorted)
    
    best_count = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify seed
    seed_sums = set()
    for a in [1,7,13]:
        for b in [4,10,16]:
            seed_sums.add(a+b)
    if pairwise_coprime(list(seed_sums)):
        best_count = len(seed_sums)
    
    # Search for 9 prime sums
    # base from 2 to 250, r1 from 1 to 50, r2 from r1+1 to 60, c1 from 1 to 50, c2 from c1+1 to 60
    # But base+r2+c2 <= 256
    
    # Optimize: precompute which (base, r1, r2, c1, c2) give all primes
    # Use a sieve-based approach
    
    # For efficiency, iterate over r1, r2, c1, c2 first (small ranges), then check valid bases
    max_r = 40
    max_c = 40
    
    for r1 in range(1, max_r):
        if time.time() > deadline - 5:
            break
        for r2 in range(r1+1, max_r + r1):
            if time.time() > deadline - 5:
                break
            for c1 in range(1, max_c):
                if time.time() > deadline - 5:
                    break
                for c2 in range(c1+1, max_c + c1):
                    if time.time() > deadline - 5:
                        break
                    # base + 0 + 0 = base must be prime
                    # base + r1, base + r2, base + c1, base + c2,
                    # base + r1 + c1, base + r1 + c2, base + r2 + c1, base + r2 + c2 must be prime
                    # base >= 2, base + r2 + c2 <= 256
                    # So base in [2, 256 - r2 - c2]
                    # And base must be prime, and all 8 other values must be prime
                    
                    # Precompute: for each base in range, check if all 9 are prime
                    # To speed up, we can precompute for each offset set which bases work
                    
                    min_base = 2
                    max_base = 256 - r2 - c2
                    if max_base < min_base:
                        continue
                    
                    # Check each base
                    for base in range(min_base, max_base + 1):
                        if base not in primes:
                            continue
                        if (base + r1) not in primes:
                            continue
                        if (base + r2) not in primes:
                            continue
                        if (base + c1) not in primes:
                            continue
                        if (base + c2) not in primes:
                            continue
                        if (base + r1 + c1) not in primes:
                            continue
                        if (base + r1 + c2) not in primes:
                            continue
                        if (base + r2 + c1) not in primes:
                            continue
                        if (base + r2 + c2) not in primes:
                            continue
                        
                        # All 9 are prime! Now check if we can choose a0, b0 such that A,B in [1,128]
                        # A = [a0, a0+r1, a0+r2], B = [b0, b0+c1, b0+c2]
                        # a0 + b0 = base
                        # a0 >= 1, b0 >= 1, a0+r2 <= 128, b0+c2 <= 128
                        # b0 = base - a0
                        # a0 >= 1, base - a0 >= 1 => a0 <= base-1
                        # a0 + r2 <= 128 => a0 <= 128 - r2
                        # base - a0 + c2 <= 128 => a0 >= base + c2 - 128
                        
                        a0_min = max(1, base + c2 - 128)
                        a0_max = min(base - 1, 128 - r2)
                        
                        if a0_min <= a0_max:
                            a0 = a0_min
                            b0 = base - a0
                            A = [a0, a0 + r1, a0 + r2]
                            B = [b0, b0 + c1, b0 + c2]
                            
                            if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                all_sums = [a + b for a in A for b in B]
                                ds = list(set(all_sums))
                                if len(ds) == 9 and pairwise_coprime(ds):
                                    return [A, B]
    
    # If we didn't find 9 prime sums, search for 8 or 7 distinct pairwise coprime sums
    # using a broader search (not requiring all prime)
    
    if time.time() < deadline - 10:
        # Search for 6+ distinct pairwise coprime sums
        # Use the same parameterization but allow non-prime sums
        # Focus on finding more distinct sums
        
        for base in range(2, 200):
            if time.time() > deadline - 3:
                break
            for r1 in range(1, 30):
                if time.time() > deadline - 3:
                    break
                for r2 in range(r1+1, 30 + r1):
                    if time.time() > deadline - 3:
                        break
                    for c1 in range(1, 30):
                        if time.time() > deadline - 3:
                            break
                        for c2 in range(c1+1, 30 + c1):
                            if time.time() > deadline - 3:
                                break
                            max_sum = base + r2 + c2
                            if max_sum > 256:
                                break
                            
                            nine_sums = [
                                base,
                                base + r1,
                                base + r2,
                                base + c1,
                                base + c2,
                                base + r1 + c1,
                                base + r1 + c2,
                                base + r2 + c1,
                                base + r2 + c2
                            ]
                            
                            ds = list(set(nine_sums))
                            if len(ds) < best_count:
                                continue
                            if not pairwise_coprime(ds):
                                continue
                            
                            a0_min = max(1, base + c2 - 128)
                            a0_max = min(base - 1, 128 - r2)
                            
                            if a0_min <= a0_max:
                                a0 = a0_min
                                b0 = base - a0
                                A = [a0, a0 + r1, a0 + r2]
                                B = [b0, b0 + c1, b0 + c2]
                                
                                if all(1 <= x <= 128 for x in A + B) and len(set(A)) == 3 and len(set(B)) == 3:
                                    all_sums = [a + b for a in A for b in B]
                                    ds2 = list(set(all_sums))
                                    if len(ds2) > best_count and pairwise_coprime(ds2):
                                        best_A = A
                                        best_B = B
                                        best_count = len(ds2)
    
    return [best_A, best_B]
