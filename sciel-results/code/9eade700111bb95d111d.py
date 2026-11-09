import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Proven valid tail: for a >= 171, any b,c > a gives b+c > 2a >= 2*171=342
    # but b+c mod a = 0 requires b+c >= a (true) and b+c = k*a for some k
    # Since b,c <= 256, b+c <= 512. For a >= 171, k can be 1 or 2.
    # k=1: b+c = a, but b > a and c >= b > a, so b+c > 2a > a. Impossible.
    # k=2: b+c = 2a. With b,c in [a+1, 256], b+c in [2a+2, 512]. 
    #   2a in [342, 512]. Need b+c = 2a with b > a, c >= b.
    #   b >= a+1, c >= b >= a+1, so b+c >= 2a+2 > 2a. Impossible!
    # So [171, 256] is valid.
    
    base_tail = list(range(171, 257))  # 86 elements
    
    # Bitset-based fast validity check
    # For each potential element x in [1, 256], precompute which existing elements
    # would create conflicts when x is added
    
    def can_add_fast(S_bits, x):
        """
        Fast check using bitsets. S_bits is a 257-bit integer where bit i is set
        if element i is in the set. x is the candidate to add.
        """
        if (S_bits >> x) & 1:
            return True
        
        # Case 1: x is smallest (a=x), need b,c in S with x < b <= c and x | (b+c)
        # For each multiple m of x where m > 2*x (since b+c >= 2*(x+1) = 2x+2),
        # check if there exist b,c in S with b > x, c >= b, b+c = m
        for m in range(2 * x + 2, 513, x):
            # Need b in [x+1, m//2] and c = m-b in [b, 256]
            # So b in [max(x+1, m-256), min(m//2, 256)]
            b_lo = max(x + 1, m - 256)
            b_hi = min(m // 2, 256)
            if b_lo > b_hi:
                continue
            # Check if there's a b in [b_lo, b_hi] in S such that m-b is also in S and m-b >= b
            for b in range(b_lo, b_hi + 1):
                c = m - b
                if c >= b and c <= 256 and ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    return False
        
        # Case 2: x is middle (b=x), need a in S with a < x, c in S with c >= x, a | (x+c)
        # For each a in S with a < x, check if there's c in S with c >= x and (x+c) % a == 0
        # (x+c) % a == 0 means x+c = k*a for some k, so c = k*a - x
        for a in range(1, x):
            if not ((S_bits >> a) & 1):
                continue
            # c = k*a - x, need c >= x and c <= 256
            # k*a - x >= x => k*a >= 2x => k >= ceil(2x/a)
            # k*a - x <= 256 => k*a <= 256+x => k <= floor((256+x)/a)
            k_lo = math.ceil(2 * x / a)
            k_hi = (256 + x) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - x
                if c >= x and c <= 256 and ((S_bits >> c) & 1):
                    return False
        
        # Case 3: x is largest (c=x), need a,b in S with a < b <= x, a | (b+x)
        # For each a in S, check if there's b in S with a < b <= x and (b+x) % a == 0
        # (b+x) % a == 0 means b+x = k*a, so b = k*a - x
        for a in range(1, x):
            if not ((S_bits >> a) & 1):
                continue
            # b = k*a - x, need a < b <= x
            # k*a - x > a => k*a > x+a => k > (x+a)/a => k >= floor((x+a)/a) + 1
            # k*a - x <= x => k*a <= 2x => k <= floor(2x/a)
            k_lo = (x + a) // a + 1
            k_hi = (2 * x) // a
            for k in range(k_lo, k_hi + 1):
                b = k * a - x
                if a < b <= x and ((S_bits >> b) & 1):
                    return False
        
        return True
    
    def try_greedy_fast(S_initial_bits, order):
        """Try greedy addition in given order using fast bitset check."""
        S_bits = S_initial_bits
        for x in order:
            if (S_bits >> x) & 1:
                continue
            if can_add_fast(S_bits, x):
                S_bits |= (1 << x)
        return S_bits
    
    # Convert base_tail to bitset
    def list_to_bits(lst):
        bits = 0
        for x in lst:
            bits |= (1 << x)
        return bits
    
    base_bits = list_to_bits(base_tail)
    
    # Candidate orders for adding elements 1..170
    candidates_to_add = list(range(1, 171))
    
    best_bits = base_bits
    best_size = len(base_tail)
    
    def bits_to_set(bits):
        return [i for i in range(1, 257) if (bits >> i) & 1]
    
    # Order 1: descending (170, 169, ..., 1)
    order_desc = list(range(170, 0, -1))
    S1_bits = try_greedy_fast(base_bits, order_desc)
    S1_size = bin(S1_bits).count('1')
    if S1_size > best_size:
        best_bits = S1_bits
        best_size = S1_size
    
    if time.time() > deadline:
        return bits_to_set(best_bits)
    
    # Order 2: ascending (1, 2, ..., 170)
    order_asc = list(range(1, 171))
    S2_bits = try_greedy_fast(base_bits, order_asc)
    S2_size = bin(S2_bits).count('1')
    if S2_size > best_size:
        best_bits = S2_bits
        best_size = S2_size
    
    if time.time() > deadline:
        return bits_to_set(best_bits)
    
    # Order 3: priority - try elements that are less likely to conflict
    def count_divisors_in_range(x, lo, hi):
        """Count how many numbers in [lo, hi] are divisible by x"""
        if x == 0:
            return 0
        return (hi // x) - ((lo - 1) // x)
    
    order_priority = sorted(candidates_to_add, key=lambda x: count_divisors_in_range(x, 171, 256))
    S3_bits = try_greedy_fast(base_bits, order_priority)
    S3_size = bin(S3_bits).count('1')
    if S3_size > best_size:
        best_bits = S3_bits
        best_size = S3_size
    
    if time.time() > deadline:
        return bits_to_set(best_bits)
    
    # Order 4: mix - try from the middle outward
    mid = 85
    order_mid = []
    for d in range(86):
        if mid - d >= 1:
            order_mid.append(mid - d)
        if mid + d <= 170 and mid + d != mid - d:
            order_mid.append(mid + d)
    S4_bits = try_greedy_fast(base_bits, order_mid)
    S4_size = bin(S4_bits).count('1')
    if S4_size > best_size:
        best_bits = S4_bits
        best_size = S4_size
    
    if time.time() > deadline:
        return bits_to_set(best_bits)
    
    # Try to extend the tail further
    def is_valid_bits(bits):
        """Check if the set represented by bits is valid."""
        S_list = bits_to_set(bits)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Try to lower the tail start
    for tail_start in range(170, 140, -1):
        tail = list(range(tail_start, 257))
        tail_bits = list_to_bits(tail)
        if is_valid_bits(tail_bits):
            order_below = list(range(tail_start - 1, 0, -1))
            S_new_bits = try_greedy_fast(tail_bits, order_below)
            S_new_size = bin(S_new_bits).count('1')
            if S_new_size > best_size:
                best_bits = S_new_bits
                best_size = S_new_size
            break
    
    if time.time() > deadline:
        return bits_to_set(best_bits)
    
    # Final: try many more random orders using the faster check
    rng = random.Random(42)
    
    # Now we can afford more iterations due to faster checks
    for trial in range(500):
        if time.time() > deadline - 2.0:
            break
        order = candidates_to_add[:]
        rng.shuffle(order)
        S_t_bits = try_greedy_fast(base_bits, order)
        S_t_size = bin(S_t_bits).count('1')
        if S_t_size > best_size:
            best_bits = S_t_bits
            best_size = S_t_size
    
    # Also try random orders with different seeds for diversity
    for seed in range(10):
        if time.time() > deadline - 1.0:
            break
        rng2 = random.Random(seed * 1000 + 7)
        for trial in range(50):
            if time.time() > deadline - 1.0:
                break
            order = candidates_to_add[:]
            rng2.shuffle(order)
            S_t_bits = try_greedy_fast(base_bits, order)
            S_t_size = bin(S_t_bits).count('1')
            if S_t_size > best_size:
                best_bits = S_t_bits
                best_size = S_t_size
    
    return bits_to_set(best_bits)
