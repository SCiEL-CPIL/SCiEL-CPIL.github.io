import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    # Step 1: Find the generator polynomial for the [17,9,5] QR code
    # g(x) is a degree 8 polynomial in GF(2)[x] such that x^17 = 1 mod g(x)
    # g(x) must be irreducible (or product of irreducibles, but for QR it's minimal poly)
    # Actually, we just need x^17 - 1 to be divisible by g(x).
    
    def gf2_poly_mul(a, b, mod_deg=17):
        # Multiply two polynomials represented as integers (bits)
        # a, b are ints
        res = 0
        # We only care about bits 0..16
        # Since deg a < 9 and deg b < 9, product deg < 17
        # So we just do standard multiplication and mask
        # But wait, if we use the polynomial as the generator, 
        # the codewords are multiples of g(x).
        # If deg(g)=8, and deg(m)<9, then deg(m*g) < 17.
        # So the codeword is just m*g in GF(2)[x].
        # We don't even need mod reduction if deg < 17.
        
        # Efficient multiplication:
        # a has bits 0..8
        # b has bits 0..8
        # res will have bits 0..16
        # We can use a simple loop or bit tricks.
        # Given small size, a loop is fine.
        
        # Actually, we can just use Python's big integers?
        # No, GF(2) multiplication is XOR-based.
        # Standard binary multiplication is NOT GF(2) multiplication.
        # GF(2) multiplication: res = sum (a_i * b_j) * x^(i+j)
        
        # Let's implement a fast GF(2) mul for small polys.
        # a, b are < 2^9
        # res < 2^17
        
        # Method:
        # for i in range(9):
        #   if a & (1 << i):
        #     res ^= b << i
        
        # This is 9 iterations. Very fast.
        
        r = 0
        # Only need to check bits 0..8 of a
        for i in range(9):
            if (a >> i) & 1:
                r ^= (b << i)
        return r & ((1 << 17) - 1) # Mask to 17 bits just in case, though not needed if deg<17

    g_poly = None
    # Candidates: x^8 + ... + 1
    # Bit 8 is 1, Bit 0 is 1. Bits 1..7 are free.
    for mid_bits in range(128):
        # Construct g
        g = (1 << 8) | (1 << 0) | mid_bits
        # Check if x^17 == 1 mod g
        # Compute x^17 mod g
        # x^17 = x * x^16 = ...
        # We can compute x^17 mod g by repeated squaring or just shifting.
        # Since g is degree 8, x^8 = g + lower terms.
        # Let's compute x^17 mod g.
        
        # Start with x (which is 2)
        # We want x^17.
        # We can just compute the remainder of x^17 divided by g.
        
        # Efficient way:
        # x^17 mod g.
        # Let's compute powers of x mod g.
        # x^0 = 1
        # x^1 = 2
        # x^2 = 4
        # ...
        # x^8 = g + (g - x^8) = g + lower_part(g)
        # So x^8 mod g = lower_part(g)
        
        lower_g = g & 0xFF # bits 0..7
        
        # x^17 = x^9 * x^8
        # x^9 = x * x^8 = x * lower_g
        # This might get messy.
        
        # Alternative: Just check if g divides x^17 - 1.
        # x^17 - 1 is a fixed polynomial.
        # We can precompute the coefficients of x^17 - 1.
        # It has 1 at pos 17 and pos 0.
        # We want to check if (x^17 + 1) % g == 0.
        
        # Let's do polynomial division of x^17 + 1 by g.
        # Dividend: 1 << 17 | 1
        # Divisor: g
        
        # Since degree is small, we can do it manually.
        # But 128 candidates * 17 steps is trivial.
        
        dividend = (1 << 17) | 1
        # We want to see if dividend is divisible by g.
        # Leading term of dividend is x^17.
        # Leading term of g is x^8.
        # Quotient term: x^9.
        # Subtract x^9 * g from dividend.
        # x^9 * g = x^9 * (x^8 + ...) = x^17 + ...
        
        # Let's just use a generic GF(2) mod function.
        
        def gf2_mod(a, b):
            # a mod b, deg(b) = 8
            # a is int, b is int
            # deg(a) can be up to 17
            # while deg(a) >= deg(b):
            #   shift = deg(a) - deg(b)
            #   a ^= b << shift
            # return a
            
            # Find deg(a)
            # Since a is small, we can just loop
            # deg(b) is 8
            # We need to cancel bits 17 down to 8
            
            # a has bit 17 set initially.
            # b has bit 8 set.
            
            # Step 1: Cancel bit 17
            # a ^= b << (17-8) = b << 9
            # Step 2: Check new highest bit
            # ...
            
            # Simpler:
            # We know b is degree 8.
            # We want a % b.
            # We can iterate from high to low.
            
            # Actually, we can just compute x^17 mod g directly using the property that x^8 = g_low + x^8? No.
            # x^8 = g - g_low = g_low (in GF(2))
            # So x^8 ≡ g_low (mod g)
            
            # Let's compute x^17 mod g using this.
            # x^17 = x^9 * x^8 ≡ x^9 * g_low
            # x^9 = x * x^8 ≡ x * g_low
            # So x^17 ≡ x * g_low * g_low = x * (g_low)^2 ? No, (g_low)^2 is not simple.
            
            # Let's stick to the division algorithm.
            
            # a = (1 << 17) | 1
            # b = g
            
            # We will cancel bits from 17 down to 8.
            # Current a
            curr_a = (1 << 17) | 1
            
            # Bit 17 is set.
            # Shift b by (17-8) = 9
            curr_a ^= (b << 9)
            
            # Now highest bit of curr_a is at most 16 (since we canceled 17)
            # Actually, b << 9 has bit 17 set.
            # b has bit 8 set. b << 9 has bit 17 set.
            # curr_a had bit 17 set. So bit 17 is canceled.
            
            # Now we need to cancel bits 16 down to 8.
            # The highest possible bit in curr_a now is 16 (from b<<9, if b has bit 7 set, then bit 16 is set).
            
            for i in range(16, 7, -1):
                if (curr_a >> i) & 1:
                    shift = i - 8
                    curr_a ^= (b << shift)
                    
            # If curr_a is 0, then g divides x^17+1.
            if curr_a == 0:
                g_poly = g
                break
                
    if g_poly is None:
        # Fallback: Greedy search
        return greedy_construct(N, MIN_DIST, deadline)

    # Step 2: Generate the [17,9,5] code
    # Codewords are m * g_poly for m in 0..511
    words = []
    for m in range(512):
        c = gf2_poly_mul(m, g_poly)
        words.append(c)
        
    # Step 3: Filter for even weight to get [17,8,6]
    even_words = [w for w in words if bin(w).count('1') % 2 == 0]
    
    # Step 4: Verify min distance
    if not verify_min_dist(even_words, N, MIN_DIST):
        # Try the other generator polynomial if possible
        # Or just fallback
        pass
    else:
        # We have 256 words
        best_words = even_words
        
        # Step 5: Greedy Extension
        # Try to add more words
        best_words = greedy_extend(best_words, N, MIN_DIST, deadline)
        
        return best_words

    # Fallback if QR code failed
    return greedy_construct(N, MIN_DIST, deadline)

def verify_min_dist(words, N, min_d):
    if len(words) < 2:
        return True
    # For 256 words, O(N^2) is 65k checks. Fast.
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            if (words[i] ^ words[j]).bit_count() < min_d:
                return False
    return True

def greedy_construct(N, min_d, deadline):
    # Start with 0
    words = [0]
    words_arr = np.array([[0]*N], dtype=np.uint8)
    
    def try_add(candidate, words_list, words_arr):
        cand_bits = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        xor = words_arr ^ cand_bits
        dists = xor.sum(axis=1)
        if dists.min() >= min_d:
            words_list.append(candidate)
            words_arr = np.vstack([words_arr, cand_bits])
            return True, words_arr
        return False, words_arr

    no_add = 0
    while time.time() < deadline - 1:
        cand = random.getrandbits(N)
        added, words_arr = try_add(cand, words, words_arr)
        if added:
            no_add = 0
        else:
            no_add += 1
            if no_add > 10000:
                break
    return words

def greedy_extend(words, N, min_d, deadline):
    # words is a list of ints
    # Convert to numpy for fast distance check
    words_arr = np.array([[w >> j & 1 for j in range(N)] for w in words], dtype=np.uint8)
    
    def try_add(candidate, words_list, words_arr):
        cand_bits = np.array([(candidate >> j) & 1 for j in range(N)], dtype=np.uint8)
        xor = words_arr ^ cand_bits
        dists = xor.sum(axis=1)
        if dists.min() >= min_d:
            words_list.append(candidate)
            words_arr = np.vstack([words_arr, cand_bits])
            return True, words_arr
        return False, words_arr

    no_add = 0
    # Try to add random words
    while time.time() < deadline - 0.5:
        cand = random.getrandbits(N)
        added, words_arr = try_add(cand, words, words_arr)
        if added:
            no_add = 0
        else:
            no_add += 1
            if no_add > 50000:
                break
                
    return words
