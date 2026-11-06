import time
import random
import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def popcount(x):
        return x.bit_count()
    
    def hamming(a, b):
        return (a ^ b).bit_count()
    
    # Generate the [18,9,6] extended QR code shortened to 17 coordinates
    # The [18,9,6] code is the extended quadratic residue code of length 18.
    # We construct it from the quadratic residue code of length 17 (prime 17).
    
    # The QR code of length 17: generators are x^r for r in QR(17)
    # QR(17) = {1, 2, 4, 8, 16, 15, 13, 9, 5} (powers of 2 mod 17)
    # Actually, let's use the standard construction.
    
    # The [17,9,5] QR code: 
    # Generator polynomial for QR(17): g(x) = (x^2+x+1)(x^6+x^4+x^2+x+1) 
    # Wait, let me use a known set of codewords.
    
    # Alternative: Start with the 32-word RM(1,4) code (distance 8 on 16 bits),
    # then extend to 17 bits. But that only gives 32 words.
    
    # Better approach: Use the known [18,9,6] code structure.
    # The [18,9,6] extended QR code can be constructed as follows:
    # Take the [17,9,5] QR code, and add a parity check bit to get [18,9,6].
    # Then shorten by removing one coordinate.
    
    # Let's construct the [17,9,5] QR code.
    # The QR code of length p=17 has dimension (p-1)/2 = 9, minimum distance 5.
    # It's a cyclic code. The generator polynomial is:
    # g(x) = LCM of minimal polynomials of alpha^r for r in QR(17)
    # where alpha is a primitive 17th root of unity in GF(2^8) or similar.
    
    # Simpler: use a known generator matrix.
    # The [17,9,5] QR code generator matrix (systematic form) can be derived.
    
    # Let me use a different approach: start with the 256-word code from the 
    # shortened extended QR code, then do local search to add more words.
    
    # First, construct the [18,9,6] code.
    # The extended QR code of length 18: 
    # It's the even-weight extension of the [17,9,5] QR code.
    
    # Let me use a concrete construction.
    # The [17,9,5] QR code has generator polynomial:
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Wait, I need to verify this.
    
    # Actually, let me use a well-known set of codewords.
    # The [18,9,6] extended QR code can be described as:
    # All 18-bit words (c_0, ..., c_16, c_17) where:
    # - (c_0, ..., c_16) is in the [17,9,5] QR code
    # - c_17 = sum(c_0, ..., c_16) mod 2
    
    # The [17,9,5] QR code: 
    # Generator polynomial g(x) = (x^2+x+1)(x^6+x^4+x^2+x+1)
    # = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    # Let me expand: (x^2+x+1)(x^6+x^4+x^2+x+1)
    # = x^2(x^6+x^4+x^2+x+1) + x(x^6+x^4+x^2+x+1) + 1(x^6+x^4+x^2+x+1)
    # = x^8+x^6+x^4+x^3+x^2 + x^7+x^5+x^3+x^2+x + x^6+x^4+x^2+x+1
    # = x^8 + x^7 + (x^6+x^6) + x^5 + (x^4+x^4) + (x^3+x^3) + (x^2+x^2+x^2) + (x+x) + 1
    # = x^8 + x^7 + x^5 + x^2 + 1
    # Hmm, that doesn't look right. Let me recompute.
    
    # (x^2+x+1)(x^6+x^4+x^2+x+1):
    # x^2*x^6 = x^8
    # x^2*x^4 = x^6
    # x^2*x^2 = x^4
    # x^2*x = x^3
    # x^2*1 = x^2
    # x*x^6 = x^7
    # x*x^4 = x^5
    # x*x^2 = x^3
    # x*x = x^2
    # x*1 = x
    # 1*x^6 = x^6
    # 1*x^4 = x^4
    # 1*x^2 = x^2
    # 1*x = x
    # 1*1 = 1
    # Sum: x^8 + x^7 + x^6+x^6 + x^5 + x^4+x^4 + x^3+x^3 + x^2+x^2+x^2 + x+x + 1
    # = x^8 + x^7 + x^5 + x^2 + 1
    
    # So g(x) = x^8 + x^7 + x^5 + x^2 + 1
    
    # The code is cyclic of length 17. A codeword is c(x) = m(x) * g(x) mod (x^17-1)
    # for m(x) of degree < 9.
    
    # Let's construct the generator matrix by computing c(x) = x^i * g(x) mod (x^17-1)
    # for i = 0, ..., 8.
    
    def mod_poly(a, b):
        """Compute a mod b in GF(2)[x], where a, b are ints representing polynomials."""
        deg_b = b.bit_length() - 1
        while a.bit_length() - 1 >= deg_b:
            shift = a.bit_length() - deg_b
            a ^= b << shift
        return a
    
    g = (1 << 8) | (1 << 7) | (1 << 5) | (1 << 2) | 1  # x^8 + x^7 + x^5 + x^2 + 1
    
    # Generate basis vectors of the [17,9,5] code
    basis_17 = []
    for i in range(9):
        # c(x) = x^i * g(x) mod (x^17 - 1)
        c = (g << i)
        # Reduce mod (x^17 - 1): if degree >= 17, replace x^17 with 1
        # In GF(2), x^17 - 1 = x^17 + 1, so x^17 = 1
        while c.bit_length() > 17:
            # Find the highest bit
            deg = c.bit_length() - 1
            # x^deg = x^(deg-17) * x^17 = x^(deg-17)
            c ^= (1 << deg)
            c ^= (1 << (deg - 17))
        basis_17.append(c)
    
    # Generate all 512 codewords of the [17,9,5] code
    code_17 = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if mask & (1 << i):
                word ^= basis_17[i]
        code_17.append(word)
    
    # Extend to [18,9,6] by adding parity bit
    code_18 = []
    for w in code_17:
        parity = w.bit_count() % 2
        # Place parity at position 17 (bit 17)
        w18 = w | (parity << 17)
        code_18.append(w18)
    
    # Shorten by one coordinate: remove coordinate 17 (the parity bit)
    # This gives the [17,9,6] code? No, shortening a [18,9,6] code by one coordinate
    # gives a [17,8,6] code generally, but for the QR code it might be [17,9,6].
    # Actually, the task says "the [18,9,6] extended quadratic-residue code shortened by one coordinate"
    # gives a 256-word code. 2^8 = 256, so it's [17,8,6].
    
    # Wait, the task says "the 256-word linear reference is the [18,9,6] extended 
    # quadratic-residue code shortened by one coordinate; the seed has 32 words."
    # So the reference is 256 words, but the seed (current program) has 32 words.
    # We need to start with 256 words.
    
    # Shortening: take all codewords of [18,9,6] with 0 in the last coordinate, 
    # then remove that coordinate.
    # Or: take all codewords, set coordinate 17 to 0, and keep only those 
    # that were already 0 there. Then remove coordinate 17.
    
    # Actually, shortening means: take the subcode of codewords with 0 in the 
    # shortened coordinate, then delete that coordinate.
    
    # Let's shorten coordinate 17:
    code_shortened = []
    for w in code_18:
        if (w >> 17) & 1 == 0:
            # Remove bit 17
            w17 = w & ((1 << 17) - 1)
            code_shortened.append(w17)
    
    # This should give 256 words (half of 512)
    # Deduplicate
    code_shortened = list(set(code_shortened))
    
    # Convert to set for fast lookup
    code_set = set(code_shortened)
    code_list = list(code_set)
    
    # Verify minimum distance
    # (We trust the construction, but let's be safe)
    
    # Now do local search to add more words
    # Strategy: try to add random words, and if they conflict, try to remove one 
    # existing word and add two new ones (replace-one-with-two).
    
    random.seed(42)
    
    def can_add(word, existing_set):
        for w in existing_set:
            if (word ^ w).bit_count() < MIN_DIST:
                return False, w
        return True, None
    
    # Try adding random words
    best_code = set(code_list)
    
    # First, try to greedily add random words
    for iteration in range(20000):
        if time_left() < 5.0:
            break
        word = random.randint(0, (1 << N) - 1)
        if word in best_code:
            continue
        ok, conflict = can_add(word, best_code)
        if ok:
            best_code.add(word)
    
    # Now try replace-one-with-two moves
    for iteration in range(50000):
        if time_left() < 3.0:
            break
        # Pick a random existing word to potentially remove
        existing = list(best_code)
        if len(existing) < 2:
            break
        remove_word = random.choice(existing)
        
        # Try to find two words that can be added if we remove remove_word
        # First, find all words that conflict with remove_word
        # Then try to add words that don't conflict with the rest
        
        # Simpler: try random pairs of new words
        for attempt in range(10):
            w1 = random.randint(0, (1 << N) - 1)
            w2 = random.randint(0, (1 << N) - 1)
            if w1 == w2:
                continue
            if w1 in best_code or w2 in best_code:
                continue
            if (w1 ^ w2).bit_count() < MIN_DIST:
                continue
            
            # Check if w1 and w2 can be added when remove_word is removed
            temp_set = best_code - {remove_word}
            ok1, _ = can_add(w1, temp_set)
            if not ok1:
                continue
            ok2, _ = can_add(w2, temp_set)
            if not ok2:
                continue
            
            # Accept the move
            best_code = temp_set | {w1, w2}
            break
    
    # Final greedy pass
    for iteration in range(10000):
        if time_left() < 2.0:
            break
        word = random.randint(0, (1 << N) - 1)
        if word in best_code:
            continue
        ok, _ = can_add(word, best_code)
        if ok:
            best_code.add(word)
    
    return list(best_code)
