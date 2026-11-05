import numpy as np

def construct():
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    poly = 0x11D
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return p

    def gf_pow(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul(result, a)
            a = gf_mul(a, a)
            n >>= 1
        return result

    # Find a primitive element
    # We need an element alpha such that its order is 255.
    # Then beta = alpha^15 will have order 17.
    # Let's test alpha = 3
    alpha = 3
    # Check if alpha is primitive
    # alpha^255 == 1 and alpha^(255/p) != 1 for p in {3, 5, 17}
    is_primitive = True
    if gf_pow(alpha, 255) != 1:
        is_primitive = False
    else:
        for p in [3, 5, 17]:
            if gf_pow(alpha, 255 // p) == 1:
                is_primitive = False
                break
    
    if not is_primitive:
        # Find a primitive element
        for a in range(2, 256):
            if gf_pow(a, 255) != 1: continue
            ok = True
            for p in [3, 5, 17]:
                if gf_pow(a, 255 // p) == 1:
                    ok = False
                    break
            if ok:
                alpha = a
                break
    
    beta = gf_pow(alpha, 15)
    
    # Verify beta^17 == 1
    # assert gf_pow(beta, 17) == 1
    # assert beta != 1
    
    # Compute minimal polynomial of beta
    # Roots are beta^(2^i) for i=0..7
    p = [1] # p(x) = 1
    
    for i in range(8):
        r = gf_pow(beta, (1 << i))
        # Multiply p(x) by (x + r)
        new_p = [0] * (len(p) + 1)
        for j in range(len(p)):
            new_p[j] ^= gf_mul(p[j], r)
            new_p[j+1] ^= p[j]
        p = new_p
        
    # p is [c0, c1, ..., c8]
    # g(x) = p[0] + p[1]x + ... + p[8]x^8
    # Note: p[8] should be 1
    
    codewords = []
    seen = set()
    
    for msg in range(256):
        m = [0] * 9
        for i in range(9):
            if (msg >> i) & 1:
                m[i] = 1
        
        h = [0] * 17
        for i in range(9):
            if m[i] == 1:
                for j in range(9):
                    if p[j] == 1:
                        h[i+j] ^= 1
        
        word = 0
        for i in range(17):
            if h[i]:
                word |= (1 << i)
        
        # The [18,9,6] code is the extension.
        # We shorten by fixing the parity bit to 0.
        # This corresponds to keeping words with even weight in the [17,9,5] code.
        if word.bit_count() % 2 == 0:
            if word not in seen:
                seen.add(word)
                codewords.append(word)
                
    # Now try to add more words greedily
    # Use NumPy for accelerated distance checks
    
    current_words = codewords[:]
    seen = set(current_words)
    
    # Create a boolean mask for seen words
    is_seen = np.zeros(2**17, dtype=bool)
    for w in current_words:
        is_seen[w] = True
        
    # Get indices of unseen words
    unseens = np.where(~is_seen)[0]
    
    # Function to compute popcount vectorized
    def popcount_vec(x):
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x = (x * 0x01010101) >> 24
        return x

    batch_size = 5000
    
    for i in range(0, len(unseens), batch_size):
        batch = unseens[i:i+batch_size]
        
        # Convert current words to numpy array
        cw_arr = np.array(current_words, dtype=np.uint32)
        
        # Compute XORs
        xor_vals = batch[:, None] ^ cw_arr[None, :]
        
        # Compute distances
        dists = popcount_vec(xor_vals)
        
        # Find candidates that are valid against all current words
        # valid_mask[i] is True if min distance >= 6
        valid_mask = (dists >= 6).all(axis=1)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            cands_to_add = batch[valid_indices]
            
            # Add valid candidates one by one, checking against the updated current_words
            for c in cands_to_add:
                valid = True
                for w in current_words:
                    if (c ^ w).bit_count() < 6:
                        valid = False
                        break
                if valid:
                    current_words.append(c)
                    seen.add(c)
                    is_seen[c] = True
                    
    return current_words
