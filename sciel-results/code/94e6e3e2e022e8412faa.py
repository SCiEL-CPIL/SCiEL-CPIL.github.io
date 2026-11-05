import numpy as np
from itertools import combinations
import time

def construct():
    start_time = time.time()
    
    # GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
    GF_POLY = 0x11D
    
    def gf_mul(a, b):
        result = 0
        for _ in range(8):
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= GF_POLY
        return result
    
    def gf_pow(base, exp):
        result = 1
        for _ in range(exp):
            result = gf_mul(result, base)
        return result
    
    # Find element of order 17
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1:
            if a != 1:
                alpha = a
                break
    assert alpha is not None
    
    # Non-residues mod 17
    residues = set(pow(i, 2, 17) for i in range(1, 17))
    non_residues = sorted(set(range(1, 17)) - residues)
    
    # Compute generator polynomial g(x)
    g = [1]
    for i in non_residues:
        ai = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j, coeff in enumerate(g):
            new_g[j+1] ^= coeff
            new_g[j] ^= gf_mul(coeff, ai)
        g = new_g
        
    assert len(g) == 9
    
    # Generate all 512 codewords of the [17,9,5] code? No, [18,9,6] shortened is [17,9,5]?
    # The [18,9,6] code has min dist 6. Shortening by one coordinate (fixing it to 0) gives a [17,9,5] code?
    # Actually, shortening a [n,k,d] code by fixing a coordinate to 0 yields a [n-1, k-1, d] code if the punctured code has d, but here we are shortening the extended code.
    # The extended QR code [18,9,6] has all even weight.
    # Shortening at coordinate 17 (parity bit) to 0 leaves the original 17 bits.
    # The distance between any two codewords in the shortened code is at least 6?
    # If two codewords differed in the parity bit, they would have different parity, but all are even, so parity bit is determined by the other 17.
    # If we fix parity to 0, we keep only codewords with even weight in first 17 bits.
    # The distance between two such codewords is the Hamming distance of their 17-bit parts.
    # In the [18,9,6] code, min dist is 6.
    # If we take two codewords with parity 0, their distance in 18 bits is >= 6.
    # Since the 18th bit is 0 for both, the distance in the first 17 bits is the same as in 18 bits.
    # So min dist is >= 6.
    # The code is linear, size 2^9 = 512?
    # Wait, the [18,9,6] code has 512 codewords.
    # Half of them have parity 0 (even weight in 18 bits).
    # Since the 18th bit is the parity of the first 17, if the 18th bit is 0, the weight of the first 17 is even.
    # So there are 256 codewords with even weight in the first 17 bits.
    # These 256 form a [17,8,6] linear code?
    # Dimension 8?
    # The kernel of the parity check map from C (dim 9) to GF(2) (dim 1) is dim 8.
    # So yes, it's a [17,8,6] linear code.
    # Size 256.
    
    words_17 = set()
    for u in range(512):
        word = 0
        for i in range(9):
            if (u >> i) & 1:
                for j in range(9):
                    if g[j]:
                        word ^= 1 << ((i + j) % 17)
        words_17.add(word)
        
    assert len(words_17) == 512
    
    # Shorten: keep codewords with even weight (parity 0)
    # In the [18,9,6] construction, the 18th bit is the parity of the first 17.
    # So "parity bit at position 17 is 0" means the weight of the 17-bit word is even.
    words = []
    for w in words_17:
        if w.bit_count() % 2 == 0:
            words.append(w)
            
    words = list(set(words))
    assert len(words) == 256, f"Got {len(words)} words after shortening"
    
    # Precompute deltas
    all_deltas = []
    for dist in range(1, 6):
        for positions in combinations(range(17), dist):
            delta = 0
            for pos in positions:
                delta |= (1 << pos)
            all_deltas.append(delta)
    all_deltas = np.array(all_deltas, dtype=np.uint32)
    
    best_words = list(words)
    best_size = 256
    
    max_restarts = 100
    
    for _ in range(max_restarts):
        if time.time() - start_time > 55:
            break
            
        current_words = list(words)
        forbidden = np.zeros(131072, dtype=bool)
        
        # Initialize forbidden
        for w in current_words:
            forbidden[w] = True
            forbidden[np.bitwise_xor(w, all_deltas)] = True
            
        while True:
            valid_indices = np.where(~forbidden)[0]
            if len(valid_indices) == 0:
                break
                
            # Random selection
            idx = np.random.choice(valid_indices)
            new_w = int(idx)
            current_words.append(new_w)
            
            # Update forbidden
            forbidden[new_w] = True
            forbidden[np.bitwise_xor(new_w, all_deltas)] = True
            
            if len(current_words) > best_size:
                best_size = len(current_words)
                best_words = list(current_words)
                
    return best_words
