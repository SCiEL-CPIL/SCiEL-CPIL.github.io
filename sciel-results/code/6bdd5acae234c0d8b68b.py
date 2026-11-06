"""Construct the largest possible collection of 18-bit words with pairwise Hamming distance at least 6."""
# EVOLVE-BLOCK-START
import time
import random

def _hamming_distance(a, b):
    return (a ^ b).bit_count()

def _is_valid(words, min_dist=6):
    """Check if all pairwise distances are at least min_dist."""
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if (wi ^ words[j]).bit_count() < min_dist:
                return False
    return True

def _construct_qr_code():
    """
    Construct the extended quadratic residue code QR(17) of length 18.
    
    The QR code over GF(2) with prime p=17 has:
    - Length 17 (cyclic code)
    - Dimension 9
    - Minimum distance 5
    - Number of codewords: 2^9 = 512
    
    We extend by one parity bit to get length 18, dimension 9, minimum distance 6.
    
    The generator polynomial is a degree 8 factor of (x^17 - 1)/(x-1) = 1 + x + ... + x^16.
    Over GF(2), this factors into two irreducible degree 8 polynomials.
    """
    # The two irreducible factors of Phi_17(x) = 1 + x + x^2 + ... + x^16 over GF(2)
    # f1 = x^8 + x^4 + x^3 + x^2 + 1
    # f2 = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    
    # Represent polynomials as lists of coefficients, index 0 is x^0
    f1 = [1, 0, 0, 1, 1, 1, 0, 0, 1]  # 1 + x^2 + x^3 + x^4 + x^8
    f2 = [1, 1, 0, 1, 1, 1, 0, 1, 1]  # 1 + x + x^3 + x^4 + x^5 + x^7 + x^8
    
    def _poly_mul(p1, p2):
        """Multiply two polynomials over GF(2)."""
        res = [0] * (len(p1) + len(p2) - 1)
        for i, c1 in enumerate(p1):
            if c1:
                for j, c2 in enumerate(p2):
                    if c2:
                        res[i + j] ^= 1
        return res
    
    # Verify that f1 * f2 = Phi_17(x)
    phi17 = [1] * 17
    prod = _poly_mul(f1, f2)
    if len(prod) != 17 or any(prod[i] != phi17[i] for i in range(17)):
        return None
    
    # Use f1 as the generator polynomial
    g = f1
    
    # Generate all 512 codewords of the cyclic code
    # Each codeword c(x) = m(x) * g(x) mod (x^17 - 1)
    # Since g(x) divides x^17 - 1, c(x) = m(x) * g(x) has degree < 17 automatically
    words_17 = []
    for msg in range(512):
        # Message polynomial m(x) of degree < 9
        m_poly = [(msg >> i) & 1 for i in range(9)]
        c_poly = _poly_mul(m_poly, g)
        
        # Pad to length 17 if necessary
        if len(c_poly) < 17:
            c_poly = c_poly + [0] * (17 - len(c_poly))
        elif len(c_poly) > 17:
            # This shouldn't happen, but truncate just in case
            c_poly = c_poly[:17]
        
        # Convert to integer
        word = 0
        for i, bit in enumerate(c_poly):
            if bit:
                word |= (1 << i)
        words_17.append(word)
    
    # Extend by parity bit to get length 18
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        w_ext = (w << 1) | parity
        words_18.append(w_ext)
    
    # Verify the code is valid (distance >= 6)
    # Full check is fast enough for 512 words
    n = len(words_18)
    for i in range(n):
        wi = words_18[i]
        for j in range(i + 1, n):
            if (wi ^ words[j]).bit_count() < 6:
                return None  # Fallback if construction is flawed
    
    return words_18

def _greedy_improve(words, min_dist=6, time_limit=5.0, seed=42):
    """Try to add random words to the code."""
    rng = random.Random(seed)
    start = time.time()
    while time.time() - start < time_limit:
        c = rng.randint(0, (1 << 18) - 1)
        if all((c ^ w).bit_count() >= min_dist for w in words):
            words.append(c)
    return words

def construct():
    """
    Construct the largest possible collection of 18-bit words with pairwise Hamming distance at least 6.
    
    The known optimal construction is the extended QR(17) code with 512 codewords.
    """
    start_time = time.time()
    
    # Try to construct the QR code (should give 512 words)
    words = _construct_qr_code()
    
    if words is None:
        # Fallback: try the other factor
        # Actually, let's just use a simpler fallback: the RM code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
    
    # If we have time and fewer than 512 words, try greedy improvement
    if time.time() - start_time < 55.0 and len(words) < 512:
        remaining_time = 55.0 - (time.time() - start_time)
        words = _greedy_improve(words, min_dist=6, time_limit=remaining_time, seed=42)
    
    return words
# EVOLVE-BLOCK-END
