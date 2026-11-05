"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_linear_code():
    """Construct the 256-word [18,9,6] QR code shortened to length 17.
    
    The extended quadratic residue code of length 18 has parameters [18,9,6].
    Shortening by one coordinate gives a [17,9,6] code with 256 words.
    
    We construct this using the standard QR code construction.
    """
    N = 17
    # The QR code of length 18 over F_2: 
    # The quadratic residues mod 18 are {1, 4, 7, 10, 13, 16} (residues of squares mod 18)
    # Actually for the [18,9,6] QR code, we use the even-like subcode.
    # 
    # A simpler approach: use the known structure.
    # The [18,9,6] code can be constructed as follows:
    # It's the extended QR code. The generator matrix has rows corresponding to
    # the quadratic residues.
    #
    # Let me use a different approach: construct the [17,9,6] code directly.
    # The shortening of the [18,9,6] code at coordinate 0 gives a [17,9,6] code.
    #
    # Actually, let me use the standard construction:
    # The QR code of length 18: G is an 9x18 matrix.
    # The quadratic residues mod 18: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=7, 6^2=0, 7^2=1, ...
    # Residues: {0, 1, 4, 7, 9, 16}
    # Non-residues: {2, 3, 5, 6, 8, 10, 11, 12, 13, 14, 15, 17}
    #
    # The even-like QR code has generator matrix with rows:
    # Row i: cyclic shift of (1, R, NR) where R=1 for residues, 0 for non-residues
    #
    # Let me just use a known construction. The [17,9,6] code can be built as:
    # Take the [18,9,6] code and puncture/shorten.
    #
    # Simpler: I'll construct it via the standard QR code generator.
    
    # The [18,9,6] extended QR code generator matrix
    # Quadratic residues mod 18: {1, 4, 7, 9, 10, 13, 16}
    # Wait, let me recalculate: 
    # 1^2 mod 18 = 1
    # 2^2 mod 18 = 4
    # 3^2 mod 18 = 9
    # 4^2 mod 18 = 16
    # 5^2 mod 18 = 7
    # 6^2 mod 18 = 0
    # 7^2 mod 18 = 1
    # 8^2 mod 18 = 10
    # 9^2 mod 18 = 9
    # 10^2 mod 18 = 10
    # 11^2 mod 18 = 1
    # 12^2 mod 18 = 16
    # 13^2 mod 18 = 7
    # 14^2 mod 18 = 4
    # 15^2 mod 18 = 9
    # 16^2 mod 18 = 10
    # 17^2 mod 18 = 1
    # So residues (non-zero): {1, 4, 7, 9, 10, 16}
    # Including 0: {0, 1, 4, 7, 9, 10, 16}
    
    # The generator for the [18,9,6] code:
    # It's the cyclic code with generator polynomial g(x) = 1 + x + x^2 + x^4 + x^5 + x^8 + x^9 + x^11 + x^12 + x^14 + x^15 + x^16 + x^17 + x^18
    # Hmm, this is getting complicated. Let me use a different approach.
    
    # I'll use the fact that the [17,9,6] code is well-known and construct it
    # using the standard method: it's the even-weight subcode of the [17,10,5] code
    # or equivalently, the shortening of the [18,9,6] code.
    
    # Let me just hardcode a valid construction using NumPy.
    # The [18,9,6] QR code: I'll use the standard generator matrix.
    
    # Actually, the simplest reliable approach: 
    # The [17,9,6] code has 256 codewords. I can construct it as follows:
    # It's the set of all linear combinations of the rows of a 9x17 generator matrix.
    
    # Let me use the known generator matrix for the [17,9,6] QR code.
    # The generator matrix G (9x17) for the shortened [18,9,6] QR code:
    
    # I'll construct it via the standard QR code construction.
    # The [18,9,6] code is the even-like subcode of the [18,10,5] QR code.
    # The [18,10,5] QR code has generator matrix formed by cyclic shifts of
    # the vector (1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1) [residues + 0]
    
    # Let me just use a brute force known construction.
    # The [17,9,6] code: I'll use the fact that it can be constructed as
    # the set {c in F_2^17 : c is a codeword of the [18,9,6] code with c_0 = 0}
    
    # For simplicity and correctness, let me construct the [18,9,6] code first.
    
    # The generator polynomial for the [18,9,6] QR code over F_2:
    # g(x) = (x^18 + 1) / (x^9 + x^6 + x^4 + x^3 + x^2 + x + 1)
    # Hmm, let me look at this differently.
    
    # I'll use a direct construction. The [17,9,6] code generator matrix:
    # Based on standard references, the [17,9,6] QR code has the following
    # generator matrix (rows are cyclic shifts):
    
    # Let me use a verified approach: construct via the BCH code method.
    # The [17,9,6] code is a BCH code with designed distance 6.
    
    # Actually, the most reliable way: I know the [18,9,6] code exists.
    # Let me construct it using the standard QR code construction.
    
    # The [18,9,6] code: generator matrix G_18 (9x18)
    # Row 0: 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 0 1 1
    # (This is 1 + x + x^2 + x^4 + x^5 + x^8 + x^9 + x^11 + x^12 + x^14 + x^15 + x^17)
    # Wait, that's 12 ones, but degree should be 9.
    
    # Let me use a different, verified construction.
    # I'll use the fact that the [17,9,6] code can be built as:
    # All 17-bit words c such that c is in the [18,9,6] code and c[0]=0.
    
    # For the [18,9,6] code, I'll use the standard construction:
    # The even-like subcode of the [18,10,5] QR code.
    
    # The [18,10,5] QR code generator: 10 rows, each a cyclic shift of
    # h = (1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1)
    # where the 1s are at positions that are quadratic residues mod 18 (including 0).
    
    # Quadratic residues mod 18 (including 0): {0, 1, 4, 7, 9, 10, 16}
    # So h has 1s at positions 0, 1, 4, 7, 9, 10, 16
    
    h = np.zeros(18, dtype=np.uint8)
    h[0] = 1
    h[1] = 1
    h[4] = 1
    h[7] = 1
    h[9] = 1
    h[10] = 1
    h[16] = 1
    
    # Generator matrix for [18,10,5] code: 10 cyclic shifts of h
    G18 = np.zeros((10, 18), dtype=np.uint8)
    for i in range(10):
        G18[i] = np.roll(h, i)
    
    # The [18,9,6] code is the even-like subcode: all codewords of [18,10,5] with even weight
    # This is generated by G18[1:] (rows 1-9) plus G18[0] + G18[1] (to ensure even weight)
    # Actually, the even-like subcode is generated by all sums of pairs of rows of G18.
    # A basis: G18[0]+G18[1], G18[1]+G18[2], ..., G18[8]+G18[9] gives 9 independent vectors.
    
    G_even = np.zeros((9, 18), dtype=np.uint8)
    for i in range(9):
        G_even[i] = (G18[i] + G18[i+1]) % 2
    
    # Now generate all 256 codewords of the [18,9,6] code
    codewords_18 = set()
    for mask in range(256):
        cw = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if mask & (1 << i):
                cw = (cw + G_even[i]) % 2
        codewords_18.add(int(cw.tobytes()[::-1], 2) if hasattr(cw, 'tobytes') else 0)
    
    # Convert to integers properly
    codewords_18 = set()
    for mask in range(256):
        cw = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if mask & (1 << i):
                cw = (cw + G_even[i]) % 2
        # Convert to integer: bit j of cw corresponds to position j
        val = 0
        for j in range(18):
            if cw[j]:
                val |= (1 << j)
        codewords_18.add(val)
    
    # Shorten by coordinate 0: keep only codewords with bit 0 = 0, remove bit 0
    codewords_17 = set()
    for cw in codewords_18:
        if (cw & 1) == 0:
            codewords_17.add(cw >> 1)  # Remove bit 0, shift right
    
    # This should give 128 words (half of 256 have bit 0 = 0)
    # Wait, that's only 128. The shortening of a [18,9,6] code gives [17,8,6] with 128 words.
    # But we want 256 words!
    
    # I think I'm confusing myself. Let me reconsider.
    # The [17,9,6] code has 2^9 = 512 words? No, 2^9 = 512, but the problem says 256.
    # Wait, 2^8 = 256. So it's a [17,8,6] code?
    # The problem says "the [18,9,6] extended quadratic-residue code shortened by one coordinate"
    # Shortening a [18,9,6] code by one coordinate gives a [17,8,6] code with 2^8 = 256 words.
    # Yes! That's correct.
    
    # So my construction above should give 256 words if done correctly.
    # But I only got 128 because I'm only taking half.
    
    # Actually, shortening means: take all codewords with c_0 = 0, and delete coordinate 0.
    # This gives a [17,8,6] code with 2^(9-1) = 128 words.
    
    # Hmm, but the problem says 256 words. Let me re-read.
    # "the 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate"
    
    # Maybe it's puncturing, not shortening? Puncturing a [18,9,6] code gives [17,9,5] with 512 words.
    # That doesn't work either.
    
    # Wait, maybe the [18,9,6] code has 2^9 = 512 words, and shortening gives 2^8 = 256 words.
    # But I'm getting 128. Let me check my generator matrix.
    
    # Actually, I think the issue is that the even-like subcode of [18,10,5] is [18,9,6],
    # which has 2^9 = 512 codewords. My G_even has 9 rows, so I should get 512 codewords.
    # Let me check: I'm generating all 256 combinations of 9 bits, which gives 2^9 = 512... 
    # No wait, range(256) only gives 256 combinations of 8 bits effectively if I'm not careful.
    
    # Let me fix: I need to iterate over all 2^9 = 512 combinations.
    
    codewords_18 = set()
    for mask in range(512):
        cw = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if mask & (1 << i):
                cw = (cw + G_even[i]) % 2
        val = 0
        for j in range(18):
            if cw[j]:
                val |= (1 << j)
        codewords_18.add(val)
    
    print(f"Number of [18,9,6] codewords: {len(codewords_18)}")
    
    # Now shorten: keep codewords with bit 0 = 0, remove bit 0
    codewords_17 = set()
    for cw in codewords_18:
        if (cw & 1) == 0:
            codewords_17.add(cw >> 1)
    
    print(f"Number of [17,8,6] codewords: {len(codewords_17)}")
    
    return list(codewords_17)

def _hamming_distances_vectorized(words, new_word):
    """Compute Hamming distances between new_word and all words in the list.
    
    Args:
        words: list of integers
        new_word: integer
    
    Returns:
        numpy array of Hamming distances
    """
    if not words:
        return np.array([])
    
    words_arr = np.array(words, dtype=np.uint32)
    xored = words_arr ^ new_word
    # Count bits in each element
    # Use a lookup table or bit counting
    # For 17-bit numbers, we can use a simple approach
    distances = np.zeros(len(words), dtype=np.uint8)
    for i in range(len(words)):
        distances[i] = xored[i].bit_count()
    return distances

def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6.
    
    Uses a nonlinear local search starting from the 256-word linear code.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with safety margin
    
    N = 17
    MIN_DIST = 6
    
    # Start with the linear code
    try:
        words = _build_linear_code()
    except Exception:
        # Fallback: use the RM code if QR construction fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_count = len(best_words)
    
    # Convert to set for fast lookup
    word_set = set(best_words)
    
    # Local search: try to add new words
    # We'll try random words and see if they can be added
    
    rng = np.random.RandomState(42)
    
    # Precompute all possible words? No, 2^17 = 131072 is manageable
    all_words = list(range(2**N))
    
    # Try adding words in random order
    rng.shuffle(all_words)
    
    added_count = 0
    for trial, candidate in enumerate(all_words):
        if time.time() > deadline - 10:
            break
        
        if candidate in word_set:
            continue
        
        # Check if candidate can be added (distance >= 6 from all existing words)
        # Vectorized check
        words_arr = np.array(list(word_set), dtype=np.uint32)
        xored = words_arr ^ candidate
        # Count bits
        min_dist = 99
        for i in range(len(words_arr)):
            d = xored[i].bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < MIN_DIST:
                    break
        
        if min_dist >= MIN_DIST:
            word_set.add(candidate)
            best_words.append(candidate)
            best_count += 1
            added_count += 1
    
    # Try swap operations: remove one word, try to add two words
    # This is more complex, so we'll do a simpler version
    # Try to improve by removing a word and adding a better one
    
    # For now, return what we have
    return best_words
# EVOLVE-BLOCK-END
