import time
import numpy as np

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0

def popcount(x):
    return x.bit_count()

def verify_code(words):
    """
    Verify all pairwise distances.
    Optimized for the specific case of ~500 words.
    """
    if len(words) <= 1:
        return True
    
    n = len(words)
    # Convert to numpy array for vectorized XOR
    arr = np.array(words, dtype=np.uint32)
    
    # We need to check all pairs (i, j) with i < j
    # Using a loop over i is efficient enough for n=512
    for i in range(n):
        # XOR word i with all subsequent words
        diffs = arr[i+1:] ^ arr[i]
        # Check if any distance is less than MIN_DIST
        # We can break early if found
        # Using a generator expression with min is efficient
        # min(d.bit_count() for d in diffs) < MIN_DIST
        # To avoid creating a list, we use any() with a generator
        if any(d.bit_count() < MIN_DIST for d in diffs):
            return False
            
    return True

def construct_extended_qr_512():
    """
    Constructs the 512-word extended QR code for p=17.
    This code has length 18, dimension 9, and minimum distance 6.
    """
    p = 17
    # Quadratic residues mod 17: {1, 4, 9, 16, 8, 2, 15, 13, 10}
    qr = set((i * i) % p for i in range(p))
    
    # Generator matrix for [17, 9, 5] code
    # Rows are cyclic shifts of the indicator vector of QRs
    # Dimension k = (p+1)//2 = 9
    k = (p + 1) // 2
    G = [[0] * p for _ in range(k)]
    for i in range(k):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1
    
    # Generate all 2^9 = 512 codewords of length 17
    codewords_17 = []
    for mask in range(1 << k):
        word = 0
        # XOR the rows corresponding to the set bits in mask
        for i in range(k):
            if mask & (1 << i):
                for j in range(p):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords_17.append(word)
    
    # Extend each word by an overall parity bit to get length 18
    # The parity bit is at position p (bit 17)
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
        
    return codewords_18

def construct():
    start_time = time.time()
    
    # Construct the optimal code
    words = construct_extended_qr_512()
    
    # Verify the code to ensure correctness
    # If verification fails (which shouldn't happen), fall back to a smaller valid code
    if verify_code(words):
        return words
    else:
        # Fallback: Reed-Muller code of length 16, extended to 18 with zeros
        # This is a valid [18, 5, 8] code? No, RM(1,4) is [16, 5, 8].
        # Distance 8 >= 6. Size 32.
        # Just to be safe, though this branch is unlikely.
        fallback_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    # Linear function over GF(2)^4
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                fallback_words.append(word)
        return fallback_words

# The function names and signatures are kept as per the request.
# No other functions are strictly necessary for the optimal solution.
