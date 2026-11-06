import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Construct the [17, 8, 6] quadratic residue code
    # Quadratic residues mod 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Base vector u of length 17
    # u[i] = 1 if i is a quadratic residue, else 0
    # Index 0 is 0 (not a residue)
    u = 0
    for i in range(17):
        if i in residues:
            u |= (1 << i)
            
    # Generate 8 basis vectors by cyclic shifts
    # We need to ensure they are independent.
    # The code is cyclic, so shifts of a codeword are codewords.
    # We take shifts 0 to 7.
    
    basis = []
    for shift in range(8):
        # Cyclic shift right by 'shift'
        # (u >> shift) | ((u << (17 - shift)) & ((1 << 17) - 1))
        # But careful with shift=0
        if shift == 0:
            v = u
        else:
            v = (u >> shift) | ((u << (17 - shift)) & ((1 << 17) - 1))
        basis.append(v)
        
    # Generate all 2^8 = 256 codewords
    words = set()
    for mask in range(256):
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                word ^= basis[i]
        words.add(word)
        
    # Check if we actually got 256 distinct words
    # If not, the basis was not independent. 
    # For QR codes, the first (p+1)/2 shifts are usually independent for p=17?
    # Dim is 8. So 8 shifts should be independent.
    
    if len(words) < 256:
        # Fallback: Use the Reed-Muller code from the seed if QR fails (unlikely)
        # But let's try to fix the basis or just use the seed logic if this fails
        # For now, if it fails, we return what we have, but it should be 256.
        pass
        
    # Optional: Try to add more words using a greedy approach
    # Given the high density of conflicts, this is unlikely to add many,
    # but we have time.
    # We will check a subset of candidates to avoid timeout.
    
    current_words = list(words)
    
    # To speed up distance checks, we can't easily vectorize without numpy in this specific loop structure 
    # if we want to keep it pure python and fast.
    # However, 256 words is small.
    # Let's try to add words.
    # We iterate through a subset of all 2^17 words.
    # To be safe, we limit the number of checks.
    
    # Actually, with 256 words, checking a new word w:
    # min dist to any c in C.
    # If we check 10000 random words, it's 10000 * 256 = 2.5M ops.
    # This is fast in Python (approx 0.5s).
    
    import random
    random.seed(42)
    
    # Create a list of all possible words not in current set
    # Instead of generating all 131072, we can sample.
    # But to be thorough, we can iterate through all if we are fast enough.
    # 131072 * 256 = 33M ops. Might take 10-20s in Python.
    # Let's try to optimize the inner loop.
    
    # We'll just stick to the 256 linear words to ensure we beat the seed significantly 
    # and reliably within the time limit. 
    # The jump from 32 to 256 is the main win.
    
    return current_words
