import time

def construct():
    start_time = time.time()
    deadline = 55.0 # 5s safety margin
    
    # 1. Generate base [18, 9, 6] QR code (512 words)
    residues = set()
    for x in range(1, 18):
        residues.add((x * x) % 17)
        
    gen_words = []
    for i in range(9):
        word = 0
        parity = 0
        for r in residues:
            pos = (i + r) % 17
            word |= (1 << pos)
            parity ^= 1
        word |= (parity << 17)
        gen_words.append(word)
        
    codes = []
    for mask in range(512):
        word = 0
        m = mask
        g_idx = 0
        while m:
            if m & 1:
                word ^= gen_words[g_idx]
            m >>= 1
            g_idx += 1
        codes.append(word)
        
    # 2. Attempt to expand the code (Nonlinear search)
    # We have 512 words. Can we add more?
    # We iterate through random candidates or a structured list.
    # Given time constraints, we might not add many.
    # But let's try a simple greedy addition of random words.
    
    # Optimization: Use a set for fast membership check? 
    # No, we need distance check against ALL existing words.
    # 512 words. Checking 1 candidate takes ~512 ops.
    # In 50 seconds, we can check millions of candidates.
    
    import random
    random.seed(42) # Deterministic
    
    # We can also try structured candidates: 
    # e.g., words with low weight, or words from other codes.
    # But random is fine.
    
    # To speed up distance check, we can pre-compute? No, just use bit_count.
    
    # Let's try to add words.
    # We stop if we hit the deadline.
    
    # A better strategy for expansion: 
    # If we can't add random words easily, maybe we can't beat 512.
    # But let's try.
    
    # Actually, if the max is 512, this loop will run for 50s doing nothing.
    # That's fine, as long as we return the 512 words.
    
    # However, if we find 1 word, score improves slightly.
    
    # Let's limit the expansion attempts to avoid wasting time if it's futile.
    # Or just run it until deadline.
    
    # One improvement: Check if the code is already maximal.
    # If we try 100,000 random candidates and find none, it's likely maximal.
    
    # Let's do a limited expansion.
    max_attempts = 200000
    attempts = 0
    
    while attempts < max_attempts and time.time() < deadline:
        candidate = random.getrandbits(18)
        if candidate in codes:
            attempts += 1
            continue
            
        # Check distance to all existing codes
        valid = True
        for c in codes:
            if (candidate ^ c).bit_count() < 6:
                valid = False
                break
                
        if valid:
            codes.append(candidate)
            # If we add one, we might be able to add more.
            # But the density increases, making it harder.
            
        attempts += 1
        
    return codes
