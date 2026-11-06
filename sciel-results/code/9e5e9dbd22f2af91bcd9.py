import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin

    N = 18
    MIN_DIST = 6
    
    # 1. Generate the base [16, 5, 8] Reed-Muller code words
    # Extended to length 18 by appending 00.
    
    def rm_word(slope, intercept):
        w = 0
        for point in range(16):
            bit = ((slope & point).bit_count() + intercept) % 2
            if bit:
                w |= (1 << point)
        return w
        
    # Basis for RM(1,4)
    g1 = rm_word(0, 1) # All 1s
    g2 = rm_word(1, 0)
    g3 = rm_word(2, 0)
    g4 = rm_word(4, 0)
    g5 = rm_word(8, 0)
    
    basis = [g1, g2, g3, g4, g5]
    
    # Generate all 32 codewords
    codewords = []
    for mask in range(1 << 5):
        w = 0
        for i in range(5):
            if (mask >> i) & 1:
                w ^= basis[i]
        codewords.append(w)
        
    codeword_set = set(codewords)
    
    # Helper to check if a candidate is valid
    def is_valid(candidate):
        if candidate in codeword_set:
            return False
        for w in codewords:
            if (w ^ candidate).bit_count() < MIN_DIST:
                return False
        return True

    # Greedy addition
    random.seed(42)
    
    # We want to reach 512 codewords.
    # If we can't find random ones, we try mutations.
    
    while len(codewords) < 512 and time.time() < deadline:
        found = False
        
        # Try random candidates first (fast to generate)
        # Limit random tries per iteration to avoid infinite loop if stuck
        for _ in range(1000):
            if time.time() > deadline:
                break
            v = random.getrandbits(N)
            # Quick filter: weight must be at least MIN_DIST? No, distance to 0 must be >= MIN_DIST.
            # Since 0 is in the code (mask=0), v must have weight >= 6.
            if v.bit_count() < MIN_DIST:
                continue
            if is_valid(v):
                codewords.append(v)
                codeword_set.add(v)
                found = True
                break
        
        if found:
            continue
            
        # If random failed, try mutations of existing codewords
        # This is a structural search
        mutated = False
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Pick a random codeword
            base_w = random.choice(codewords)
            # Generate a random mutation mask with weight 3 or 4
            # We need to flip enough bits to potentially jump out of the forbidden zone
            # Distance 6 from base_w is required.
            # So we flip at least 6 bits?
            # If we flip 6 bits, distance to base_w is 6.
            # But it might be close to OTHER codewords.
            # Let's try weight 6, 7, 8 mutations.
            wt = random.choice([6, 7, 8, 9])
            # Choose 'wt' random bits to flip
            bits = random.sample(range(N), wt)
            mask = 0
            for b in bits:
                mask |= (1 << b)
            v = base_w ^ mask
            
            if v in codeword_set:
                continue
            if is_valid(v):
                codewords.append(v)
                codeword_set.add(v)
                mutated = True
                break
                
        if not mutated and not found:
            # If we are stuck, we might need to restart or accept.
            # But 512 is achievable. Let's keep trying with different seeds or just keep going.
            # To avoid infinite loop on a hard spot, we can randomize the 'random' seed state.
            pass

    return codewords
