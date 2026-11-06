import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    # 1. Initialize with the Reed-Muller construction (32 words)
    # This provides a valid starting core.
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # We also include the zero word and all-weight-18 word? 
    # The RM code above generates 32 words. 
    # Let's verify the initial set is valid.
    # The seed program says "Initial Reed-Muller construction has 32 words".
    # We assume they are valid for d>=6.
    
    current_words = words[:]
    n_words = len(current_words)
    
    # Precompute weights for faster distance checks? 
    # Distance between u and v is popcount(u ^ v).
    # We can use numpy for batch checking if we convert to arrays, 
    # but integer XOR + bit_count is quite fast in Python 3.10+ if optimized.
    
    # To speed up the greedy addition, we can maintain a list of words.
    # Checking a new candidate against all existing words takes O(N) time.
    
    # Strategy: 
    # Phase 1: Greedy Random Addition
    # Try to add random 18-bit words to the set if they are at distance >= 6 from all existing.
    # Phase 2: Local Search / Annealing
    # If greedy stagnates, try to "swap" or "flip" bits in existing words to free up space for new ones, 
    # or simply try more random candidates with a different distribution.
    
    # However, the known optimum is 512 (linear QR code). 
    # The QR code is a specific linear code. 
    # Can we find it via search? Unlikely to hit the exact 512-dimensional subspace randomly.
    # But maybe we can find a *different* large code?
    # A_2(18, 6) is actually 512. 
    # The QR code achieves this. 
    # Are there other codes of size 512? 
    # If the QR code is unique up to equivalence, random search won't find it.
    # But maybe we can find a code of size > 32 easily?
    
    # Let's implement a robust greedy search.
    # We will try to add random words.
    # To make it faster, we can use a "rejection sampling" approach.
    
    # Optimization: 
    # Instead of checking every new word against ALL existing words, 
    # we can use a trie or a spatial index, but for N=18, simple loop is likely fast enough for a few hundred words.
    # 512 words * 512 checks * 18 bits is small.
    
    # Let's try to recover the QR code structure or just grow the set.
    # Actually, the inspiration says: "Greedy random addition starting from Reed-Muller seed should find many more valid codewords... Measured: 1 -> 6.031."
    # This suggests that random addition works well.
    
    # Let's implement a greedy loop that runs until time limit.
    
    random.seed(42) # Deterministic for reproducibility in evaluation
    
    # We can also try to add the zero word and all-ones word if not present?
    # The RM code generated above:
    # slope=0, int=0 -> 0
    # slope=0, int=1 -> 1111111111111111 (16 ones)
    # Wait, the RM code of length 16? The task is length 18.
    # The seed program's fallback:
    # for point in range(16): ... word |= bit << point
    # This creates 16-bit words. 
    # The task requires 18-bit words. 
    # The seed program's fallback returns these 16-bit words? 
    # "Each integer encodes one N-bit binary codeword, including leading zeros."
    # If N=18, the integers should be in [0, 2**18).
    # The seed program's fallback produces words < 2**16. This is valid (leading zeros).
    # But distance is calculated on the 18-bit representation? 
    # "Every pair is checked exactly using integer XOR and population count."
    # If we have 16-bit words, their XOR is 16-bit. Popcount is on that.
    # Leading zeros don't affect popcount of XOR if both have 0 there.
    # So 16-bit words embedded in 18-bit space are fine.
    
    # Let's try to add random 18-bit words.
    
    def is_valid_candidate(candidate, existing_words):
        for w in existing_words:
            if (candidate ^ w).bit_count() < 6:
                return False
        return True

    # To speed up, we can store words in a list.
    # We will iterate random candidates.
    
    # Improvement: Use a "batch" random generation.
    # Generate 1000 random 18-bit words, check them one by one.
    
    # Also, we can try to "perturb" existing words.
    
    # Let's start the loop.
    last_improvement_time = time.time()
    
    while time.time() < deadline - 5.0:
        # Generate a batch of random candidates
        # We can bias towards low weight or high weight? 
        # Random uniform is a good start.
        
        # Try adding random words
        added = 0
        for _ in range(1000):
            if time.time() > deadline - 5.0:
                break
            candidate = random.randint(0, (1 << 18) - 1)
            if is_valid_candidate(candidate, current_words):
                current_words.append(candidate)
                added += 1
                last_improvement_time = time.time()
        
        if added > 0:
            # If we added some, continue
            pass
        else:
            # If we didn't add any in a batch, try a different strategy:
            # Try to find a word that is "close" to existing ones but just outside the forbidden zone?
            # Or just keep trying random.
            # Maybe increase the batch size or change randomness?
            # For now, just keep looping. The probability of finding a valid word decreases as the set grows.
            
            # Alternative: Try to add words that are complements of existing words?
            # If w is in the code, is ~w in the code?
            # For linear codes, no. For nonlinear, maybe.
            # Let's try adding complements of existing words.
            for w in current_words[:]:
                comp = w ^ ((1 << 18) - 1)
                if comp not in current_words:
                    if is_valid_candidate(comp, current_words):
                        current_words.append(comp)
                        added += 1
                        last_improvement_time = time.time()
                        break # Restart the check if we added one
            
            if added == 0:
                # Try flipping bits of existing words to create new valid words
                # This is a local search.
                for w in current_words:
                    for i in range(18):
                        new_w = w ^ (1 << i)
                        if new_w not in current_words:
                            if is_valid_candidate(new_w, current_words):
                                current_words.append(new_w)
                                added += 1
                                last_improvement_time = time.time()
                                break
                    if added > 0:
                        break
                
                if added == 0:
                    # Stagnation. 
                    # We can try to "swap" a word: remove one, add two? 
                    # Or just accept that we might be stuck at a local maximum.
                    # Given the time limit, we should probably just return what we have.
                    # But let's try one more thing: random restart with a different seed?
                    # No, we want to maximize the single set.
                    
                    # If we are stuck, we can break and return.
                    # But maybe we can try more randoms?
                    # Let's break if we haven't improved in a while?
                    # The deadline check handles the global stop.
                    # To avoid wasting time on impossible searches, we can break if no improvement in 5 seconds.
                    if time.time() - last_improvement_time > 5.0:
                        break

    return current_words
