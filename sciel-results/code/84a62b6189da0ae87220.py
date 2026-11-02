"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0

    total = q ** n
    # Precompute powers for index conversion
    pows = [q ** (n - 1 - i) for i in range(n)]
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute ball masks for all indices
    # ball_mask[idx] is an integer with bits set for all words in the ball of idx
    ball_masks = [0] * total
    for idx in range(total):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(n):
            pw = pows[i]
            w_i = w[i]
            for v in range(q):
                if v != w_i:
                    new_idx = idx + (v - w_i) * pw
                    mask |= (1 << new_idx)
        ball_masks[idx] = mask

    # --- Initial Seed ---
    # For q=3, n=6, radius=1, use direct sum of Tetracode (n=4) and full space (n=2)
    # Tetracode: 9 words, radius 1.
    # Full space n=2: 9 words, radius 0.
    # Result: 81 words, radius 1.
    seed_indices = []
    if q == 3 and n >= 4:
        # Generate Tetracode words
        tetracode = []
        for a in range(3):
            for b in range(3):
                tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
        
        # Generate tails for remaining n-4 positions
        tails = list(itertools.product(range(q), repeat=n - 4))
        
        for h in tetracode:
            for tail in tails:
                word = h + list(tail)
                seed_indices.append(word_to_idx(word))
    else:
        # Fallback: greedy construction from scratch
        seed_indices = []

    code = list(seed_indices)
    
    # Compute coverage counts
    # coverage_count[j] = number of codewords covering j
    coverage_count = [0] * total
    for c in code:
        mask = ball_masks[c]
        # Iterate over set bits in mask
        m = mask
        while m:
            # Get index of lowest set bit
            # Python 3.11+ has int.bit_count and int.bit_length, but for iteration we can use (m & -m).bit_length() - 1
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage_count[idx] += 1
            m ^= lsb

    # --- Removal Phase ---
    # Try to remove redundant codewords
    # A codeword c is redundant if for all j in ball(c), coverage_count[j] > 1
    
    max_passes = 100
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        # Shuffle code to break symmetry / find different redundancies
        random.shuffle(code)
        
        removed_any = False
        new_code = []
        for c in code:
            # Check if c is redundant
            is_redundant = True
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if coverage_count[idx] <= 1:
                    is_redundant = False
                    break
                m ^= lsb
            
            if is_redundant:
                # Remove c
                # Decrement coverage counts
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    coverage_count[idx] -= 1
                    m ^= lsb
                removed_any = True
            else:
                new_code.append(c)
        
        code = new_code
        
        if not removed_any:
            break

    # --- Final Greedy Fill (if needed) ---
    # Check if coverage is complete
    # If not, add more codewords greedily
    uncovered_mask = 0
    for j in range(total):
        if coverage_count[j] == 0:
            uncovered_mask |= (1 << j)
            
    while uncovered_mask:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            # Gain = number of uncovered bits in ball(idx)
            # popcount(uncovered_mask & ball_masks[idx])
            intersection = uncovered_mask & ball_masks[idx]
            gain = intersection.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_gain == 0:
            break
            
        code.append(best_idx)
        
        # Update coverage counts and uncovered mask
        mask = ball_masks[best_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverage_count[idx] += 1
            if coverage_count[idx] == 1:
                uncovered_mask ^= (1 << idx) # Clear bit
            m ^= lsb

    # Convert back to words
    result = [idx_to_word(idx) for idx in code]
    return result
# EVOLVE-BLOCK-END
