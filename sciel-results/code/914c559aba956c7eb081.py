import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 18
    MIN_DIST = 6
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    # Construct the extended QR(17) code deterministically
    p = 17
    
    # Quadratic residues modulo 17: {1, 2, 4, 8, 9, 13, 15, 16}
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    # Create the base vector v of length 17
    v = 0
    for i in range(p):
        if i in residues:
            v |= (1 << i)
    
    # Generate the 8 cyclic shifts of v
    basis = []
    current = v
    mask_p = (1 << p) - 1
    for _ in range((p - 1) // 2):  # 8 shifts
        basis.append(current)
        current = ((current << 1) | (current >> (p - 1))) & mask_p
    
    # Add the all-ones vector (9th basis vector)
    basis.append((1 << p) - 1)
    
    num_basis = len(basis)  # 9
    
    # Vectorized generation of all 512 codewords using NumPy
    coeffs = np.arange(1 << num_basis, dtype=np.uint16)
    code_words = np.zeros(1 << num_basis, dtype=np.uint32)
    
    for i in range(num_basis):
        mask = (coeffs & (1 << i)) != 0
        code_words[mask] ^= basis[i]
    
    # Compute parity bits and form 18-bit words
    weights = np.array([w.bit_count() for w in code_words], dtype=np.uint8)
    parities = weights % 2
    full_words = (code_words.astype(np.uint32) << 1) | parities.astype(np.uint32)
    
    code = full_words.tolist()
    
    # Verify no duplicates
    if len(set(code)) != len(code):
        code = list(set(code))
    
    best_code = list(code)
    best_size = len(best_code)
    
    # Convert to numpy array for vectorized distance checks
    code_arr = np.array(best_code, dtype=np.uint32)
    
    # Phase 1: Deterministic structured search for additional words
    # Enumerate all 18-bit words by weight 6 to 12
    # Use vectorized checks against the current code
    
    # Pre-generate all candidates by weight for efficient iteration
    # Total candidates: sum(C(18,w) for w in 6..12)
    # C(18,6)=18564, C(18,7)=31824, C(18,8)=43758, C(18,9)=48620,
    # C(18,10)=43758, C(18,11)=31824, C(18,12)=18564 => ~236,912 total
    
    def generate_words_of_weight(n, w):
        """Generate all n-bit words with exactly w bits set."""
        if w == 0:
            yield 0
            return
        if w == n:
            yield (1 << n) - 1
            return
        # Use combinations of positions
        from itertools import combinations as comb
        for positions in comb(range(n), w):
            word = 0
            for pos in positions:
                word |= (1 << pos)
            yield word
    
    # Try to add words, starting from lower weights (more constrained)
    # Use vectorized distance check
    def can_add_vectorized(w, code_arr, min_dist):
        """Check if word w can be added to code_arr with min distance min_dist."""
        xor_vals = code_arr ^ w
        # Count bits in each xor value
        # Use a fast method: convert to bytes and use lookup or bit manipulation
        # For 512 words, this is fast enough
        dists = np.array([v.bit_count() for v in xor_vals], dtype=np.uint8)
        return np.min(dists) >= min_dist
    
    # Phase 1: Vectorized Greedy Search
    # Precompute lookup table for 6-bit popcount
    popcount6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)
    
    random.seed(42)
    attempts = 0
    max_attempts = 2000000 # Increase attempts since check is faster
    
    # Batch size for vectorized check
    batch_size = 2000
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        # Generate a batch of random candidates
        # Filter by weight 6-12 to avoid trivial rejections
        candidates = []
        # Generate more than needed to account for weight filtering
        num_gen = batch_size * 2 
        raw_candidates = np.random.randint(0, 1 << N, size=num_gen, dtype=np.uint32)
        
        # Filter by weight
        weights = np.array([w.bit_count() for w in raw_candidates], dtype=np.uint8)
        valid_weights = (weights >= 6) & (weights <= 12)
        candidates = raw_candidates[valid_weights]
        
        if len(candidates) == 0:
            attempts += batch_size
            continue
        
        # Limit batch size to avoid memory issues (e.g., max 5000)
        if len(candidates) > 5000:
            candidates = candidates[:5000]
            
        cand_arr = candidates.astype(np.uint32)
        
        # Vectorized distance check
        # xors shape: (B, M)
        xors = cand_arr[:, None] ^ code_arr[None, :]
        
        # Compute popcount using lookup table
        # Split into 3 parts of 6 bits
        part1 = xors & 0x3F
        part2 = (xors >> 6) & 0x3F
        part3 = (xors >> 12) & 0x3F
        
        dists = popcount6[part1] + popcount6[part2] + popcount6[part3]
        
        min_dists = np.min(dists, axis=1)
        valid_mask = min_dists >= MIN_DIST
        
        if np.any(valid_mask):
            valid_candidates = cand_arr[valid_mask]
            # Add them one by one to update code_arr
            for w in valid_candidates:
                if w not in best_code:
                    best_code.append(int(w))
                    code_arr = np.array(best_code, dtype=np.uint32)
        
        attempts += batch_size
    
    # Phase 2: Swap-based local search (if time remains)
    # Try to swap pairs of codewords to find a larger code
    # This is more expensive but might find improvements
    
    if time.time() < deadline - 5.0 and len(best_code) < 600:
        # Try limited swap search
        random.seed(123)
        swap_attempts = 0
        max_swap_attempts = 50000
        
        while time.time() < deadline - 5.0 and swap_attempts < max_swap_attempts:
            swap_attempts += 1
            
            # Pick two random codewords to potentially replace
            if len(best_code) < 2:
                break
            
            i, j = random.sample(range(len(best_code)), 2)
            
            # Generate random replacement for codeword i
            w_new = random.randint(0, (1 << N) - 1)
            wt_new = w_new.bit_count()
            if wt_new < 6 or wt_new > 12:
                continue
            
            # Check if w_new can replace best_code[i]
            # Remove best_code[i], check if w_new is valid against remaining
            remaining = [c for k, c in enumerate(best_code) if k != i]
            
            valid = True
            for c in remaining:
                if (w_new ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid and w_new not in remaining:
                # Also check distance to best_code[j] specifically (already in remaining)
                # Replace
                best_code[i] = w_new
                # Don't increase size, just swap
    
    return best_code
