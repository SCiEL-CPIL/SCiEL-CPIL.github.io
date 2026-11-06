import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s kill
    
    N = 17
    MIN_DIST = 6
    UNIVERSE_SIZE = 1 << N
    
    # 1. Construct the 256-word linear code: shortened [18,9,6] extended QR code
    # Quadratic residues mod 19
    qres = set()
    for x in range(1, 19):
        qres.add((x*x) % 19)
    
    # Generator matrix for [18,9,6] extended QR code
    G = np.zeros((9, 18), dtype=np.uint8)
    for i in range(9):
        for j in range(17):
            pos = (j - i) % 19
            if pos in qres:
                G[i, j] = 1
        G[i, 17] = G[i, :17].sum() % 2
        
    # Generate all 256 codewords of the [18,9,6] code
    all_cw_18 = set()
    for mask in range(256):
        cw = np.zeros(18, dtype=np.uint8)
        for i in range(9):
            if (mask >> i) & 1:
                cw = (cw + G[i]) % 2
        val = 0
        for j in range(18):
            val = (val << 1) | int(cw[j])
        all_cw_18.add(val)
        
    # Shorten: delete coordinate 17 (parity), keep codewords with 0 there
    # Coordinate 17 is LSB in our encoding (j=17 is LSB)
    cw_17 = set()
    for val in all_cw_18:
        if (val & 1) == 0:
            cw_17.add(val >> 1)
            
    # Verify distance
    cw_list = list(cw_17)
    min_d = 999
    for i in range(len(cw_list)):
        for j in range(i+1, len(cw_list)):
            d = (cw_list[i] ^ cw_list[j]).bit_count()
            if d < min_d:
                min_d = d
                
    if min_d < 6:
        # Fallback: Greedy construction
        cw_set = set()
        for w in range(1 << 17):
            if all((w ^ cw).bit_count() >= 6 for cw in cw_set):
                cw_set.add(w)
        cw_list = list(cw_set)
        
    # 2. Local Search / Simulated Annealing
    # We represent the code as a list of integers.
    # We will use a NumPy array of bits for fast distance calculation.
    
    current_code = list(cw_list)
    current_set = set(current_code)
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Precompute bits for current code
    def to_bits_array(code_list):
        if not code_list:
            return np.zeros((0, N), dtype=np.uint8)
        arr = np.zeros((len(code_list), N), dtype=np.uint8)
        for i, w in enumerate(code_list):
            for j in range(N):
                arr[i, j] = (w >> (N-1-j)) & 1
        return arr
    
    def int_to_bits(w):
        return np.array([(w >> (N-1-j)) & 1 for j in range(N)], dtype=np.uint8)
    
    current_bits = to_bits_array(current_code)
    
    rng = random.Random(42)
    
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute a pool of candidate words near existing codewords
    # These are more likely to be compatible
    
    def generate_candidate(code_list, rng):
        """Generate a candidate word that might be compatible with the code"""
        # Strategy 1: Random word
        if rng.random() < 0.3:
            return rng.randint(0, UNIVERSE_SIZE - 1)
        
        # Strategy 2: Perturb an existing codeword
        if code_list and rng.random() < 0.7:
            base = code_list[rng.randint(0, len(code_list)-1)]
            # Flip 1-3 random bits
            num_flips = rng.randint(1, 3)
            positions = rng.sample(range(N), num_flips)
            w = base
            for pos in positions:
                w ^= (1 << pos)
            return w
        
        # Strategy 3: Random
        return rng.randint(0, UNIVERSE_SIZE - 1)
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best
            current_code = list(best_code)
            current_set = set(current_code)
            current_bits = to_bits_array(current_code)
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.3)
            if max_no_improve > 50000:
                max_no_improve = 50000
            
        # Pick a candidate word
        w = generate_candidate(current_code, rng)
        
        if w in current_set:
            no_improve += 1
            continue
            
        # Check conflicts
        w_bits = int_to_bits(w)
        if len(current_bits) > 0:
            diffs = current_bits ^ w_bits
            weights = diffs.sum(axis=1)
            conflict_indices = np.where(weights < MIN_DIST)[0]
        else:
            conflict_indices = np.array([], dtype=int)
        
        if len(conflict_indices) == 0:
            # Can add
            current_code.append(w)
            current_set.add(w)
            if len(current_bits) == 0:
                current_bits = w_bits.reshape(1, -1)
            else:
                current_bits = np.vstack([current_bits, w_bits])
            no_improve = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)
        else:
            # Try to kick out a conflicting word
            # Pick a random conflict index
            idx = conflict_indices[rng.randint(0, len(conflict_indices)-1)]
            old_word = current_code[idx]
            
            # Try to find a new position for old_word
            # We want a new word w_new such that:
            # 1. dist(w_new, w) >= 6
            # 2. dist(w_new, current_code[k]) >= 6 for all k != idx
            #
            # Strategy: Try a few random candidates for w_new
            success = False
            for _ in range(15):
                w_new = generate_candidate(current_code, rng)
                if w_new in current_set:
                    continue
                if w_new == w:
                    continue
                w_new_bits = int_to_bits(w_new)
                
                # Check dist to w
                if (w_new_bits ^ w_bits).sum() < MIN_DIST:
                    continue
                    
                # Check dist to others
                # Exclude idx
                mask = np.ones(len(current_code), dtype=bool)
                mask[idx] = False
                others_bits = current_bits[mask]
                if len(others_bits) == 0:
                    continue
                diffs_others = others_bits ^ w_new_bits
                weights_others = diffs_others.sum(axis=1)
                if np.any(weights_others < MIN_DIST):
                    continue
                    
                # Success: Replace current_code[idx] with w_new
                current_set.discard(old_word)
                current_set.add(w_new)
                current_code[idx] = w_new
                current_bits[idx] = w_new_bits
                success = True
                break
            
            if not success:
                no_improve += 1
                # Try a swap move to explore the space
                if rng.random() < 0.4 and len(current_code) > 0:
                    swap_idx = rng.randint(0, len(current_code)-1)
                    w_swap = generate_candidate(current_code, rng)
                    if w_swap in current_set:
                        continue
                    if w_swap == current_code[swap_idx]:
                        continue
                    w_swap_bits = int_to_bits(w_swap)
                    
                    mask = np.ones(len(current_code), dtype=bool)
                    mask[swap_idx] = False
                    others_bits = current_bits[mask]
                    if len(others_bits) > 0:
                        diffs_others = others_bits ^ w_swap_bits
                        weights_others = diffs_others.sum(axis=1)
                        if np.any(weights_others < MIN_DIST):
                            continue
                    # Swap successful
                    old_word = current_code[swap_idx]
                    current_set.discard(old_word)
                    current_set.add(w_swap)
                    current_code[swap_idx] = w_swap
                    current_bits[swap_idx] = w_swap_bits
                    no_improve += 1

    # Return best code
    result = best_code
    # Final verification
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST}"
            
    return result
