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
        return np.array([[ (w >> (N-1-j)) & 1 for j in range(N)] for w in code_list], dtype=np.uint8)
    
    current_bits = to_bits_array(current_code)
    
    rng = random.Random(42)
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    # Candidate pool: We can just pick random words, or words near the code.
    # Picking random words is O(1).
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best
            current_code = list(best_code)
            current_set = set(current_code)
            current_bits = to_bits_array(current_code)
            T = 50.0
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5)
            
        # Pick a random word
        w = rng.randint(0, UNIVERSE_SIZE - 1)
        
        if w in current_set:
            no_improve += 1
            continue
            
        # Check conflicts
        w_bits = np.array([(w >> (N-1-j)) & 1 for j in range(N)], dtype=np.uint8)
        diffs = current_bits ^ w_bits
        weights = diffs.sum(axis=1)
        conflict_indices = np.where(weights < MIN_DIST)[0]
        
        if len(conflict_indices) == 0:
            # Can add
            current_code.append(w)
            current_set.add(w)
            current_bits = np.vstack([current_bits, w_bits])
            no_improve = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)
                # Optionally reset no_improve or adjust T
        else:
            # Try to kick out a conflicting word
            # Pick a random conflict index
            idx = conflict_indices[rng.randint(0, len(conflict_indices)-1)]
            
            # Try to find a new position for current_code[idx]
            # We want a new word w_new such that:
            # 1. dist(w_new, w) >= 6
            # 2. dist(w_new, current_code[k]) >= 6 for all k != idx
            #
            # Strategy: Try a few random candidates for w_new
            success = False
            for _ in range(10):
                w_new = rng.randint(0, UNIVERSE_SIZE - 1)
                if w_new in current_set:
                    continue
                if w_new == w:
                    continue
                w_new_bits = np.array([(w_new >> (N-1-j)) & 1 for j in range(N)], dtype=np.uint8)
                
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
                current_code[idx] = w_new
                current_set.discard(current_code[idx]) # Wait, current_code[idx] is already updated. Need to discard old.
                # Let's do this properly
                old_word = current_code[idx]
                current_set.discard(old_word)
                current_set.add(w_new)
                current_code[idx] = w_new
                current_bits[idx] = w_new_bits
                success = True
                break
            
            if not success:
                no_improve += 1
                # Accept with probability exp(-1/T) to allow exploration?
                # Or just reject.
                # Let's try a "swap" move instead if kick-out fails
                # Swap: Replace a random codeword c with a random word w if w doesn't conflict with others (except c)
                # This changes the set but keeps size constant.
                if rng.random() < 0.5:
                    swap_idx = rng.randint(0, len(current_code)-1)
                    w_swap = rng.randint(0, UNIVERSE_SIZE - 1)
                    if w_swap in current_set:
                        continue
                    w_swap_bits = np.array([(w_swap >> (N-1-j)) & 1 for j in range(N)], dtype=np.uint8)
                    
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
                    # This doesn't improve size, so no_improve += 1
                    no_improve += 1

        T = max(T * cooling, T_min)
        
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
