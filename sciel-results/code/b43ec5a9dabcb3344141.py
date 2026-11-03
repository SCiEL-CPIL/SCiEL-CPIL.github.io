import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover bits for each word
    # cover_bits[i] is an integer where bit j is 1 if word j is within radius of word i
    cover_bits = [0] * total_space
    
    # Build cover bits
    # For each word i, find all words j within distance <= radius
    # Since radius is small (1), we can generate neighbors directly
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i  # self
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    # Create neighbor
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        
        cover_bits[i] = mask
    
    total_mask = (1 << total_space) - 1
    
    def popcount(x):
        return bin(x).count('1')

    def greedy_cover():
        uncovered = total_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 10.0:
                break
                
            best_idx = -1
            best_gain = -1
            
            # Find word with max coverage of uncovered set
            for i in range(total_space):
                if i in code: # Optimization: skip if already in code? 
                    # Actually, we can just check all, or maintain a set. 
                    # With 2187 items, 'in' on list is O(N). 
                    # Let's use a set for code membership in greedy.
                    pass
            
            # To speed up greedy, we don't strictly need to skip if we just pick max gain.
            # But picking a word already in code is useless.
            # Let's just iterate all.
            for i in range(total_space):
                gain = popcount(cover_bits[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
                
            code.append(best_idx)
            uncovered &= ~cover_bits[best_idx]
            
        return code, uncovered

    def local_search_reduce(code, uncovered):
        if not code:
            return code, uncovered
            
        # cover_count[w] = number of codewords covering word w
        cover_count = np.zeros(total_space, dtype=np.int32)
        
        for idx in code:
            # Extract bits from cover_bits[idx] and increment cover_count
            # This is a bit slow if done naively.
            # We can use numpy if we convert cover_bits to numpy arrays?
            # Or just loop. 15 bits set per word. 15 * len(code) operations.
            # len(code) ~ 180. 180 * 15 = 2700 ops. Very fast.
            mask = cover_bits[idx]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
                
            random.shuffle(code)
            n_cw = len(code)
            
            for i in range(n_cw):
                idx = code[i]
                mask = cover_bits[idx]
                
                # Check if removable
                # Find any word j in mask with cover_count[j] == 1
                is_removable = True
                m = mask
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        is_removable = False
                        break
                    m ^= lsb
                
                if is_removable:
                    # Remove idx
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                    
                    code.pop(i)
                    improved = True
                    # Update uncovered if we want to track it, but local search usually just reduces code size
                    # We need to ensure we don't uncover words.
                    # Since it was removable, all words were covered by others, so uncovered doesn't change.
                    break
                else:
                    # Try swap
                    # Find critical words for idx
                    critical_mask = 0
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        if cover_count[bit_idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                    
                    if critical_mask:
                        # Find candidate k not in code that covers critical_mask
                        # (critical_mask & ~cover_bits[k]) == 0
                        # i.e., critical_mask is subset of cover_bits[k]
                        found_swap = False
                        # Randomize candidate order? Or just iterate?
                        # Iterating all 2187 is fast enough.
                        for k in range(total_space):
                            if k in code:
                                continue
                            if (critical_mask & ~cover_bits[k]) == 0:
                                # Swap idx with k
                                # Remove idx
                                m = mask
                                while m:
                                    lsb = m & -m
                                    bit_idx = lsb.bit_length() - 1
                                    cover_count[bit_idx] -= 1
                                    m ^= lsb
                                    
                                # Add k
                                new_mask = cover_bits[k]
                                m = new_mask
                                while m:
                                    lsb = m & -m
                                    bit_idx = lsb.bit_length() - 1
                                    cover_count[bit_idx] += 1
                                    m ^= lsb
                                    
                                code[i] = k
                                improved = True
                                found_swap = True
                                break
                        
                        if found_swap:
                            break
                            
        return code, uncovered

    def simulated_annealing(code, uncovered):
        if not code:
            return code, uncovered
            
        current_code = code[:]
        current_uncovered = uncovered
        
        # Compute cover_count
        cover_count = np.zeros(total_space, dtype=np.int32)
        for idx in current_code:
            m = cover_bits[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                cover_count[bit_idx] += 1
                m ^= lsb
                
        def get_energy(c_uncovered, c_code):
            return len(c_code) + 100 * popcount(c_uncovered)
            
        current_energy = get_energy(current_uncovered, current_code)
        best_code = current_code[:]
        best_uncovered = current_uncovered
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 1.0
        cooling_rate = 0.999
        
        T = T_start
        sa_start = time.time()
        
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue
                    
                # New uncovered = current_uncovered & ~cover_bits[new_idx]
                new_uncovered = current_uncovered & ~cover_bits[new_idx]
                new_code = current_code + [new_idx]
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            elif move_type < 0.6:
                # Remove
                if len(current_code) <= 1:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_pos]
                
                # Find words uniquely covered by idx
                # unique_mask = cover_bits[idx] & (cover_count == 1)
                unique_mask = 0
                m = cover_bits[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                    
                new_uncovered = current_uncovered | unique_mask
                new_code = current_code[:remove_pos] + current_code[remove_pos+1:]
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    m = cover_bits[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            else:
                # Swap
                if not current_code:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)
                
                if new_idx == old_idx:
                    T *= cooling_rate
                    continue
                    
                # 1. Remove old_idx
                # unique_mask for old_idx
                unique_old = 0
                m = cover_bits[old_idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_old |= lsb
                    m ^= lsb
                    
                temp_uncovered = current_uncovered | unique_old
                
                # 2. Add new_idx
                new_uncovered = temp_uncovered & ~cover_bits[new_idx]
                
                new_code = current_code[:]
                new_code[remove_pos] = new_idx
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    # Update cover_count
                    # Remove old
                    m = cover_bits[old_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                        
                    # Add new
                    m = cover_bits[new_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] += 1
                        m ^= lsb
                        
                    current_code = new_code
                    current_uncovered = new_uncovered
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = current_code[:]
                        best_uncovered = current_uncovered
                        
            T *= cooling_rate
            
        # Return best if valid (uncovered == 0)
        if best_uncovered == 0:
            return best_code, best_uncovered
        else:
            # Fallback to current if valid
            if current_uncovered == 0:
                return current_code, current_uncovered
            else:
                return code, uncovered

    # Initial greedy construction
    best_code, best_uncovered = greedy_cover()
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Run restarts if time permits
    while time.time() < deadline - 20.0:
        cand_code, cand_uncovered = greedy_cover()
        cand_code, _ = local_search_reduce(cand_code, cand_uncovered)
        
        # Apply simulated annealing to further reduce
        cand_code, cand_uncovered = simulated_annealing(cand_code, cand_uncovered)
        
        # Only accept if valid and smaller
        if cand_uncovered == 0 and len(cand_code) < len(best_code):
            best_code = cand_code
            best_uncovered = cand_uncovered
    
    # Final polish
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    if time.time() < deadline - 5.0:
        best_code, best_uncovered = simulated_annealing(best_code, best_uncovered)
    best_code, _ = local_search_reduce(best_code, best_uncovered)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
