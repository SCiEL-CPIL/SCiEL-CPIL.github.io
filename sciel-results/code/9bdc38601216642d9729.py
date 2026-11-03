import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    total_space = q ** n
    
    # Precompute word to index and index to word
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
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i  # self
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
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
            mask = cover_bits[idx]
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
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                    
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Try swap
                    critical_mask = 0
                    m = mask
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        if cover_count[bit_idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                    
                    if critical_mask:
                        found_swap = False
                        for k in range(total_space):
                            if k in code:
                                continue
                            if (critical_mask & ~cover_bits[k]) == 0:
                                # Swap idx with k
                                m = mask
                                while m:
                                    lsb = m & -m
                                    bit_idx = lsb.bit_length() - 1
                                    cover_count[bit_idx] -= 1
                                    m ^= lsb
                                    
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
        
        while T > T_end:
            if time.time() > deadline - 2.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.3:
                new_idx = random.randint(0, total_space - 1)
                if new_idx in current_code:
                    T *= cooling_rate
                    continue
                    
                new_uncovered = current_uncovered & ~cover_bits[new_idx]
                new_code = current_code + [new_idx]
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
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
                if len(current_code) <= 1:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                idx = current_code[remove_pos]
                
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
                if not current_code:
                    T *= cooling_rate
                    continue
                    
                remove_pos = random.randint(0, len(current_code) - 1)
                old_idx = current_code[remove_pos]
                new_idx = random.randint(0, total_space - 1)
                
                if new_idx == old_idx:
                    T *= cooling_rate
                    continue
                
                unique_old = 0
                m = cover_bits[old_idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if cover_count[bit_idx] == 1:
                        unique_old |= lsb
                    m ^= lsb
                    
                temp_uncovered = current_uncovered | unique_old
                new_uncovered = temp_uncovered & ~cover_bits[new_idx]
                
                new_code = current_code[:]
                new_code[remove_pos] = new_idx
                new_energy = get_energy(new_uncovered, new_code)
                
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    m = cover_bits[old_idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        cover_count[bit_idx] -= 1
                        m ^= lsb
                        
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
            
        if best_uncovered == 0:
            return best_code, best_uncovered
        else:
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
        
        cand_code, cand_uncovered = simulated_annealing(cand_code, cand_uncovered)
        
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
