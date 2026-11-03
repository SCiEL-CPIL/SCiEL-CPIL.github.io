import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = 150.0
    
    if n <= 1:
        return "1"
    
    perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    total_perms = len(perms)
    full_mask = (1 << total_perms) - 1
    
    init_window = tuple(range(1, n))
    init_mask = 0
    init_seq = list(init_window)
    
    beam_width = 2000
    beam = [(n - 1, init_window, init_mask, init_seq)]
    
    best_result = None
    best_len = float('inf')
    
    while beam:
        if time.time() - start_time > deadline - 5:
            break
            
        beam.sort(key=lambda x: x[0] + (total_perms - x[2].bit_count()))
        beam = beam[:beam_width]
        
        next_beam = []
        
        for cost, window, mask, seq in beam:
            if mask == full_mask:
                if cost < best_len:
                    best_len = cost
                    best_result = seq
                continue
                
            if cost + (total_perms - mask.bit_count()) >= best_len:
                continue
                
            for sym in range(1, n + 1):
                new_window = window[1:] + (sym,)
                new_seq = seq + [sym]
                new_cost = cost + 1
                
                full_window = window + (sym,)
                new_mask = mask
                
                if full_window in perm_to_idx:
                    idx = perm_to_idx[full_window]
                    bit = 1 << idx
                    if not (mask & bit):
                        new_mask = mask | bit
                        if new_mask == full_mask:
                            if new_cost < best_len:
                                best_len = new_cost
                                best_result = new_seq
                            continue
                
                next_beam.append((new_cost, new_window, new_mask, new_seq))
                
        beam = next_beam
        
    if best_result:
        return "".join(map(str, best_result))
        
    # Fallback
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
