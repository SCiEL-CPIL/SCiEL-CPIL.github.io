import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate perms of 1..n-1
    perms_list = list(itertools.permutations(range(1, n)))
    # Convert to strings for faster overlap check
    # Map 1..n-1 to chars? 
    # n-1 <= 6. We can map 1->'1', ..., 6->'6'.
    # String length n-1.
    
    str_perms = ["".join(map(str, p)) for p in perms_list]
    
    # Random start
    indices = list(range(len(str_perms)))
    random.shuffle(indices)
    current_indices = indices[:]
    
    def get_overlap_str(s1, s2):
        # Max k such that s1[-k:] == s2[:k]
        # k can be up to len(s1)-1
        max_k = len(s1) - 1
        for k in range(max_k, 0, -1):
            if s1[-k:] == s2[:k]:
                return k
        return 0

    def calc_total_overlap(idxs):
        total = 0
        for i in range(len(idxs) - 1):
            total += get_overlap_str(str_perms[idxs[i]], str_perms[idxs[i+1]])
        return total

    best_overlap = calc_total_overlap(current_indices)
    best_indices = current_indices[:]
    
    N = len(current_indices)
    
    # Local Search
    # Use Or-opt with full recalc, but limit iterations based on time.
    # Or use 2-opt on short segments.
    
    while time.time() - start_time < deadline - 2.0:
        # Try Or-opt
        seg_len = random.choice([1, 2, 3])
        if N < seg_len:
            continue
        i = random.randint(0, N - seg_len)
        j = random.randint(0, N - seg_len)
        if i == j:
            continue
            
        # Remove and insert
        new_indices = current_indices[:]
        seg = new_indices[i:i+seg_len]
        del new_indices[i:i+seg_len]
        
        # Adjust j
        if i < j:
            j = j - seg_len
            
        new_indices[j:j] = seg
        
        # Calc overlap
        new_overlap = calc_total_overlap(new_indices)
        if new_overlap > best_overlap:
            best_overlap = new_overlap
            best_indices = new_indices[:]
            current_indices = new_indices[:]
        elif new_overlap > calc_total_overlap(current_indices):
            # Accept if better than current (Hill Climbing)
            # But we need current_overlap to avoid recalc
            pass
            
    # We need to track current_overlap to avoid recalc in the 'elif'
    # Let's refactor to track current_overlap
    
    # Restart with proper tracking
    current_indices = list(range(len(str_perms)))
    random.shuffle(current_indices)
    current_overlap = calc_total_overlap(current_indices)
    best_overlap = current_overlap
    best_indices = current_indices[:]
    
    while time.time() - start_time < deadline - 2.0:
        seg_len = random.choice([1, 2, 3])
        if N < seg_len: continue
        i = random.randint(0, N - seg_len)
        j = random.randint(0, N - seg_len)
        if i == j: continue
        
        new_indices = current_indices[:]
        seg = new_indices[i:i+seg_len]
        del new_indices[i:i+seg_len]
        if i < j:
            j = j - seg_len
        new_indices[j:j] = seg
        
        new_overlap = calc_total_overlap(new_indices)
        
        if new_overlap > current_overlap:
            current_indices = new_indices
            current_overlap = new_overlap
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_indices = current_indices[:]
                
    # Reconstruct
    word = []
    if not best_indices:
        return ""
        
    p_str = str_perms[best_indices[0]]
    block = p_str + str(n) + p_str
    word.append(block)
    
    for k in range(1, len(best_indices)):
        p_prev_str = str_perms[best_indices[k-1]]
        p_curr_str = str_perms[best_indices[k]]
        o = get_overlap_str(p_prev_str, p_curr_str)
        
        block = p_curr_str + str(n) + p_curr_str
        word.append(block[o:])
        
    return "".join(word)
