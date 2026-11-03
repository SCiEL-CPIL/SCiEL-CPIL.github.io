import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def calculate_max_overlap(h):
        # C_j = sum_{i=0}^{N-1-j} h[i] * (1 - h[i+j])
        max_c = 0.0
        prefix_sum = np.cumsum(h)
        
        for j in range(N):
            if N-1-j >= 0:
                s_j = prefix_sum[N-1-j]
            else:
                s_j = 0.0
            
            if j == 0:
                r_j = np.sum(h * h)
            else:
                r_j = np.sum(h[:N-j] * h[j:])
            
            c_j = s_j - r_j
            if c_j > max_c:
                max_c = c_j
        return max_c

    def get_blocks(h):
        # Returns list of (start, end, value) for contiguous blocks
        blocks = []
        if N == 0: return blocks
        curr_val = h[0]
        start = 0
        for i in range(1, N):
            if h[i] != curr_val:
                blocks.append((start, i-1, curr_val))
                start = i
                curr_val = h[i]
        blocks.append((start, N-1, curr_val))
        return blocks

    # Initialize
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N//2, replace=False)
    h[ones_indices] = 1.0
    
    best_score = calculate_max_overlap(h)
    best_h = h.copy()
    
    stagnation_counter = 0
    
    while time.time() < deadline:
        current_score = calculate_max_overlap(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        # If stagnating, try a random restart or more aggressive move
        if stagnation_counter > 1000:
             # Random restart
             h = np.zeros(N)
             ones_indices = np.random.choice(N, N//2, replace=False)
             h[ones_indices] = 1.0
             current_score = calculate_max_overlap(h)
             if current_score < best_score:
                 best_score = current_score
                 best_h = h.copy()
             stagnation_counter = 0
             continue

        # Perform a cluster swap
        blocks = get_blocks(h)
        zero_blocks = [b for b in blocks if b[2] == 0]
        one_blocks = [b for b in blocks if b[2] == 1]
        
        if not zero_blocks or not one_blocks:
            break
            
        # Pick random lengths to match
        # To make it efficient, pick a random block from one set, then find a block of same length in the other
        # Or just pick random i, k and try to extend to blocks?
        # Simpler: Pick a random zero block and a random one block. If lengths differ, swap the common length part?
        # Or just pick random start indices i (in 0 block) and k (in 1 block) and swap a random length L segment?
        
        # Let's do: Pick a random zero block (s0, e0, 0) and a random one block (s1, e1, 1).
        # Let L = min(e0-s0+1, e1-s1+1).
        # Swap a random segment of length L from each.
        
        s0, e0, _ = np.random.choice(zero_blocks)
        s1, e1, _ = np.random.choice(one_blocks)
        
        len0 = e0 - s0 + 1
        len1 = e1 - s1 + 1
        L = min(len0, len1)
        
        # Pick random start offset within the blocks
        off0 = np.random.randint(0, len0 - L + 1)
        off1 = np.random.randint(0, len1 - L + 1)
        
        start0 = s0 + off0
        start1 = s1 + off1
        
        # Swap
        h[start0:start0+L] = 1.0
        h[start1:start1+L] = 0.0
        
        new_score = calculate_max_overlap(h)
        
        if new_score < current_score:
            # Keep
            pass
        else:
            # Revert
            h[start0:start0+L] = 0.0
            h[start1:start1+L] = 1.0

    return best_h
