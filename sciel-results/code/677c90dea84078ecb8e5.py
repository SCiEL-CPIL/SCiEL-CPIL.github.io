import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill (evaluator says 120s in contract, but directive says 150s wall-clock. Using 140 to be safe for both interpretations if 120 is strict, but 150 is the hard kill. Let's use 115 to be safe for the 120s contract mentioned in the text, wait. "The evaluator KILLS the program after 150s wall-clock." But the contract says "Each evaluation must finish within 120 s." This is contradictory. Usually, the contract is the hard limit for scoring, and the kill is the hard limit for execution. If I run for 120s, I might be killed at 120 by the evaluator logic, or I might be fine until 150. To be safe, I will target finishing by 115s.)
    
    # Re-reading: "Each evaluation must finish within 120 s." and "evaluator KILLS the program after 150s wall-clock". 
    # If I exceed 120s, do I score 0? "an invalid result, an exception or a timeout scores 0". 
    # So I must return within 120s. I will set deadline to 115s.
    deadline = 115.0

    n = 256
    block_size = 8
    num_blocks = n // block_size  # 32 blocks

    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Vectorized mono count
    def count_mono_vec(c):
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))

    # Initial Coloring: Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    current_score = count_mono_vec(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Block-based Local Search
    # Neighborhood: 
    # 1. Swap two blocks
    # 2. Reverse a single block
    # 3. Rotate a single block (cyclic shift)
    
    # To make this faster, we can precompute the "block indices" for each AP?
    # No, an AP can span multiple blocks. 
    # Since M is ~10k-20k, and we do this in a loop, we need to be careful.
    # Let's estimate M.
    # d=1: 253 APs. d=85: 1 AP. Sum_{d=1}^{85} (256 - 3d) = 85*256 - 3*(85*86/2) = 21760 - 3*3655 = 21760 - 10965 = 10795.
    # So M ~ 10,800.
    # A single full evaluation takes ~10k vector ops. Very fast.
    # We can afford many iterations.
    
    # Strategy: 
    # Iterate through random pairs of blocks or single blocks.
    # Apply move, evaluate, accept if better (greedy) or with SA probability.
    # Since block moves change 8 bits, the delta can be larger.
    
    # Let's use a simple Hill Climbing with random restarts if stuck, or SA.
    # Given the time limit, a robust SA with block moves is good.
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Pre-compute block boundaries
    # block_id for each index: i // block_size
    
    # Helper to get block start/end
    def get_block_range(b):
        start = b * block_size
        end = start + block_size
        return start, end

    # For speed, let's precompute the indices for each block
    block_indices = [np.arange(b*block_size, (b+1)*block_size) for b in range(num_blocks)]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose a random move
        move_type = np.random.randint(3)
        
        old_colors = colors.copy()
        
        if move_type == 0:
            # Swap two distinct blocks
            b1, b2 = np.random.choice(num_blocks, 2, replace=False)
            # Swap contents
            colors[block_indices[b1]], colors[block_indices[b2]] = colors[block_indices[b2]].copy(), colors[block_indices[b1]].copy()
            
        elif move_type == 1:
            # Reverse one block
            b1 = np.random.randint(num_blocks)
            idx = block_indices[b1]
            colors[idx] = colors[idx][::-1].copy()
            
        else:
            # Rotate one block by 1
            b1 = np.random.randint(num_blocks)
            idx = block_indices[b1]
            colors[idx] = np.roll(colors[idx], 1)
            
        new_score = count_mono_vec(colors)
        delta = new_score - current_score
        
        if delta < 0:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
        else:
            # Acceptance probability
            if delta == 0:
                if np.random.random() < 0.5:
                    current_score = new_score
            else:
                r = np.random.random()
                if r < np.exp(-delta / T):
                    current_score = new_score
                else:
                    # Reject move, revert
                    colors = old_colors
            
        T *= cooling_rate
        
        # Occasionally, if we are stuck, do a random restart of the tail?
        # Or just keep going. The block moves are quite powerful.
        
    return [int(c) for c in best_colors]
