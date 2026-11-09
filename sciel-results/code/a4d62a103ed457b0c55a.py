"""
Combined program for Football Pool Problem K_3(7,1).
Uses bitmask-based greedy set cover for initialization and aggressive local search 
(removal and swap moves) for optimization.
"""
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** 7  # 2187
        
        # Precompute ball masks
        # We map each word (tuple of 7 ints) to an index 0..2186
        # ball_mask[i] is a bitmask where bit j is 1 if word j is within distance 1 of word i
        
        # Generate all words
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbor masks
        neighbor_masks = [0] * N
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
            neighbor_masks[i] = mask
        
        full_mask = (1 << N) - 1
        
        # Helper: Convert list of indices to list of words
        def indices_to_words(indices):
            return [list(all_words[idx]) for idx in indices]

        # Greedy Set Cover using bitmasks
        # Returns a list of indices
        def greedy_set_cover(seed=None):
            if seed is not None:
                random.seed(seed)
            
            covered = 0
            code_indices = []
            
            # To randomize tie-breaking, we can shuffle the order we check indices,
            # but that requires storing the list. 
            # A better randomization: when multiple indices have the same max gain,
            # pick one randomly. 
            # However, for speed, we'll just iterate 0..N-1. 
            # To break ties randomly, we can add a small random noise to the gain?
            # Or simply shuffle the list of candidate indices each step? That's O(N^2) log N.
            # Let's stick to deterministic greedy first, then rely on local search shuffles.
            # Actually, let's use a "randomized greedy" where we pick the best among a random subset?
            # No, full greedy is fast enough for N=2187.
            
            # To allow different greedy paths, we can shuffle the indices list once at start.
            candidates = list(range(N))
            random.shuffle(candidates)
            
            while covered != full_mask:
                best_gain = -1
                best_idx = -1
                
                # Iterate through shuffled candidates
                for idx in candidates:
                    gain = (neighbor_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        # Early exit if we cover everything remaining?
                        # If gain == remaining bits, we can't do better.
                        # But we might want to pick a different one if there are ties?
                        # For now, first max found.
                        
                if best_gain == 0:
                    break # Should not happen if valid, but safety
            
                code_indices.append(best_idx)
                covered |= neighbor_masks[best_idx]
                
            return code_indices

        # Local Search Optimizer
        # Takes a list of indices, returns optimized list of indices
        def optimize(initial_indices):
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # Use a set for O(1) lookup for swaps later, but for removal we just iterate list.
            # For cover_count, we use an array of size N.
            cover_count = [0] * N
            in_code = [False] * N
            
            for idx in current_indices:
                in_code[idx] = True
                mask = neighbor_masks[idx]
                m = mask
                while m:
                    lb = m & -m
                    idx_nb = lb.bit_length() - 1
                    cover_count[idx_nb] += 1
                    m ^= lb

            improved = True
            # Loop until no improvement or time limit
            while improved and time.time() < deadline - 5:
                improved = False
                random.shuffle(current_indices)
                
                # Removal Pass
                i = 0
                while i < len(current_indices):
                    if time.time() > deadline - 5:
                        break
                    
                    idx = current_indices[i]
                    mask = neighbor_masks[idx]
                    
                    # Check if idx is redundant
                    # Redundant if all neighbors have cover_count > 1
                    m = mask
                    redundant = True
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] <= 1:
                            redundant = False
                            break
                        m ^= lb
                    
                    if redundant:
                        # Remove idx
                        m = mask
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            cover_count[idx_nb] -= 1
                            m ^= lb
                        
                        in_code[idx] = False
                        current_indices.pop(i)
                        improved = True
                        # Do not increment i, check next element
                    else:
                        i += 1

                # Swap Pass
                # Try to swap A in S with B not in S if it allows further reduction
                # We do this a few times per loop if size is still large
                if len(current_indices) > 190:
                    # Number of swap attempts
                    swap_attempts = 50
                    for _ in range(swap_attempts):
                        if time.time() > deadline - 5:
                            break
                        if not current_indices:
                            break
                        
                        # Pick random A from current set
                        idx_a = random.choice(current_indices)
                        
                        # Pick random B not in current set
                        # We can pick random indices until we find one not in set
                        idx_b = random.randint(0, N - 1)
                        while in_code[idx_b]:
                            idx_b = random.randint(0, N - 1)
                        
                        mask_a = neighbor_masks[idx_a]
                        mask_b = neighbor_masks[idx_b]
                        
                        # Check validity of swap A->B
                        # Words covered only by A (cover_count == 1 and in mask_a)
                        # must be covered by B.
                        valid_swap = True
                        m = mask_a
                        while m:
                            lb = m & -m
                            idx_nb = lb.bit_length() - 1
                            if cover_count[idx_nb] == 1:
                                # Check if B covers idx_nb
                                if not (mask_b & (1 << idx_nb)):
                                    valid_swap = False
                                    break
                            m ^= lb
                        
                        if valid_swap:
                            # Perform Swap
                            # Update cover counts
                            m_a = mask_a
                            m_b = mask_b
                            
                            # Subtract A
                            while m_a:
                                lb = m_a & -m_a
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] -= 1
                                m_a ^= lb
                                
                            # Add B
                            while m_b:
                                lb = m_b & -m_b
                                idx_nb = lb.bit_length() - 1
                                cover_count[idx_nb] += 1
                                m_b ^= lb
                                
                            # Update structures
                            in_code[idx_a] = False
                            in_code[idx_b] = True
                            current_indices[current_indices.index(idx_a)] = idx_b
                            
                            # After swap, we might be able to remove more.
                            # The outer loop will handle removal in the next iteration.
                            improved = True
                            # Break out of swap loop to let removal run?
                            # Or continue swapping?
                            # Let's break to allow removal pass to clean up.
                            break 

            return current_indices

        # Main Loop: Multi-start
        best_size = float('inf')
        best_indices = []
        
        # Initial best: Tetracode (243 words) as a safe fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_tetracode = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        best_indices = [word_to_idx[c] for c in initial_tetracode]
        best_size = len(best_indices)
        
        restart_count = 0
        while time.time() < deadline - 10:
            # 1. Generate initial solution using Greedy
            # Use a different seed for each restart to vary the greedy path
            greedy_indices = greedy_set_cover(seed=restart_count)
            
            # 2. Optimize
            optimized_indices = optimize(greedy_indices)
            
            # 3. Update best
            if len(optimized_indices) < best_size:
                best_size = len(optimized_indices)
                best_indices = optimized_indices
                
            restart_count += 1
            
            # If we hit a very good score, we can try to break it down further with more aggressive swaps
            # but the optimize function already does swaps.
            
        # Return the best found
        return indices_to_words(best_indices)

    # Generic fallback for other parameters
    # Simple greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(N))
    code_set = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(N):
            if idx in code_set:
                continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1:
            break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
