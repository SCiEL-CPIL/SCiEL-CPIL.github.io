"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    Uses a greedy set cover approach followed by local search (removal/swap) to minimize size.
    Optimized for q=3, n=7, radius=1 using bitsets.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Handle trivial cases or specific known structures
    if q == 3 and n == 7 and radius == 1:
        # Generate all 3^7 = 2187 words
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        
        # Map word to index
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute neighbors as bitmasks
        # Each word has 1 + 7*2 = 15 neighbors
        neighbor_masks = [0] * N
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    mask |= (1 << idx)
            neighbor_masks[i] = mask
        
        # Initial Solution: Tetracode construction (243 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        best_size = len(initial_codewords)
        best_mask = 0
        for c in initial_codewords:
            best_mask |= (1 << word_to_idx[c])

        # Local Search Loop
        # We maintain the current set of codewords as a list of indices.
        # We use a "covered_mask" to track which words are covered.
        # Actually, for redundancy check, we need to know if removing a codeword leaves any word uncovered.
        # A word w is covered if there exists c in S such that w in neighbors[c].
        # Equivalently, the union of neighbor_masks for c in S is the full_mask.
        # To check if c is redundant: Is (Union_{c' in S \ {c}} neighbor_masks[c']) == full_mask?
        # This is expensive if done naively.
        
        # Better approach for redundancy:
        # Count how many codewords cover each word.
        # cover_count[w] = number of c in S such that w in neighbors[c].
        # c is redundant if min(cover_count[w] for w in neighbors[c]) > 1.
        
        # Let's implement the cover_count approach with bitset acceleration where possible.
        # Since N=2187, cover_count is an array of 2187 ints.
        
        def optimize(initial_indices):
            current_indices = list(initial_indices)
            random.shuffle(current_indices)
            
            # Initialize cover counts
            cover_count = [0] * N
            for idx in current_indices:
                mask = neighbor_masks[idx]
                # Iterate through set bits in mask
                m = mask
                while m:
                    # Get lowest set bit
                    lb = m & -m
                    idx_nb = lb.bit_length() - 1
                    cover_count[idx_nb] += 1
                    m ^= lb

            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Shuffle to break ties/bias
                random.shuffle(current_indices)
                
                for i in range(len(current_indices)):
                    if time.time() > deadline - 5: break
                    idx = current_indices[i]
                    
                    # Check redundancy
                    mask = neighbor_masks[idx]
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
                        
                        current_indices.pop(i)
                        improved = True
                        if len(current_indices) < best_size:
                            global_best_size = len(current_indices)
                            # We need to update global best mask? 
                            # For now, just track size and the list.
                        # Continue loop without incrementing i as we popped
                        continue

            # Try Swap Moves: 
            # Sometimes we can swap A with B to allow removal of C.
            # Heuristic: For each A in S, pick random B not in S.
            # If S' = S - {A} + {B} is valid (covers all), check if we can remove more from S'.
            # If size(S' after reduction) < size(S after reduction), keep S'.
            
            # This is complex to implement efficiently in the loop.
            # Simpler Swap Heuristic:
            # 1. Pick random A in S.
            # 2. Pick random B not in S.
            # 3. Calculate new cover_count if we swap A->B.
            # 4. If valid, perform swap.
            # 5. Run reduction pass again.
            
            # To save time, we will only do a few rounds of swaps if we haven't reached a very small size.
            if len(current_indices) > 190: # Threshold for aggressive swapping
                swap_rounds = 10
                for _ in range(swap_rounds):
                    if time.time() > deadline - 5: break
                    if not current_indices: break
                    
                    idx_a = random.choice(current_indices)
                    # Pick random B not in S
                    # Approximation: pick random index, check if in S (using a set for S)
                    s_set = set(current_indices)
                    while True:
                        idx_b = random.randint(0, N-1)
                        if idx_b not in s_set:
                            break
                    
                    # Check if swap A->B is valid
                    # New cover_count = old - neighbors[A] + neighbors[B]
                    # We need to ensure no word drops to 0.
                    
                    # Efficient check:
                    # Find words covered ONLY by A.
                    # If any such word is NOT covered by B, the swap is invalid.
                    
                    mask_a = neighbor_masks[idx_a]
                    mask_b = neighbor_masks[idx_b]
                    
                    # Words covered only by A:
                    # cover_count[w] == 1 and w in neighbors[A]
                    # Let's iterate neighbors of A.
                    
                    valid_swap = True
                    m = mask_a
                    while m:
                        lb = m & -m
                        idx_nb = lb.bit_length() - 1
                        if cover_count[idx_nb] == 1:
                            # This word is only covered by A.
                            # Check if B covers it.
                            if not (mask_b & (1 << idx_nb)):
                                valid_swap = False
                                break
                        m ^= lb
                    
                    if valid_swap:
                        # Perform Swap
                        # Update cover_count
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
                            
                        # Replace in list
                        current_indices[current_indices.index(idx_a)] = idx_b
                        s_set.discard(idx_a)
                        s_set.add(idx_b)
                        
                        # Run a quick reduction pass after swap
                        improved_sub = True
                        while improved_sub and time.time() < deadline - 5:
                            improved_sub = False
                            random.shuffle(current_indices)
                            for i in range(len(current_indices)):
                                if time.time() > deadline - 5: break
                                idx = current_indices[i]
                                mask = neighbor_masks[idx]
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
                                    m = mask
                                    while m:
                                        lb = m & -m
                                        idx_nb = lb.bit_length() - 1
                                        cover_count[idx_nb] -= 1
                                        m ^= lb
                                    
                                    current_indices.pop(i)
                                    improved_sub = True
                                    if len(current_indices) < best_size:
                                        global_best_size = len(current_indices)
                                    continue

            return current_indices

        # Multi-start Local Search
        # Start from the 243 set, but with different random shuffles
        # Also try starting from a greedy set cover if time permits
        
        initial_list = [word_to_idx[c] for c in initial_codewords]
        
        # Run multiple restarts
        restarts = 0
        while time.time() < deadline - 10:
            current_solution = optimize(initial_list)
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                # Store best solution
                best_indices = current_solution
            
            # Try a different start: Greedy Set Cover
            if restarts == 0 and time.time() < deadline - 30:
                # Simple Greedy
                uncovered_mask = (1 << N) - 1
                greedy_indices = []
                # Precompute neighbor masks list for faster access? Already have neighbor_masks.
                
                # To speed up greedy, we can estimate the number of newly covered words.
                # Since N is small, we can just scan.
                while uncovered_mask and time.time() < deadline - 15:
                    best_idx = -1
                    best_gain = 0
                    # Sample a subset of candidates if full scan is too slow?
                    # Full scan is 2187 iterations. Each iteration involves bitwise AND and BIT_COUNT.
                    # Python int bit_count is fast (C implementation).
                    for idx in range(N):
                        if idx in greedy_indices: continue # O(1) if set, O(N) if list. Use set.
                        # We don't have a set for greedy_indices yet. Let's use a boolean array.
                        # Actually, just checking `if idx in greedy_set` where greedy_set is a set.
                        pass
                    
                    # Let's implement a simpler greedy with a set for O(1) lookup
                    greedy_set = set(greedy_indices)
                    for idx in range(N):
                        if idx in greedy_set: continue
                        gain = (neighbor_masks[idx] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                            if gain == uncovered_mask.bit_count(): break
                            
                    if best_idx == -1 or best_gain == 0: break
                    greedy_indices.append(best_idx)
                    greedy_set.add(best_idx)
                    uncovered_mask &= ~neighbor_masks[best_idx]
                
                if len(greedy_indices) < best_size:
                    # Optimize this greedy solution
                    optimized_greedy = optimize(greedy_indices)
                    if len(optimized_greedy) < best_size:
                        best_size = len(optimized_greedy)
                        best_indices = optimized_greedy
                
                restarts += 1
                continue

            # Random restart: Perturb the initial list
            # Shuffle initial list and maybe add/remove a few random elements?
            # Just shuffling is enough for the removal heuristic to explore different paths.
            
            # If we haven't improved in a while, try a random restart from scratch?
            # Greedy is expensive. Let's stick to optimizing the tetracode seed with different shuffles.
            # The `optimize` function already shuffles internally.
            
            restarts += 1
            
        # Return the best found
        # best_indices is the list of indices
        result = [list(all_words[idx]) for idx in best_indices]
        return result

    # Generic fallback for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for w in all_words:
        nbs = {word_to_idx[w]}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                nw = list(w)
                nw[pos] = val
                nbs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbs)
    
    uncovered = set(range(len(all_words)))
    code_set = []
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for idx in range(len(all_words)):
            if idx in code_set: continue
            cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1: break
        code_set.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered.discard(nb)
    
    return [list(all_words[idx]) for idx in code_set]
# EVOLVE-BLOCK-END
