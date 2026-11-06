import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Pre-compute all vectors as uint8 array for fast XOR and popcount
    all_indices = np.arange(TOTAL, dtype=np.uint32)
    all_vectors = (all_indices[:, None] >> np.arange(N, dtype=np.uint32)[None, :]) & 1
    all_vectors = all_vectors.astype(np.uint8)
    
    # Seed: RM(1,4) on 16 bits, extended with leading zero to make 17-bit
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            seed_words.append(word)
    seed_indices = np.array(seed_words, dtype=np.int64)
    
    # Precompute conflict sets using bit-parallel approach
    # For each codeword, we need to know which other words are within distance < MIN_DIST
    # We'll build a conflict bitmap incrementally
    
    # Strategy: Use a "least remaining conflicts" greedy heuristic
    # 1. Start with seed
    # 2. Maintain a set of available candidates
    # 3. For each candidate, count how many conflicts it has with already-selected words
    # 4. Pick the candidate with fewest conflicts (or zero conflicts)
    # 5. Add it, update conflict counts for remaining candidates
    
    best_code = []
    best_size = 0
    
    # Multiple restarts with different initial orderings
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Start with seed
        selected = set(seed_indices.tolist())
        selected_list = list(seed_indices.tolist())
        
        # Build initial conflict tracking
        # For each unselected word, track how many selected words are too close
        # Use a dictionary: word_index -> conflict_count
        conflict_counts = defaultdict(int)
        
        # Initialize conflict counts by checking each non-seed word against seed
        # This is O(TOTAL * seed_size * N) which is manageable
        seed_arr = np.array(list(selected), dtype=np.uint32)
        seed_vecs = all_vectors[seed_arr]
        
        # For all non-seed words, compute min distance to seed
        non_seed_mask = np.ones(TOTAL, dtype=bool)
        non_seed_mask[seed_arr] = False
        non_seed_indices = np.where(non_seed_mask)[0]
        
        # Compute distances from non-seed words to each seed word
        # Batch processing to avoid memory issues
        batch_size = 10000
        for i in range(0, len(non_seed_indices), batch_size):
            batch = non_seed_indices[i:i+batch_size]
            batch_vecs = all_vectors[batch]
            # Compute XOR with each seed vector
            # dists[j, k] = popcount(batch_vecs[j] XOR seed_vecs[k])
            # Use broadcasting: (batch_len, 1, N) XOR (1, seed_len, N)
            xor_matrix = batch_vecs[:, None, :] ^ seed_vecs[None, :, :]
            dists = np.sum(xor_matrix, axis=2)  # (batch_len, seed_len)
            min_dists = np.min(dists, axis=1)  # (batch_len,)
            # Words with min_dist < MIN_DIST have conflicts
            for j, idx in enumerate(batch):
                if min_dists[j] < MIN_DIST:
                    conflict_counts[int(idx)] = 1  # Has at least one conflict
        
        # Now iteratively add words with fewest conflicts
        # We'll use a priority queue approach
        import heapq
        
        # Build initial priority queue: (conflict_count, word_index)
        # Only include words with conflict_count == 0 first, then 1, etc.
        # Actually, let's just pick words with 0 conflicts first
        available = [idx for idx in non_seed_indices if conflict_counts.get(int(idx), 0) == 0]
        
        if not available:
            # No zero-conflict words, try words with 1 conflict
            available = [idx for idx in non_seed_indices if conflict_counts.get(int(idx), 0) <= 1]
        
        # Shuffle available to break ties
        np.random.shuffle(available)
        
        # Greedy selection: pick word, add it, update conflicts for others
        while time.time() < deadline - 2:
            if not available:
                break
                
            # Pick the first available word (we could be smarter here)
            chosen = available.pop(0)
            chosen_int = int(chosen)
            
            if chosen_int in selected:
                continue
                
            selected.add(chosen_int)
            selected_list.append(chosen_int)
            
            # Update conflict counts for remaining available words
            # A word w now has a conflict if dist(w, chosen) < MIN_DIST
            chosen_vec = all_vectors[chosen_int]
            
            new_available = []
            for idx in available:
                idx_int = int(idx)
                if idx_int in selected:
                    continue
                # Check distance to chosen
                dist = np.sum(all_vectors[idx_int] != chosen_vec)
                if dist < MIN_DIST:
                    # This word now conflicts with chosen
                    conflict_counts[idx_int] = conflict_counts.get(idx_int, 0) + 1
                    # If it now has conflicts, remove from available (if we're only keeping 0-conflict)
                    if conflict_counts[idx_int] > 0:
                        continue
                new_available.append(idx)
            
            available = new_available
            
            # If available is empty, try to find more words with 0 conflicts
            if not available:
                # Rescan all non-selected words for 0-conflict candidates
                # This is expensive, so only do it occasionally
                if len(selected_list) % 10 == 0:
                    for idx in range(TOTAL):
                        if idx in selected:
                            continue
                        if conflict_counts.get(idx, 0) == 0:
                            # Verify it's actually 0-conflict
                            is_conflict = False
                            for sel in selected_list:
                                if np.sum(all_vectors[idx] != all_vectors[sel]) < MIN_DIST:
                                    is_conflict = True
                                    break
                            if not is_conflict:
                                available.append(idx)
        
        if len(selected_list) > best_size:
            best_size = len(selected_list)
            best_code = selected_list[:]
    
    return best_code
