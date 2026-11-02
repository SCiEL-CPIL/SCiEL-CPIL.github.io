<<<<<<< SEARCH
    def run_greedy(seed):
        """Randomized greedy with bitmask coverage."""
        rng = random.Random(seed)
        coverage = 0
        selected_indices = []
        
        # We need to track which blocks are selected to avoid re-selection
        # Since we don't remove blocks in greedy, we can just keep a set of selected indices
        selected_set = set()
        
        while coverage != full_mask:
            if time.time() > deadline - 5.0:
                return None
            
            best_gain = -1
            best_cands = []
            
            # Find block with max marginal gain
            # Optimization: If we have many blocks, this loop is the bottleneck.
            # 75k blocks * ~300 steps = 22.5M iterations.
            # Each iteration: bit_count(mask & ~coverage).
            # This should be fast enough in Python if we minimize overhead.
            
            for j in range(n_blocks):
                if j in selected_set:
                    continue
                new_cov = block_masks[j] & ~coverage
                gain = new_cov.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [j]
                elif gain == best_gain:
                    best_cands.append(j)
            
            if not best_cands:
                break
                
            b = rng.choice(best_cands)
            selected_indices.append(b)
            selected_set.add(b)
            coverage |= block_masks[b]
            
        return selected_indices
=======
    def run_greedy(seed):
        """Randomized greedy with bitmask coverage using numpy for gain computation."""
        rng = random.Random(seed)
        coverage_words = [0] * n_words  # coverage as list of uint64 words
        selected_indices = []
        selected_set = set()
        
        # Precompute not-coverage for each word
        while True:
            if time.time() > deadline - 5.0:
                return None
            
            # Check if fully covered
            fully_covered = True
            for w in range(n_words):
                if coverage_words[w] != (0xFFFFFFFFFFFFFFFF if w < n_words - 1 else ((1 << (n_t - (n_words - 1) * 64)) - 1)):
                    fully_covered = False
                    break
            if fully_covered:
                break
            
            # Compute gains using numpy
            # For each word, compute (mask_word & ~coverage_word) and popcount
            gains = np.zeros(n_blocks, dtype=np.int32)
            for w in range(n_words):
                not_cov = ~coverage_words[w] & 0xFFFFFFFFFFFFFFFF
                masked = masks_by_word[w] & not_cov
                gains += popcount_u64(masked)
            
            # Mask out already selected blocks
            for j in selected_set:
                gains[j] = -1
            
            best_gain = gains.max()
            if best_gain <= 0:
                break
                
            # Collect candidates with max gain
            max_cands = np.where(gains == best_gain)[0]
            b = int(rng.choice(max_cands))
            
            selected_indices.append(b)
            selected_set.add(b)
            # Update coverage
            for w in range(n_words):
                coverage_words[w] |= (block_masks[b] >> (w * 64)) & 0xFFFFFFFFFFFFFFFF
            
        return selected_indices
>>>>>>> REPLACE

