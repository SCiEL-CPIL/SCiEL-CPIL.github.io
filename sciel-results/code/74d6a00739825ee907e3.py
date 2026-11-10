<<<<<<< SEARCH
    if best_result is None:
        # Fallback: run one more greedy with max time
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline - 1)
        if not is_valid(best_result):
            # If still invalid (should not happen), return a dummy valid cover?
            # Construct a trivial cover: all k-subsets? That's huge.
            # Better: just return the best effort. But it must be valid.
            # If greedy failed to cover, we have a problem.
            # Let's force a valid cover by adding blocks until valid.
            # This is slow but ensures validity.
            current_cover_mask = 0
            for b in best_result:
                current_cover_mask |= get_block_mask(b)
            
            # Find uncovered t-subsets
            full_mask = (1 << n_t) - 1
            uncovered = full_mask & ~current_cover_mask
            
            # Add blocks to cover uncovered
            while uncovered and time.time() < deadline - 0.1:
                idx = (uncovered & -uncovered).bit_length() - 1
                seed = set(all_t[idx])
                block = list(seed)
                # Grow to k
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block = sorted(block)
                if block not in best_result:
                    best_result.append(block)
                    current_cover_mask |= get_block_mask(block)
                    uncovered &= ~current_cover_mask
            
            # Final check
            if not is_valid(best_result):
                # Extreme fallback: return all blocks? No, too many.
                # Just return what we have. The evaluator will mark invalid.
                # But we tried our best.
                pass
    
    return best_result
=======
    if best_result is None:
        # Fallback: run one more greedy with max time
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline - 2)
        if not is_valid(best_result):
            # Force a valid cover by adding blocks until valid
            current_cover_mask = 0
            for b in best_result:
                current_cover_mask |= get_block_mask(b)
            
            full_mask = (1 << n_t) - 1
            uncovered = full_mask & ~current_cover_mask
            
            while uncovered and time.time() < deadline - 0.1:
                idx = (uncovered & -uncovered).bit_length() - 1
                seed = set(all_t[idx])
                block = list(seed)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block = sorted(block)
                if block not in best_result:
                    best_result.append(block)
                    current_cover_mask |= get_block_mask(block)
                    uncovered &= ~current_cover_mask
    
    return best_result
>>>>>>> REPLACE
