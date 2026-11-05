import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n  # 2187
    shape = (q,) * n
    
    # Generate all words as numpy array (2187, 7)
    words_np = np.array(np.unravel_index(np.arange(total_words), shape)).T
    
    # Precompute ball masks
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        w = words_np[i]
        ball_masks[i, i] = True
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                idx = int(np.ravel_multi_index(new_w, shape))
                ball_masks[i, idx] = True
    
    # Greedy Set Cover
    def greedy_set_cover(perturb=0.0, seed=None, time_budget=20.0):
        uncovered = np.ones(total_words, dtype=bool)
        selected = []
        rng = np.random.default_rng(seed) if seed is not None else None
        deadline = start_time + time_budget
        
        while uncovered.any():
            if time.time() > deadline:
                break
            gains = (ball_masks & uncovered[np.newaxis, :]).sum(axis=1).astype(float)
            if perturb > 0 and rng is not None:
                gains += rng.random(total_words) * perturb
            best_idx = int(np.argmax(gains))
            if gains[best_idx] < 0.01:
                break
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return selected
    
    # Optimized Local Search using coverage counts
    def local_search(selected_indices, time_budget=40.0):
        selected_set = set(selected_indices)
        # Initialize coverage counts
        cover_counts = np.zeros(total_words, dtype=int)
        for idx in selected_set:
            cover_counts[ball_masks[idx]] += 1
        
        deadline = start_time + time_budget
        improved = True
        
        def try_remove_redundant():
            """Remove any redundant codewords. Returns True if any were removed."""
            nonlocal cover_counts, selected_set
            removed_any = False
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                for idx in list(selected_set):
                    if time.time() > deadline:
                        break
                    # Tentatively remove
                    cover_counts[ball_masks[idx]] -= 1
                    if not np.any(cover_counts == 0):
                        # Safe to remove
                        selected_set.remove(idx)
                        removed_any = True
                        changed = True
                    else:
                        # Restore
                        cover_counts[ball_masks[idx]] += 1
            return removed_any
        
        # Phase 1: Remove redundant codewords
        while (time.time() < deadline):
            if not try_remove_redundant():
                break
        
        # Phase 2: 1-for-1 swap with fixed logic
        while (time.time() < deadline - 5):
            improved = False
            sel_list = list(selected_set)
            for idx1 in sel_list:
                if time.time() > deadline - 5:
                    break
                if idx1 not in selected_set:
                    continue
                # Remove idx1
                cover_counts[ball_masks[idx1]] -= 1
                uncovered_mask = (cover_counts == 0)
                
                if not np.any(uncovered_mask):
                    # idx1 was redundant, remove it
                    selected_set.remove(idx1)
                    improved = True
                else:
                    # Find best replacement: maximize new coverage
                    gains = (ball_masks & uncovered_mask[np.newaxis, :]).sum(axis=1)
                    # Exclude current set from candidates
                    for k in selected_set:
                        gains[k] = -1
                    best_k = int(np.argmax(gains))
                    
                    if gains[best_k] > 0 and best_k not in selected_set:
                        # Apply swap: remove idx1, add best_k
                        selected_set.remove(idx1)
                        selected_set.add(best_k)
                        cover_counts[ball_masks[best_k]] += 1
                        # Now try to remove redundant words
                        before_size = len(selected_set)
                        try_remove_redundant()
                        if len(selected_set) < before_size:
                            improved = True
                        # Even without further removals, the swap maintains coverage
                        # so we accept it (size stays same, but may enable future swaps)
                        # Actually, only accept if size decreased or it's a neutral swap
                        # For now, accept all valid swaps
                    else:
                        # Restore idx1
                        cover_counts[ball_masks[idx1]] += 1
            
            if not improved:
                break
        
        # Phase 3: 2-for-2 swap - try replacing two codewords with two better ones
        while (time.time() < deadline - 3):
            improved = False
            sel_list = list(selected_set)
            if len(sel_list) < 2:
                break
            # Try a limited number of pairs
            for i in range(min(len(sel_list), 20)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, min(len(sel_list), 20)):
                    if time.time() > deadline - 3:
                        break
                    idx1, idx2 = sel_list[i], sel_list[j]
                    if idx1 not in selected_set or idx2 not in selected_set:
                        continue
                    # Remove both
                    cover_counts[ball_masks[idx1]] -= 1
                    cover_counts[ball_masks[idx2]] -= 1
                    uncovered_mask = (cover_counts == 0)
                    
                    if not np.any(uncovered_mask):
                        # Both were redundant
                        selected_set.remove(idx1)
                        selected_set.remove(idx2)
                        improved = True
                        continue
                    
                    # Find best two replacements
                    gains = (ball_masks & uncovered_mask[np.newaxis, :]).sum(axis=1)
                    for k in selected_set:
                        gains[k] = -1
                    
                    # Get top candidates
                    top_candidates = np.argsort(gains)[::-1][:10]
                    
                    # Try pairs of top candidates
                    found_swap = False
                    for a in range(min(5, len(top_candidates))):
                        for b in range(a + 1, min(5, len(top_candidates))):
                            k1, k2 = int(top_candidates[a]), int(top_candidates[b])
                            if k1 in selected_set or k2 in selected_set:
                                continue
                            # Check if k1 and k2 together cover all uncovered
                            combined_cover = ball_masks[k1] | ball_masks[k2]
                            if (combined_cover & uncovered_mask).all():
                                # Apply swap
                                selected_set.add(k1)
                                selected_set.add(k2)
                                cover_counts[ball_masks[k1]] += 1
                                cover_counts[ball_masks[k2]] += 1
                                selected_set.remove(idx1)
                                selected_set.remove(idx2)
                                # Try to remove more redundancies
                                try_remove_redundant()
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    
                    if not found_swap:
                        # Restore
                        cover_counts[ball_masks[idx1]] += 1
                        cover_counts[ball_masks[idx2]] += 1
            
            if not improved:
                break
        
        # Final polish: multiple passes of removals and swaps
        for _ in range(3):
            if time.time() > deadline:
                break
            try_remove_redundant()
            # Quick swap pass
            sel_list = list(selected_set)
            for idx1 in sel_list:
                if time.time() > deadline:
                    break
                if idx1 not in selected_set:
                    continue
                cover_counts[ball_masks[idx1]] -= 1
                uncovered_mask = (cover_counts == 0)
                if not np.any(uncovered_mask):
                    selected_set.remove(idx1)
                else:
                    gains = (ball_masks & uncovered_mask[np.newaxis, :]).sum(axis=1)
                    for k in selected_set:
                        gains[k] = -1
                    best_k = int(np.argmax(gains))
                    if gains[best_k] > 0 and best_k not in selected_set:
                        selected_set.remove(idx1)
                        selected_set.add(best_k)
                        cover_counts[ball_masks[best_k]] += 1
                    else:
                        cover_counts[ball_masks[idx1]] += 1
        
        return selected_set

    best_result = None
    best_size = float('inf')
    
    # Run multiple restarts with adaptive time budget
    num_restarts = 8
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        remaining = time_limit - 20 - elapsed
        if remaining < 10:
            break
        
        # Allocate time: greedy gets 20-30% of remaining, local search gets the rest
        greedy_budget = max(5.0, remaining * 0.25)
        ls_budget = max(15.0, remaining - greedy_budget - 5.0)
        
        if restart == 0:
            selected = greedy_set_cover(perturb=0.0, seed=None, time_budget=greedy_budget)
        elif restart == 1:
            selected = greedy_set_cover(perturb=0.2, seed=42, time_budget=greedy_budget)
        elif restart == 2:
            selected = greedy_set_cover(perturb=0.5, seed=123, time_budget=greedy_budget)
        elif restart == 3:
            selected = greedy_set_cover(perturb=0.8, seed=456, time_budget=greedy_budget)
        elif restart == 4:
            selected = greedy_set_cover(perturb=1.0, seed=789, time_budget=greedy_budget)
        elif restart == 5:
            selected = greedy_set_cover(perturb=0.3, seed=321, time_budget=greedy_budget)
        elif restart == 6:
            selected = greedy_set_cover(perturb=0.6, seed=654, time_budget=greedy_budget)
        else:
            selected = greedy_set_cover(perturb=0.9, seed=restart * 17 + 3, time_budget=greedy_budget)
        
        result_set = local_search(selected, time_budget=ls_budget)
        
        # Validate result
        if result_set:
            # Check coverage
            cov = np.zeros(total_words, dtype=bool)
            for idx in result_set:
                cov |= ball_masks[idx]
            if cov.all():
                if len(result_set) < best_size:
                    best_size = len(result_set)
                    best_result = result_set
                    if best_size <= 186:
                        break
    
    if best_result is None:
        # Fallback: Return a valid, though potentially large, code
        # Simple grid: cover all words
        best_result = set(range(total_words))
    
    # Final high-precision polish: try to remove any remaining redundancies
    # and do one more round of swaps on the best result
    if best_result and time.time() < time_limit - 5:
        selected_set = set(best_result)
        cover_counts = np.zeros(total_words, dtype=int)
        for idx in selected_set:
            cover_counts[ball_masks[idx]] += 1
        
        polish_deadline = time_limit - 5
        
        # Remove redundancies
        changed = True
        while changed and time.time() < polish_deadline:
            changed = False
            for idx in list(selected_set):
                if time.time() > polish_deadline:
                    break
                cover_counts[ball_masks[idx]] -= 1
                if not np.any(cover_counts == 0):
                    selected_set.remove(idx)
                    changed = True
                else:
                    cover_counts[ball_masks[idx]] += 1
        
        # Quick swap pass
        sel_list = list(selected_set)
        for idx1 in sel_list:
            if time.time() > polish_deadline:
                break
            if idx1 not in selected_set:
                continue
            cover_counts[ball_masks[idx1]] -= 1
            uncovered_mask = (cover_counts == 0)
            if not np.any(uncovered_mask):
                selected_set.remove(idx1)
            else:
                gains = (ball_masks & uncovered_mask[np.newaxis, :]).sum(axis=1)
                for k in selected_set:
                    gains[k] = -1
                best_k = int(np.argmax(gains))
                if gains[best_k] > 0 and best_k not in selected_set:
                    selected_set.remove(idx1)
                    selected_set.add(best_k)
                    cover_counts[ball_masks[best_k]] += 1
                else:
                    cover_counts[ball_masks[idx1]] += 1
        
        # Verify and update
        cov = np.zeros(total_words, dtype=bool)
        for idx in selected_set:
            cov |= ball_masks[idx]
        if cov.all() and len(selected_set) < best_size:
            best_result = selected_set
            best_size = len(best_result)
    
    # Convert to list of lists of ints
    return [[int(x) for x in words_np[i]] for i in best_result]
