import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    all_indices = np.arange(total, dtype=np.int32)
    
    # Precompute ball membership as a boolean matrix for fast subset checks
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True
    
    best_size = total
    best_set = None
    
    # Improved greedy construction with multiple restarts
    def greedy_construct(seed_offset=0):
        rng = np.random.RandomState(seed_offset)
        sel = []
        cc = np.zeros(total, dtype=np.int32)
        unc = np.ones(total, dtype=bool)
        
        while np.any(unc):
            if time.time() > deadline - 15.0:
                return None, None
            counts = unc[balls].sum(axis=1)
            max_c = np.max(counts)
            if max_c == 0:
                break
            cands = np.where(counts == max_c)[0]
            # Random tie-breaking for diversity
            best_idx = cands[rng.randint(len(cands))]
            sel.append(best_idx)
            cc[balls[best_idx]] += 1
            unc = (cc == 0)
        
        return sel, cc
    
    # Try a few greedy starts and keep the best
    best_greedy_sel = None
    best_greedy_cc = None
    best_greedy_size = total
    
    for restart in range(3):
        sel, cc = greedy_construct(restart * 1000)
        if sel is None:
            break
        if len(sel) < best_greedy_size:
            best_greedy_size = len(sel)
            best_greedy_sel = sel
            best_greedy_cc = cc.copy()
    
    if best_greedy_sel is None:
        # Fallback to deterministic greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_greedy_sel = selected
        best_greedy_cc = cover_count.copy()
    
    selected = best_greedy_sel.copy()
    cover_count = best_greedy_cc.copy()
    best_size = len(selected)
    best_set = selected.copy()
    
    # Improved local search: focus on effective swaps and removals
    # Use a more aggressive cleanup and swap strategy
    
    def try_remove_redundant(sel, cc):
        """Remove all redundant codewords (those whose removal doesn't uncover any word)"""
        new_sel = []
        changed = False
        for c in sel:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                changed = True
            else:
                new_sel.append(c)
        return new_sel, cc, changed
    
    def try_swap_improve(sel, cc):
        """Try to swap codewords to reduce size or maintain coverage with fewer words"""
        improved = False
        new_sel = []
        
        for c in sel:
            # Check if c is redundant
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                improved = True
                continue
            
            # Find critical words (covered only by c)
            critical_mask = (cc[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            
            if len(critical_words) == 0:
                # c covers no critical words, can remove
                cc[balls[c]] -= 1
                improved = True
                continue
            
            # Find candidates that cover all critical words
            candidate_mask = np.ones(total, dtype=bool)
            for cw in critical_words:
                candidate_mask &= ball_membership[:, cw]
            
            valid_candidates = np.where(candidate_mask)[0]
            valid_candidates = valid_candidates[valid_candidates != c]
            
            if len(valid_candidates) > 0:
                # Prefer candidates that cover more other words (better swap)
                # Score by how many additional words they cover
                scores = np.zeros(len(valid_candidates), dtype=np.int32)
                for i, cp in enumerate(valid_candidates):
                    # Count how many words cp covers that are currently covered exactly once
                    # (excluding critical words which are already required)
                    extra_cover = np.sum(cc[balls[cp]] == 1) - len(critical_words)
                    scores[i] = extra_cover
                
                best_i = np.argmax(scores)
                c_prime = valid_candidates[best_i]
                
                # Perform swap
                cc[balls[c]] -= 1
                cc[balls[c_prime]] += 1
                new_sel.append(c_prime)
                improved = True
            else:
                new_sel.append(c)
        
        return new_sel, cc, improved
    
    # Main local search loop
    while time.time() < deadline - 3.0:
        # Phase 1: Remove redundant
        selected, cover_count, changed = try_remove_redundant(selected, cover_count)
        
        if not changed and len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Phase 2: Try swaps
        if time.time() < deadline - 3.0:
            new_selected, new_cc, improved = try_swap_improve(selected, cover_count)
            if improved:
                selected = new_selected
                cover_count = new_cc
                # After swap, try removal again
                selected, cover_count, changed = try_remove_redundant(selected, cover_count)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
                    if best_size <= 403:
                        break
            else:
                # No improvement from swap, try a random perturbation
                if len(selected) > 100 and np.random.random() < 0.3:
                    # Randomly remove a codeword and try to fix
                    idx = np.random.randint(len(selected))
                    c = selected[idx]
                    
                    # Check if removal causes uncovered words
                    cover_count[balls[c]] -= 1
                    uncovered = np.where(cover_count == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Safe removal
                        selected.pop(idx)
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_set = selected.copy()
                    else:
                        # Need to fix: find codewords that cover uncovered words
                        cover_count[balls[c]] += 1  # Revert
                        # Try to find a replacement
                        fixed = False
                        for _ in range(5):
                            # Pick a random uncovered word and find a codeword covering it
                            if len(uncovered) == 0:
                                break
                            uw = uncovered[np.random.randint(len(uncovered))]
                            # Find codewords covering uw
                            coverers = np.where(ball_membership[:, uw])[0]
                            coverers = coverers[coverers != c]
                            if len(coverers) > 0:
                                c_new = coverers[np.random.randint(len(coverers))]
                                if c_new not in selected:
                                    # Try swapping c with c_new
                                    cover_count[balls[c]] -= 1
                                    cover_count[balls[c_new]] += 1
                                    selected[idx] = c_new
                                    fixed = True
                                    break
                        
                        if not fixed:
                            # Perturbation failed, try another approach
                            pass
        
        if best_size <= 403:
            break

    # Final polish: aggressive cleanup on best_set
    if best_set is not None:
        selected = best_set.copy()
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Multiple rounds of removal and swap
        for _ in range(5):
            if time.time() > deadline - 1.0:
                break
            
            # Remove redundant
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected
            
            # Try swaps
            if time.time() < deadline - 1.0:
                new_selected, cover_count, improved = try_swap_improve(selected, cover_count)
                if improved:
                    selected = new_selected
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    # Convert best_set to output format
    if best_set is None:
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
