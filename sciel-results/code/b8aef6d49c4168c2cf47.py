import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        if n > 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback: return all words
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words (vectorized)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
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
    
    # Helper: convert index to digit list
    def idx_to_digits(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        return digits
    
    # Helper: greedy construction
    def greedy_construct(rng, deadline_inner):
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered) and time.time() < deadline_inner:
            # Count uncovered neighbors for each word
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            # Pick candidate: random among max-count
            candidates = np.where(counts == max_count)[0]
            if len(candidates) > 1 and rng:
                best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered[balls[best_idx]] = False
        
        return selected, cover_count, uncovered
    
    # Helper: aggressive pruning
    def prune(selected, cover_count, deadline_inner):
        changed = True
        while changed and time.time() < deadline_inner:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    # Helper: vectorized swap-based local search
    def local_search_swaps(selected, cover_count, deadline_inner, rng):
        if not selected:
            return selected, cover_count
        
        # Pre-allocate masks
        mask_u = np.zeros(total, dtype=bool)
        mask_v = np.zeros(total, dtype=bool)
        
        # Track best
        best_selected = list(selected)
        best_cover = cover_count.copy()
        best_len = len(selected)
        
        # Run swap iterations
        swap_count = 0
        while time.time() < deadline_inner and swap_count < 50000:
            swap_count += 1
            
            # Pick random codeword to potentially replace
            i = rng.randint(0, len(selected) - 1)
            u = selected[i]
            
            # Pick random replacement
            v = rng.randint(0, total - 1)
            if v == u:
                continue
            
            # Check safety using boolean masks
            mask_u.fill(False)
            mask_u[balls[u]] = True
            mask_v.fill(False)
            mask_v[balls[v]] = True
            
            # Risk: words covered by u but not by v
            risk = mask_u & ~mask_v
            
            safe = True
            if np.any(risk):
                if np.min(cover_count[risk]) <= 1:
                    safe = False
            
            if safe:
                # Perform swap
                cover_count[balls[u]] -= 1
                cover_count[balls[v]] += 1
                selected[i] = v
                
                # Try to prune after swap
                pruned = False
                changed = True
                while changed and time.time() < deadline_inner:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            changed = True
                            pruned = True
                        else:
                            new_selected.append(c)
                    selected = new_selected
                
                # Update best if improved
                if len(selected) < best_len:
                    best_len = len(selected)
                    best_selected = list(selected)
                    best_cover = cover_count.copy()
        
        return best_selected, best_cover
    
    # Helper: delete-and-repair move
    def delete_and_repair(selected, cover_count, deadline_inner, rng):
        """Try to remove a codeword; if it creates uncovered words, try to repair with one addition."""
        if not selected:
            return selected, cover_count
        
        best_selected = list(selected)
        best_cover = cover_count.copy()
        best_len = len(selected)
        
        attempts = 0
        while time.time() < deadline_inner and attempts < 10000:
            attempts += 1
            
            # Pick random codeword to try removing
            i = rng.randint(0, len(selected) - 1)
            c = selected[i]
            
            # Check what would be uncovered if we remove c
            # Words only covered by c (cover_count == 1) in c's ball
            only_covered = (cover_count[balls[c]] == 1)
            if not np.any(only_covered):
                # Safe removal - no uncovered words
                cover_count[balls[c]] -= 1
                new_selected = selected[:i] + selected[i+1:]
                selected = new_selected
                best_len = len(selected)
                best_selected = list(selected)
                best_cover = cover_count.copy()
                continue
            
            # Some words would be uncovered. Try to find a single word that covers all of them.
            uncovered_words = np.where(only_covered)[0]
            uncovered_set = set(uncovered_words.tolist())
            
            # Find a word v whose ball covers all uncovered_words
            # For efficiency, check candidates that are neighbors of uncovered words
            candidate_set = set()
            for uw in uncovered_words:
                for nb in balls[uw]:
                    candidate_set.add(int(nb))
            
            found_repair = False
            for v in candidate_set:
                v = int(v)
                # Check if v's ball covers all uncovered_words
                v_ball = set(balls[v].tolist())
                if uncovered_set.issubset(v_ball):
                    # Safe: remove c, add v
                    cover_count[balls[c]] -= 1
                    cover_count[balls[v]] += 1
                    new_selected = selected[:i] + selected[i+1:] + [v]
                    selected = new_selected
                    found_repair = True
                    
                    # Try to prune after this move
                    changed = True
                    while changed and time.time() < deadline_inner:
                        changed = False
                        new_sel = []
                        for cc in selected:
                            if np.min(cover_count[balls[cc]]) > 1:
                                cover_count[balls[cc]] -= 1
                                changed = True
                            else:
                                new_sel.append(cc)
                        selected = new_sel
                    
                    if len(selected) < best_len:
                        best_len = len(selected)
                        best_selected = list(selected)
                        best_cover = cover_count.copy()
                    break
            
            if not found_repair:
                # Try a few random candidates as well
                for _ in range(3):
                    v = rng.randint(0, total - 1)
                    v_ball = set(balls[v].tolist())
                    if uncovered_set.issubset(v_ball):
                        cover_count[balls[c]] -= 1
                        cover_count[balls[v]] += 1
                        new_selected = selected[:i] + selected[i+1:] + [v]
                        selected = new_selected
                        found_repair = True
                        
                        changed = True
                        while changed and time.time() < deadline_inner:
                            changed = False
                            new_sel = []
                            for cc in selected:
                                if np.min(cover_count[balls[cc]]) > 1:
                                    cover_count[balls[cc]] -= 1
                                    changed = True
                                else:
                                    new_sel.append(cc)
                            selected = new_sel
                        
                        if len(selected) < best_len:
                            best_len = len(selected)
                            best_selected = list(selected)
                            best_cover = cover_count.copy()
                        break
        
        return best_selected, best_cover
    
    # Main algorithm
    rng = random.Random(42)
    
    # Phase 1: Initial greedy construction (deterministic for reproducibility)
    selected, cover_count, uncovered = greedy_construct(None, deadline - 60.0)
    
    # Verify coverage
    if np.any(uncovered):
        # Add missing words
        missing = np.where(uncovered)[0]
        for m in missing:
            selected.append(m)
            cover_count[balls[m]] += 1
    
    # Phase 2: Aggressive pruning
    selected, cover_count = prune(selected, cover_count, deadline - 50.0)
    
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_len = len(selected)
    
    # Phase 3: Intensive local search with swaps (vectorized masks)
    if time.time() < deadline - 20.0:
        selected, cover_count = local_search_swaps(selected, cover_count, deadline - 15.0, rng)
        if len(selected) < best_len:
            best_len = len(selected)
            best_selected = list(selected)
            best_cover = cover_count.copy()
    
    # Phase 4: Delete-and-repair moves
    if time.time() < deadline - 10.0:
        selected, cover_count = delete_and_repair(best_selected, best_cover.copy(), deadline - 5.0, rng)
        if len(selected) < best_len:
            best_len = len(selected)
            best_selected = list(selected)
            best_cover = cover_count.copy()
    
    # Phase 5: Final pruning pass
    selected, cover_count = prune(best_selected, best_cover, deadline - 2.0)
    
    # Final verification
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
    
    # Convert to digit lists
    result = [idx_to_digits(w) for w in selected]
    
    return result
