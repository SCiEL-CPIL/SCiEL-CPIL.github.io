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
    num_neighbors = 1 + n * (q - 1)  # 17
    balls = np.zeros((total, num_neighbors), dtype=np.int32)
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
    
    # Precompute ball sets for fast intersection
    ball_sets = [set(balls[w].tolist()) for w in range(total)]
    
    best_size = total
    best_set = None
    
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
            best_idx = cands[rng.randint(len(cands))]
            sel.append(best_idx)
            cc[balls[best_idx]] += 1
            unc = (cc == 0)
        
        return sel, cc
    
    def remove_redundant(sel, cc):
        """Remove all redundant codewords using vectorized checks"""
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
        """Try to swap codewords to enable further removals"""
        improved = False
        new_sel = []
        
        for c in sel:
            c_ball = balls[c]
            if np.min(cc[c_ball]) > 1:
                cc[c_ball] -= 1
                improved = True
                continue
            
            # Find critical words (covered only by c)
            critical_mask = (cc[c_ball] == 1)
            critical_words = c_ball[critical_mask]
            
            if len(critical_words) == 0:
                cc[c_ball] -= 1
                improved = True
                continue
            
            # Find candidates that cover all critical words
            # Use set intersection for efficiency
            candidate_set = set(ball_sets[critical_words[0]])
            for cw in critical_words[1:]:
                candidate_set.intersection_update(ball_sets[cw])
            
            # Remove current codeword from candidates
            candidate_set.discard(c)
            
            if not candidate_set:
                new_sel.append(c)
                continue
            
            # Score candidates by how many additional words they cover
            # Use a smooth surrogate: count words with low coverage
            cand_list = list(candidate_set)
            if len(cand_list) > 15:
                cand_list = random.sample(cand_list, 15)
            
            best_prime = c
            best_score = -1
            for cp in cand_list:
                cp_ball = balls[cp]
                # Score: number of words in cp_ball with coverage <= 2
                # (these will benefit from the swap)
                score = np.sum(cc[cp_ball] <= 2)
                if score > best_score:
                    best_score = score
                    best_prime = cp
            
            if best_prime != c:
                cc[c_ball] -= 1
                cc[balls[best_prime]] += 1
                new_sel.append(best_prime)
                improved = True
            else:
                new_sel.append(c)
        
        return new_sel, cc, improved
    
    # Run multiple greedy restarts
    best_greedy_sel = None
    best_greedy_cc = None
    best_greedy_size = total
    
    for restart in range(5):
        sel, cc = greedy_construct(restart * 7919)
        if sel is None:
            break
        if len(sel) < best_greedy_size:
            best_greedy_size = len(sel)
            best_greedy_sel = sel
            best_greedy_cc = cc.copy()
    
    if best_greedy_sel is None:
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
    
    # Main local search loop
    while time.time() < deadline - 3.0:
        # Phase 1: Remove redundant
        selected, cover_count, changed = remove_redundant(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # Phase 2: Try swaps
        if time.time() < deadline - 3.0:
            new_selected, new_cc, improved = try_swap_improve(selected, cover_count)
            if improved:
                selected = new_selected
                cover_count = new_cc
                # After swap, try removal again
                selected, cover_count, changed = remove_redundant(selected, cover_count)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
                    if best_size <= 403:
                        break
            else:
                # No improvement from swap, try random perturbation
                if len(selected) > 100 and random.random() < 0.4:
                    idx = random.randint(len(selected))
                    c = selected[idx]
                    
                    cover_count[balls[c]] -= 1
                    uncovered = np.where(cover_count == 0)[0]
                    
                    if len(uncovered) == 0:
                        selected.pop(idx)
                        if len(selected) < best_size:
                            best_size = len(selected)
                            best_set = selected.copy()
                    else:
                        # Try to find a replacement
                        fixed = False
                        for _ in range(8):
                            if len(uncovered) == 0:
                                break
                            uw = uncovered[random.randint(len(uncovered))]
                            # Find codewords covering uw via ball_sets
                            coverers = [w for w in range(total) if uw in ball_sets[w]]
                            coverers = [c2 for c2 in coverers if c2 != c and c2 not in selected]
                            if coverers:
                                c_new = random.choice(coverers)
                                cover_count[balls[c]] -= 1
                                cover_count[balls[c_new]] += 1
                                selected[idx] = c_new
                                fixed = True
                                break
                        
                        if not fixed:
                            cover_count[balls[c]] += 1  # Revert
    
    # Final high-precision polish pass
    if best_set is not None:
        selected = best_set.copy()
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        # Multiple rounds of aggressive cleanup
        for _ in range(10):
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
