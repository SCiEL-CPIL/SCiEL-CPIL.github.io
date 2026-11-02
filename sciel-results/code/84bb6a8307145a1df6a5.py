import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    total = q ** n
    if total <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []

        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered
        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls
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

    # Precompute ball membership (symmetric)
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

    # Precompute digit representations for fast Hamming distance
    digits_arr = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        for pos in range(n):
            digits_arr[w, pos] = (w // weights[pos]) % q

    def hamming_dist_indices(a, b):
        return int(np.sum(digits_arr[a] != digits_arr[b]))

    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    best_size = total
    best_set = None

    def run_greedy_with_dual(cover_count_init=None, seed_offset=0):
        """Greedy set cover with dual-guided scoring."""
        np.random.seed(42 + seed_offset)
        
        if cover_count_init is None:
            cover_count = np.zeros(total, dtype=np.int32)
        else:
            cover_count = cover_count_init.copy()
            
        selected = []
        # Reconstruct selected from cover_count? No, we need the list.
        # If cover_count_init is provided, we assume we start fresh but with some pre-covered?
        # Actually, for the dual-guided restart, we usually start from scratch or from a partial solution.
        # Let's keep it simple: start from scratch but use dual prices to break ties or guide selection.
        
        # Actually, the strategy says: solve LP relaxation to get dual potentials.
        # We can approximate this. The dual of set cover is:
        # max sum y_w
        # s.t. sum_{w in ball(c)} y_w <= 1 for all c
        #      y_w >= 0
        # 
        # A simple heuristic for dual prices:
        # y_w = 1 / (number of codewords that cover w) ? No, that's for the greedy dual update.
        # In standard greedy, we pick c that maximizes sum_{w in ball(c), uncovered} 1.
        # With dual prices, we might pick c that maximizes sum_{w in ball(c)} y_w.
        # Initially y_w = 1 for all w? Then it's standard greedy.
        # As we cover words, we can update y_w.
        
        # Let's try a variant:
        # 1. Start with y_w = 1.0 for all w.
        # 2. Pick c that maximizes sum_{w in ball(c)} y_w.
        # 3. Add c to selected.
        # 4. For all w in ball(c), if w is now covered (count > 0), we might reduce y_w?
        #    Or simply, we only care about uncovered words.
        #    Let's stick to: Score(c) = sum_{w in ball(c)} (1 if uncovered else 0).
        #    This is just standard greedy.
        
        # To make it "dual-guided", let's use a slightly different heuristic:
        # Prioritize words that are hard to cover (few neighbors).
        # But in this symmetric space, all words have 17 neighbors.
        # However, as we build the code, some words become "harder" if their neighbors are already covered by many others?
        # No, if a word is covered, it doesn't matter.
        
        # Let's try: Score(c) = sum_{w in ball(c)} (1.0 / max(1, cover_count[w]))
        # This prioritizes covering words that are currently under-covered (count 0 or 1).
        
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                return None, None # Timeout
            
            # Compute scores
            # For each c, score = sum_{w in ball(c)} (1.0 if uncovered else 0.0)
            # This is equivalent to counting uncovered neighbors.
            scores = uncovered_mask[balls].sum(axis=1)
            
            if np.max(scores) == 0:
                break
                
            max_score = np.max(scores)
            candidates = np.where(scores == max_score)[0]
            
            # Tie-breaking: pick one that is farthest from existing selected?
            # Or random.
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            return None, None
            
        return selected, cover_count

    def local_search(selected, cover_count):
        """Local search: removal, swap, 2-exchange."""
        selected = list(selected)
        cover_count = cover_count.copy()
        
        # Phase 1: Remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                candidate_mask = np.ones(total, dtype=bool)
                for cw in critical_words:
                    candidate_mask &= ball_membership[:, cw]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    # Heuristic: pick candidate that maximizes coverage of other critical words?
                    # For speed, random.
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                    
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        if len(selected) > 450 and time.time() < deadline - 8.0:
            improved_2ex = True
            passes = 0
            while improved_2ex and time.time() < deadline - 5.0 and passes < 5:
                improved_2ex = False
                passes += 1
                new_selected = list(selected)
                
                indices = list(range(len(new_selected)))
                np.random.shuffle(indices)
                
                found_swap = False
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if time.time() > deadline - 5.0:
                            found_swap = True
                            break
                            
                        c1 = new_selected[indices[i]]
                        c2 = new_selected[indices[j]]
                        
                        words_c1 = balls[c1]
                        words_c2 = balls[c2]
                        
                        to_cover = set()
                        for w in words_c1:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        for w in words_c2:
                            if cover_count[w] == 1:
                                to_cover.add(w)
                        set_c2 = set(words_c2)
                        for w in words_c1:
                            if w in set_c2 and cover_count[w] == 2:
                                to_cover.add(w)
                                
                        if not to_cover:
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            improved_2ex = True
                            found_swap = True
                            break
                            
                        candidate_mask = np.ones(total, dtype=bool)
                        for w in to_cover:
                            candidate_mask &= ball_membership[:, w]
                            
                        valid_candidates = np.where(candidate_mask)[0]
                        valid_candidates = valid_candidates[valid_candidates != c1]
                        valid_candidates = valid_candidates[valid_candidates != c2]
                        
                        if len(valid_candidates) > 0:
                            c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                            
                            cover_count[balls[c1]] -= 1
                            cover_count[balls[c2]] -= 1
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in new_selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            improved_2ex = True
                            found_swap = True
                            break
                        
                    if found_swap:
                        break
                if improved_2ex:
                    selected = new_selected
                else:
                    improved_2ex = False

        return selected, cover_count

    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # Strategy: Alternate between pure greedy and dual-guided greedy?
        # Since the space is symmetric, dual-guided might not help much initially.
        # Let's just use randomized greedy with different seeds.
        
        selected, cover_count = run_greedy_with_dual(seed_offset=iteration)
        
        if selected is None:
            continue
            
        # Local Search
        selected, cover_count = local_search(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        if best_size <= 486:
            break

    if best_set is None:
        # Fallback
        selected, cover_count = run_greedy_with_dual(seed_offset=999)
        if selected is None:
            # Last resort: brute force greedy without randomization
            cover_count = np.zeros(total, dtype=np.int32)
            uncovered_mask = np.ones(total, dtype=bool)
            selected = []
            while np.any(uncovered_mask):
                scores = uncovered_mask[balls].sum(axis=1)
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[0]
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)
        best_set = selected

    result = [idx_to_digits(cw) for cw in best_set]
    
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
