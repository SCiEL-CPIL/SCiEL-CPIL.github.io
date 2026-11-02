import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls as indices
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
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
                
    # 1. Greedy Initialization - vectorized
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered = np.ones(total, dtype=bool)
    
    while np.any(uncovered) and time.time() < deadline - 20.0:
        counts = np.sum(uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered = (cov == 0)
    
    # 2. Local Search
    in_code = np.zeros(total, dtype=bool)
    for c in selected:
        in_code[c] = True
        
    def add_code(c):
        cov[balls[c]] += 1
        in_code[c] = True
        selected.append(c)
        
    def remove_code(c):
        cov[balls[c]] -= 1
        in_code[c] = False
        selected.remove(c)
        
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        for c in list(selected):
            if not in_code[c]:
                continue
            if np.min(cov[balls[c]]) > 1:
                remove_code(c)
                improved = True
                
        # Phase 2: Single swap
        for c in list(selected):
            if not in_code[c]:
                continue
            unique_neighbors = balls[c][cov[balls[c]] == 1]
            if len(unique_neighbors) == 0:
                continue
            
            valid_w_set = set(balls[unique_neighbors[0]].tolist())
            for v in unique_neighbors[1:]:
                valid_w_set.intersection_update(balls[v].tolist())
                
            w_list = [w for w in valid_w_set if not in_code[w]]
            if not w_list:
                continue
                
            best_w = -1
            best_score = -1
            for w in w_list:
                score = np.sum(cov[balls[w]] > 1)
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                remove_code(c)
                add_code(best_w)
                improved = True
                break
                
        # Phase 3: Pair swap - replace two codes with two others
        if not improved and len(selected) >= 2 and time.time() < deadline - 10.0:
            # Try swapping pairs (i, j) with (w1, w2)
            # Condition: all words uniquely covered by i or j must be covered by w1 or w2
            # We'll try a limited number of random pairs to save time
            import random
            random.seed(42)  # Deterministic
            n_selected = len(selected)
            if n_selected >= 2:
                # Sample some pairs
                max_pair_tries = 50
                tried = 0
                for _ in range(max_pair_tries):
                    if time.time() > deadline - 10.0:
                        break
                    i = random.randint(0, n_selected - 1)
                    j = random.randint(0, n_selected - 1)
                    if i == j:
                        continue
                    c1 = selected[i]
                    c2 = selected[j]
                    if not in_code[c1] or not in_code[c2]:
                        continue
                    tried += 1
                    
                    # Words uniquely covered by c1 or c2
                    # A word v is uniquely covered by {c1,c2} if cov[v]==2 and v in ball[c1] or ball[c2]
                    # and no other code covers it
                    # Actually: v must be in ball[c1] or ball[c2], and the only codes covering v are c1 and/or c2
                    # So: for v in ball[c1] U ball[c2], if cov[v] == (1 if v in ball[c1] only else 0) + (1 if v in ball[c2] only else 0) + (1 if v in both else 1)
                    # This is complex. Simpler: after removing c1 and c2, find uncovered words, then find w1,w2 that cover them.
                    
                    # Temporarily remove c1 and c2
                    cov[balls[c1]] -= 1
                    cov[balls[c2]] -= 1
                    in_code[c1] = False
                    in_code[c2] = False
                    selected.remove(c1)
                    selected.remove(c2)
                    
                    # Find words now uncovered
                    uncovered_words = np.where(cov == 0)[0]
                    if len(uncovered_words) == 0:
                        # Just re-add c1 and c2 (no improvement possible with pair swap here)
                        add_code(c1)
                        add_code(c2)
                        continue
                    
                    # Need to find w1, w2 not in code such that ball[w1] U ball[w2] covers all uncovered_words
                    # And also, the words that were covered by c1 or c2 but not by others should still be covered
                    # Actually, we just need all words to be covered.
                    
                    # For each uncovered word v, w1 or w2 must be in ball[v]
                    # So w1 and w2 must cover all uncovered_words together.
                    
                    # Strategy: pick w1 to cover as many uncovered_words as possible, then w2 covers the rest
                    # For each candidate w1 not in code:
                    #   covered_by_w1 = set(balls[w1]) ∩ uncovered_words
                    #   remaining = uncovered_words - covered_by_w1
                    #   Need w2 such that ball[w2] covers remaining
                    #   w2 must be in intersection of ball[v] for v in remaining, and not in code
                    
                    # This is expensive. Let's limit: only try if len(uncovered_words) is small
                    if len(uncovered_words) > 10:
                        add_code(c1)
                        add_code(c2)
                        continue
                    
                    # Find candidates for w1
                    # w1 should be in ball[v] for at least one v in uncovered_words
                    # To limit search, consider w1 in union of balls of uncovered_words
                    candidate_w1_set = set()
                    for v in uncovered_words:
                        candidate_w1_set.update(balls[v].tolist())
                    candidate_w1_set = [w for w in candidate_w1_set if not in_code[w]]
                    
                    if not candidate_w1_set:
                        add_code(c1)
                        add_code(c2)
                        continue
                    
                    # Try a limited number of w1 candidates
                    best_pair = None
                    best_pair_score = -1
                    for w1 in candidate_w1_set[:20]:  # Limit to 20
                        covered1 = set(balls[w1].tolist())
                        remaining = [v for v in uncovered_words if v not in covered1]
                        if not remaining:
                            # w1 alone covers all uncovered
                            # Find w2 that maximizes redundancy (any w2 not in code that doesn't break coverage)
                            # Actually we just need to add w1 and some w2. But we removed 2 codes, so we need to add 2.
                            # w2 can be anything not in code that helps with redundancy.
                            # For simplicity, pick w2 that maximizes sum(cov[balls[w2]] > 1)
                            # But we need w2 not in code and w2 != w1
                            # Let's just pick w2 = w1? No, codewords must be distinct.
                            # Pick a random w2 not in code
                            all_w = [w for w in range(total) if not in_code[w] and w != w1]
                            if not all_w:
                                continue
                            w2 = all_w[0]  # Simplistic
                            score = np.sum(cov[balls[w1]] > 1) + np.sum(cov[balls[w2]] > 1)
                            if score > best_pair_score:
                                best_pair_score = score
                                best_pair = (w1, w2)
                        else:
                            # Find w2 in intersection of ball[v] for v in remaining
                            valid_w2_set = set(balls[remaining[0]].tolist())
                            for v in remaining[1:]:
                                valid_w2_set.intersection_update(balls[v].tolist())
                            valid_w2_set = [w for w in valid_w2_set if not in_code[w] and w != w1]
                            if not valid_w2_set:
                                continue
                            for w2 in valid_w2_set[:10]:
                                score = np.sum(cov[balls[w1]] > 1) + np.sum(cov[balls[w2]] > 1)
                                if score > best_pair_score:
                                    best_pair_score = score
                                    best_pair = (w1, w2)
                    
                    if best_pair is not None:
                        w1, w2 = best_pair
                        add_code(w1)
                        add_code(w2)
                        improved = True
                        break
                    else:
                        # Re-add c1 and c2
                        add_code(c1)
                        add_code(c2)
    
    # 3. Final pass: remove redundant again
    for c in list(selected):
        if in_code[c] and np.min(cov[balls[c]]) > 1:
            remove_code(c)
            
    # Verify coverage
    final_cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cov[balls[c]] += 1
        
    if np.any(final_cov == 0):
        missing = np.where(final_cov == 0)[0]
        for m in missing:
            add_code(m)
            final_cov[balls[m]] += 1
            
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
