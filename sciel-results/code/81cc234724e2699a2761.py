import time
import numpy as np
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
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
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
                
    # Build inverse mapping
    word_to_codewords = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            word_to_codewords[w].append(c)
    
    # Precompute degree
    degree = np.zeros(total, dtype=np.int32)
    for c in range(total):
        neighbors = set()
        for w in balls[c]:
            for c2 in word_to_codewords[w]:
                if c2 != c:
                    neighbors.add(c2)
        degree[c] = len(neighbors)
    
    best_size = total
    best_set = None

    # Precompute adjacency: adj[a][b] = True if codeword b is in the ball of codeword a
    # This enables O(1) swap candidate lookup instead of intersecting Python sets.
    adj = np.zeros((total, total), dtype=bool)
    for c in range(total):
        adj[c, balls[c]] = True

    np.random.seed(42)
    random.seed(42)

    while time.time() < deadline - 10.0:
        # Incremental greedy with cand_count
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        cand_count = np.full(total, ball_size, dtype=np.int32)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            
            # Prefer low degree among max-count candidates
            if len(candidates) > 1:
                degs = degree[candidates]
                min_deg = np.min(degs)
                low_deg_candidates = candidates[degs == min_deg]
                best_idx = low_deg_candidates[np.random.randint(len(low_deg_candidates))]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in word_to_codewords[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
            
            if time.time() > deadline - 15.0:
                break
        
        if uncovered_count > 0:
            continue
            
        selected = np.array(selected, dtype=np.int32)
        
        # Vectorized pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            mins = np.min(cover_count[balls[selected]], axis=1)
            removable_mask = mins > 1
            
            if not np.any(removable_mask):
                break
                
            remove_idx = selected[removable_mask]
            selected = selected[~removable_mask]
            cover_count[balls[remove_idx]] -= 1
            changed = True
            
        # Local search with swaps (fast version using adjacency matrix)
        def try_swaps(selected, cover_count, max_iters=300):
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) == 0:
                    break

                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue

                if len(selected) > 1:
                    degs = degree[selected]
                    min_deg = np.min(degs)
                    low_deg = np.where(degs == min_deg)[0]
                    c_idx = low_deg[np.random.randint(len(low_deg))]
                else:
                    c_idx = 0
                c = selected[c_idx]

                c_words = balls[c]
                unique_mask = cover_count[c_words] == 1
                unique_words = c_words[unique_mask]

                if len(unique_words) == 0:
                    continue

                # Fast candidate intersection using adjacency matrix
                # A candidate c' must be in the ball of every unique word of c
                # i.e., adj[c', w] for all w in unique_words
                # adj is [total x total], adj[c', w] = True iff w is in ball of c'
                # So we need: for all w in unique_words, adj[c', w] == True
                # This is: np.all(adj[:, unique_words], axis=1)
                # But that's expensive. Instead use the fact that we only need
                # candidates that are NOT already selected.
                # Use: adj[c', unique_words].all(axis=1) gives a boolean array over all c'
                # Then mask out already-selected codewords.
                # Optimization: pick the unique word with fewest codeword neighbors first
                # to minimize the search space.
                if len(unique_words) > 1:
                    # Pick the word with smallest degree (fewest codewords in its ball)
                    word_degs = degree[unique_words]
                    best_w_idx = np.argmin(word_degs)
                    first_w = unique_words[best_w_idx]
                    # Start with candidates from the most constrained word
                    candidate_mask = adj[:, first_w].copy()  # codewords whose ball contains first_w
                    for w in unique_words:
                        if w == first_w:
                            continue
                        candidate_mask &= adj[:, w]
                else:
                    candidate_mask = adj[:, unique_words[0]].copy()

                # Exclude already-selected codewords
                candidate_mask[selected] = False
                # Exclude c itself
                candidate_mask[c] = False

                if not np.any(candidate_mask):
                    continue

                cand_indices = np.where(candidate_mask)[0]
                cand_degs = degree[cand_indices]
                min_cd = np.min(cand_degs)
                low_cd = cand_indices[cand_degs == min_cd]
                c_prime = low_cd[np.random.randint(len(low_cd))]

                cover_count[balls[c]] -= 1
                cover_count[balls[c_prime]] += 1
                selected[c_idx] = c_prime

                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True

            return selected, cover_count

        selected, cover_count = try_swaps(selected, cover_count)
        
        # 2-for-1 swaps (fast version using adjacency matrix)
        def try_2_for_1(selected, cover_count, max_iters=80):
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if len(selected) < 2:
                    break

                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue

                degs = degree[selected]
                sorted_indices = np.argsort(degs)
                k = max(2, len(sorted_indices) // 5)
                low_deg_indices = sorted_indices[:k]
                idx1, idx2 = np.random.choice(low_deg_indices, 2, replace=False)
                c1 = selected[idx1]
                c2 = selected[idx2]

                words_c1 = balls[c1]
                words_c2 = balls[c2]

                unique_c1 = words_c1[cover_count[words_c1] == 1]
                unique_c2 = words_c2[cover_count[words_c2] == 1]

                # Find shared words with cover_count == 2 using vectorized ops
                # shared words are those in both balls with cover_count == 2
                # Use the fact that a word w is in ball of c1 iff adj[c1, w]
                # and in ball of c2 iff adj[c2, w]
                shared_mask = adj[c1] & adj[c2]  # words in both balls
                shared_words = np.where(shared_mask)[0]
                shared_unique = shared_words[cover_count[shared_words] == 2]

                all_to_cover = np.concatenate([unique_c1, unique_c2, shared_unique])

                if len(all_to_cover) == 0:
                    continue

                # Fast candidate intersection using adjacency matrix
                if len(all_to_cover) > 1:
                    word_degs = degree[all_to_cover]
                    best_w_idx = np.argmin(word_degs)
                    first_w = all_to_cover[best_w_idx]
                    candidate_mask = adj[:, first_w].copy()
                    for w in all_to_cover:
                        if w == first_w:
                            continue
                        candidate_mask &= adj[:, w]
                else:
                    candidate_mask = adj[:, all_to_cover[0]].copy()

                # Exclude c1, c2, and all already-selected
                candidate_mask[selected] = False
                candidate_mask[c1] = False
                candidate_mask[c2] = False

                if not np.any(candidate_mask):
                    continue

                cand_indices = np.where(candidate_mask)[0]
                cand_degs = degree[cand_indices]
                min_cd = np.min(cand_degs)
                low_cd = cand_indices[cand_degs == min_cd]
                c3 = low_cd[np.random.randint(len(low_cd))]

                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1
                cover_count[balls[c3]] += 1

                new_selected = []
                for i, c in enumerate(selected):
                    if i != idx1 and i != idx2:
                        new_selected.append(c)
                new_selected.append(c3)
                selected = np.array(new_selected, dtype=np.int32)

                changed = True
                while changed and time.time() < deadline - 5.0:
                    mins = np.min(cover_count[balls[selected]], axis=1)
                    removable_mask = mins > 1
                    if not np.any(removable_mask):
                        break
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    changed = True

            return selected, cover_count

        selected, cover_count = try_2_for_1(selected, cover_count)

        # Final polish: aggressive iterative removal + swaps in remaining time
        if time.time() < deadline - 3.0:
            polish_deadline = deadline - 3.0
            # Alternate between removals and swaps until no progress or time runs out
            for _ in range(500):
                if time.time() > polish_deadline:
                    break
                # Try removals
                mins = np.min(cover_count[balls[selected]], axis=1)
                removable_mask = mins > 1
                if np.any(removable_mask):
                    remove_idx = selected[removable_mask]
                    selected = selected[~removable_mask]
                    cover_count[balls[remove_idx]] -= 1
                    continue
                # No removals possible, try a few swaps
                if len(selected) > 1:
                    degs = degree[selected]
                    min_deg = np.min(degs)
                    low_deg = np.where(degs == min_deg)[0]
                    c_idx = low_deg[np.random.randint(len(low_deg))]
                    c = selected[c_idx]
                    c_words = balls[c]
                    unique_mask = cover_count[c_words] == 1
                    unique_words = c_words[unique_mask]
                    if len(unique_words) == 0:
                        break
                    # Fast candidate lookup
                    if len(unique_words) > 1:
                        word_degs = degree[unique_words]
                        best_w_idx = np.argmin(word_degs)
                        first_w = unique_words[best_w_idx]
                        candidate_mask = adj[:, first_w].copy()
                        for w in unique_words:
                            if w == first_w:
                                continue
                            candidate_mask &= adj[:, w]
                    else:
                        candidate_mask = adj[:, unique_words[0]].copy()
                    candidate_mask[selected] = False
                    candidate_mask[c] = False
                    if not np.any(candidate_mask):
                        break
                    cand_indices = np.where(candidate_mask)[0]
                    cand_degs = degree[cand_indices]
                    min_cd = np.min(cand_degs)
                    low_cd = cand_indices[cand_degs == min_cd]
                    c_prime = low_cd[np.random.randint(len(low_cd))]
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    selected[c_idx] = c_prime
                    # Try removals again after swap
                    continue
                break

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        best_set = np.arange(total, dtype=np.int32)
        
    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
