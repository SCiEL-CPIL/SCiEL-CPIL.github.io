"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _START[0]

def construct(q, n, radius):
    _START[0] = time.time()
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** 8  # 6561

    def idx_to_digits(i):
        d = [0] * 8
        for p in range(8):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(8):
            v += d[p] * mul
            mul *= 3
        return v

    BALL_SIZE = 17
    ball_matrix = np.zeros((N, BALL_SIZE), dtype=np.int32)
    for i in range(N):
        d = idx_to_digits(i)
        ball_matrix[i, 0] = i
        idx = 1
        for p in range(8):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    ball_matrix[i, idx] = digits_to_idx(d)
                    idx += 1
                    d[p] = orig

    # Build inverse index: for each word w, which centers cover it
    centers_covering = [None] * N
    for i in range(N):
        for w in ball_matrix[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(N):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx(w))
    seed_set = list(set(seed_set))

    # Greedy fill with vectorized gain computation
    covered = np.zeros(N, dtype=np.int8)
    cover_count = np.zeros(N, dtype=np.int32)
    chosen = list(seed_set)
    for i in seed_set:
        covered[ball_matrix[i]] = 1
        cover_count[ball_matrix[i]] += 1

    def greedy_fill(chosen, cover_count, time_budget=15.0):
        t_start = time.time()
        while time.time() - t_start < time_budget and _time_left() > 25.0:
            gains = BALL_SIZE - cover_count[ball_matrix].sum(axis=1)
            best_i = int(np.argmax(gains))
            if gains[best_i] <= 0:
                break
            chosen.append(best_i)
            cover_count[ball_matrix[best_i]] += 1
        return chosen, cover_count

    chosen, cover_count = greedy_fill(chosen, cover_count)

    # Aggressive trimming of redundant codewords
    def trim(chosen, cover_count, time_budget=10.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            order = list(chosen)
            rng.shuffle(order)
            for i in order:
                if i not in chosen:
                    continue
                b = ball_matrix[i]
                if np.min(cover_count[b]) >= 2:
                    cover_count[b] -= 1
                    chosen.remove(i)
                    improved = True
        return chosen, cover_count

    chosen, cover_count = trim(chosen, cover_count)

    # Local search: targeted swap via inverse index
    def local_search(chosen, cover_count, time_budget=50.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            indices = list(range(len(chosen)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() - t_start > time_budget or _time_left() < 15.0:
                    break
                i = chosen[idx]
                bp_i = ball_matrix[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    chosen.pop(idx)
                    improved = True
                    break
                
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                chosen_set = set(chosen)
                candidates = [c for c in candidate_set if c not in chosen_set]
                if not candidates:
                    continue
                
                # Score: minimize wasted coverage after the swap
                best_swap = None
                best_waste = None
                for j in candidates:
                    if time.time() - t_start > time_budget or _time_left() < 15.0:
                        return chosen, cover_count
                    bp_j = ball_matrix[j]
                    in_i = (cover_count[bp_i] > 0)
                    waste = int(np.sum((cover_count[bp_j] >= 2) & in_i))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                
                if best_swap is not None:
                    j = best_swap
                    cover_count[ball_matrix[i]] -= 1
                    cover_count[ball_matrix[j]] += 1
                    chosen[idx] = j
                    improved = True
                    break
        
        return chosen, cover_count

    chosen, cover_count = local_search(chosen, cover_count, time_budget=50.0)
    chosen, cover_count = trim(chosen, cover_count)

    # Simulated annealing for further improvement
    def improved_local_search(chosen, cover_count, time_budget=30.0):
        t_start = time.time()
        best_set = set(chosen)
        best_size = len(chosen)
        cur_set = set(chosen)
        cur_size = len(cur_set)
        
        T = 5.0
        T_min = 0.1
        cool_rate = 0.999
        
        while time.time() - t_start < time_budget and _time_left() > 10.0:
            # Try removals first (always accept)
            made_progress = False
            removal_order = list(cur_set)
            rng.shuffle(removal_order)
            for i in removal_order:
                if i not in cur_set:
                    continue
                b = ball_matrix[i]
                if np.min(cover_count[b]) >= 2:
                    cover_count[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    made_progress = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
            
            if not made_progress:
                # Try swaps: remove i, add j (1-for-1)
                i = rng.choice(list(cur_set))
                b = ball_matrix[i]
                would_uncover = b[cover_count[b] == 1]
                
                if len(would_uncover) == 0:
                    cover_count[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
                    continue
                
                target_words = would_uncover.tolist()
                candidate_set = None
                for tw in target_words:
                    covering = set(centers_covering[tw]) - cur_set
                    if candidate_set is None:
                        candidate_set = covering
                    else:
                        candidate_set &= covering
                    if not candidate_set:
                        break
                
                if candidate_set:
                    candidates = list(candidate_set)
                    best_j = None
                    best_score = -1
                    for j in candidates:
                        bj = ball_matrix[j]
                        score = int(np.sum(cover_count[bj] >= 2))
                        if score > best_score:
                            best_score = score
                            best_j = j
                    
                    if best_j is not None:
                        bj = ball_matrix[best_j]
                        cover_count[b] -= 1
                        cover_count[bj] += 1
                        cur_set.discard(i)
                        cur_set.add(best_j)
                        if cur_size < best_size:
                            best_size = cur_size
                            best_set = set(cur_set)
                else:
                    # No exact swap found; try accepting a worsening move via SA
                    i = rng.choice(list(cur_set))
                    b = ball_matrix[i]
                    would_uncover = b[cover_count[b] == 1]
                    
                    if len(would_uncover) > 0 and len(would_uncover) <= 3:
                        target_words = would_uncover.tolist()
                        best_j = None
                        best_covered = 0
                        all_candidates = set()
                        for tw in target_words[:2]:
                            all_candidates.update(set(centers_covering[tw]) - cur_set)
                        
                        if all_candidates:
                            sample = rng.sample(list(all_candidates), min(10, len(all_candidates)))
                            for j in sample:
                                bj = ball_matrix[j]
                                covered_count = sum(1 for tw in target_words if tw in bj)
                                if covered_count > best_covered:
                                    best_covered = covered_count
                                    best_j = j
                            
                            if best_j is not None and best_covered >= len(target_words) - 1:
                                bj = ball_matrix[best_j]
                                new_uncovered = 0
                                for tw in target_words:
                                    if tw not in bj:
                                        new_uncovered += 1
                                if new_uncovered < len(target_words):
                                    cover_count[b] -= 1
                                    cover_count[bj] += 1
                                    cur_set.discard(i)
                                    cur_set.add(best_j)
                                    if cur_size < best_size:
                                        best_size = cur_size
                                        best_set = set(cur_set)
                                elif T > T_min:
                                    delta = new_uncovered - len(target_words)
                                    if rng.random() < np.exp(-delta / T):
                                        cover_count[b] -= 1
                                        cover_count[bj] += 1
                                        cur_set.discard(i)
                                        cur_set.add(best_j)
                                        if cur_size < best_size:
                                            best_size = cur_size
                                            best_set = set(cur_set)
            
            T = max(T * cool_rate, T_min)
            
            if T <= T_min and cur_size >= best_size:
                T = 2.0
        
        # Restore best
        cur_set = best_set
        return cur_set, cover_count

    chosen_set, cover_count = improved_local_search(set(chosen), cover_count, time_budget=30.0)

    # Final removals pass with more budget
    chosen, cover_count = trim(list(chosen_set), cover_count, time_budget=10.0)

    # Final verification
    final_covered = np.zeros(N, dtype=np.int8)
    for i in chosen:
        final_covered[ball_matrix[i]] = 1
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen.append(i)
    
    # One more removal pass after final verification
    final_covcount = np.zeros(N, dtype=np.int32)
    for i in chosen:
        final_covcount[ball_matrix[i]] += 1
    
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in list(chosen):
            b = ball_matrix[i]
            if np.min(final_covcount[b]) >= 2:
                final_covcount[b] -= 1
                chosen.remove(i)
                improved = True
    
    chosen = list(set(chosen))
    return [idx_to_digits(i) for i in chosen]
