"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** 8  # 6561

    # Precompute digits and idx conversions
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

    # Precompute balls as a 2D array: ball_matrix[i] contains the 17 indices in ball of i
    # Ball size = 1 + 8*2 = 17
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

    # Precompute inverse coverage: for each word j, which codewords cover it
    # inv_matrix[j] = list of codeword indices i such that j is in ball_matrix[i]
    # We'll build this on the fly during search, but precompute the structure
    # Actually, let's precompute for each word j, the 17 words at distance 1 from j
    # That's already ball_matrix[j]! The ball of j contains j and its 16 neighbors.
    # A codeword i covers word j iff j is in ball_matrix[i] iff i is in ball_matrix[j]
    # (by symmetry of Hamming distance)
    # So inv_matrix = ball_matrix (same structure, symmetric)

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
    chosen = list(seed_set)
    for i in seed_set:
        covered[ball_matrix[i]] = 1

    def greedy_fill(chosen, covered, time_budget=20.0):
        t_start = time.time()
        while time.time() - t_start < time_budget and _time_left() > 20.0:
            # Vectorized: for each candidate i, gain = sum of (1 - covered[j]) for j in ball_matrix[i]
            # = BALL_SIZE - sum(covered[ball_matrix[i]])
            gains = BALL_SIZE - covered[ball_matrix].sum(axis=1)
            best_i = int(np.argmax(gains))
            if gains[best_i] <= 0:
                break
            chosen.append(best_i)
            covered[ball_matrix[best_i]] = 1
        return chosen, covered

    chosen, covered = greedy_fill(chosen, covered)

    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.append(i)
                covered[ball_matrix[i]] = 1

    # Build coverage count
    covcount = np.zeros(N, dtype=np.int32)
    for i in chosen:
        covcount[ball_matrix[i]] += 1

    # Local search: removals
    chosen_set = set(chosen)
    def try_removals(chosen_set, covcount, time_budget=15.0):
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_budget and _time_left() > 15.0:
            improved = False
            order = list(chosen_set)
            rng.shuffle(order)
            for i in order:
                if i not in chosen_set:
                    continue
                if np.min(covcount[ball_matrix[i]]) >= 2:
                    covcount[ball_matrix[i]] -= 1
                    chosen_set.discard(i)
                    improved = True
        return chosen_set, covcount

    chosen_set, covcount = try_removals(chosen_set, covcount)

    # Improved local search with effective swaps using inverse coverage
    def improved_local_search(chosen_set, covcount, time_budget=50.0):
        t_start = time.time()
        best_set = set(chosen_set)
        best_size = len(chosen_set)
        cur_set = set(chosen_set)
        cur_size = len(cur_set)
        
        # Precompute: for each word j, the set of codewords that cover it
        # This is the same as ball_matrix[j] intersected with cur_set
        # We'll maintain this dynamically
        
        # Count how many words have covcount == 1 (critical words)
        critical_count = int(np.sum(covcount == 1))
        
        while time.time() - t_start < time_budget and _time_left() > 10.0:
            # Try removals first
            made_progress = False
            for i in list(cur_set):
                if i not in cur_set:
                    continue
                b = ball_matrix[i]
                if np.min(covcount[b]) >= 2:
                    # Safe removal
                    covcount[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    made_progress = True
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
            
            if not made_progress:
                # Try swaps: remove i, add j
                # For each i in cur_set, find words that would be uncovered
                # Then find j that covers those words
                i = rng.choice(list(cur_set))
                b = ball_matrix[i]
                would_uncover = b[covcount[b] == 1]
                
                if len(would_uncover) == 0:
                    # Should have been caught by removals, but handle anyway
                    covcount[b] -= 1
                    cur_set.discard(i)
                    cur_size -= 1
                    if cur_size < best_size:
                        best_size = cur_size
                        best_set = set(cur_set)
                    continue
                
                # Target: find j not in cur_set such that ball_matrix[j] covers all words in would_uncover
                # And ideally, j covers words that are currently over-covered
                target_words = would_uncover.tolist()
                
                # Strategy: for each target word, find codewords that cover it
                # The intersection of these sets gives candidate j's
                candidate_set = None
                for tw in target_words:
                    # Codewords that cover tw are those i' such that tw in ball_matrix[i']
                    # i.e., i' in ball_matrix[tw]
                    covering = set(ball_matrix[tw].tolist()) - cur_set
                    if candidate_set is None:
                        candidate_set = covering
                    else:
                        candidate_set &= covering
                    
                    if not candidate_set:
                        break
                
                if candidate_set:
                    # Pick the best candidate: one that maximizes "slack" (covers over-covered words)
                    candidates = list(candidate_set)
                    best_j = None
                    best_score = -1
                    for j in candidates:
                        bj = ball_matrix[j]
                        # Score: number of words in bj with covcount >= 2 (slack we can use)
                        score = int(np.sum(covcount[bj] >= 2))
                        if score > best_score:
                            best_score = score
                            best_j = j
                    
                    if best_j is not None:
                        # Perform the swap
                        bj = ball_matrix[best_j]
                        covcount[b] -= 1
                        covcount[bj] += 1
                        cur_set.discard(i)
                        cur_set.add(best_j)
                        # cur_size stays the same
                        if cur_size < best_size:
                            best_size = cur_size
                            best_set = set(cur_set)
        
        # Restore best
        cur_set = best_set
        return cur_set, covcount

    chosen_set, covcount = improved_local_search(chosen_set, covcount)

    # Final removals pass
    chosen_set, covcount = try_removals(chosen_set, covcount, time_budget=10.0)

    # Final verification
    final_covered = np.zeros(N, dtype=np.int8)
    for i in chosen_set:
        final_covered[ball_matrix[i]] = 1
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen_set.add(i)

    chosen = list(chosen_set)
    return [idx_to_digits(i) for i in chosen]
