# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin vs 150s kill

    if q == 3 and n == 7 and radius == 1:
        return _ternary7_r1(deadline)

    # Generic fallback: greedy set cover with NumPy
    return _generic_greedy(q, n, radius, deadline)


def _ternary7_r1(deadline):
    n = 7
    q = 3
    total = q ** n  # 2187

    # Precompute all words as (2187, 7) array
    all_words = np.array(np.unravel_index(np.arange(total), [3] * n)).T

    # Precompute ball_indices: for each codeword index c, the 15 words in its ball (dist <= 1)
    # Ball size = 1 + 2*7 = 15
    ball_indices = np.empty((total, 15), dtype=np.int32)
    for c in range(total):
        w = all_words[c]
        ball = [c]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w2 = w.copy()
                    w2[i] = new_val
                    idx = int(np.dot(w2, [3 ** (n - 1 - j) for j in range(n)]))
                    ball.append(idx)
        ball_indices[c] = ball

    # Greedy set cover with vectorized marginal gain
    covered = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    chosen = []
    
    while True:
        if time.time() > deadline - 25:  # leave time for local search
            break
        if covered.all():
            break
        
        if len(chosen) == 0:
            # First codeword: all have gain 15, pick index 0
            chosen.append(0)
            cover_count[ball_indices[0]] += 1
            covered[ball_indices[0]] = True
            continue
        
        # Compute marginal gain for each unchosen codeword
        # gain[c] = number of NEW words covered by adding c
        ball_cov = covered[ball_indices]  # (2187, 15)
        gains = (1 - ball_cov).sum(axis=1)  # (2187,)
        
        # Mask out already chosen
        for c in chosen:
            gains[c] = -1
        
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        
        chosen.append(best)
        cover_count[ball_indices[best]] += 1
        covered[ball_indices[best]] = True
    
    # Local search polish: remove redundant codewords
    if time.time() < deadline - 5:
        chosen = _local_search_polish(chosen, ball_indices, cover_count, total, deadline)
    
    # Convert to list of codewords
    result = [all_words[c].tolist() for c in chosen]
    return result


def _local_search_polish(chosen, ball_indices, cover_count, total, deadline):
    chosen_list = list(chosen)
    
    # Try to remove codewords that are redundant (their entire ball is covered by others)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(chosen_list)):
            if time.time() > deadline - 3:
                break
            c = chosen_list[i]
            ball = ball_indices[c]
            if np.all(cover_count[ball] > 1):
                # Safe to remove
                chosen_list.pop(i)
                cover_count[ball] -= 1
                improved = True
                break  # restart scan since indices changed
    
    # Try 1-swap: remove one codeword, add another if it reduces total or maintains coverage
    # This is more complex; do a simple version
    if time.time() < deadline - 3:
        chosen_set = set(chosen_list)
        for i in range(len(chosen_list)):
            if time.time() > deadline - 3:
                break
            c = chosen_list[i]
            ball = ball_indices[c]
            # Check if removing c leaves any word uncovered
            if not np.all(cover_count[ball] > 1):
                continue
            # c is redundant, try to see if we can add a different codeword instead
            # that covers more uncovered words
            # For simplicity, just remove it
            chosen_list.pop(i)
            cover_count[ball] -= 1
            break
    
    return chosen_list


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = np.array(np.unravel_index(np.arange(total), [q] * n)).T
    covered = np.zeros(total, dtype=bool)
    chosen = []
    
    while not covered.all() and time.time() < deadline - 5:
        best_gain = -1
        best_c = -1
        for c in range(total):
            if c in set(chosen):
                continue
            ball = _ball_generic(c, all_words[c], q, n, radius)
            gain = sum(1 for w in ball if not covered[w])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain <= 0:
            break
        chosen.append(best_c)
        ball = _ball_generic(best_c, all_words[best_c], q, n, radius)
        for w in ball:
            covered[w] = True
    
    return [all_words[c].tolist() for c in chosen]


def _ball_generic(c_idx, word, q, n, radius):
    ball = [c_idx]
    for r in range(1, radius + 1):
        for positions in combinations(range(n), r):
            for values in product(range(q), repeat=r):
                w2 = list(word)
                for pos, val in zip(positions, values):
                    if val != w2[pos]:
                        w2[pos] = val
                idx = int(np.dot(w2, [q ** (n - 1 - j) for j in range(n)]))
                ball.append(idx)
    return ball
# EVOLVE-BLOCK-END
