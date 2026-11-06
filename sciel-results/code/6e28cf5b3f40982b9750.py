"""Greedy + local-search ternary covering code K_3(7,1)."""
import itertools
import random
import time

_DEADLINE = 140.0  # safety margin under 150s kill

def _all_words(q, n):
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def _within_radius(w, c, r):
    d = 0
    for i in range(len(w)):
        if w[i] != c[i]:
            d += 1
            if d > r:
                return False
    return True

def _covers(w, c, r):
    d = 0
    for i in range(len(w)):
        if w[i] != c[i]:
            d += 1
            if d > r:
                return False
    return True

def construct(q, n, radius):
    t0 = time.time()
    words = _all_words(q, n)
    N = len(words)

    # Greedy covering: repeatedly add the codeword that covers the most uncovered words.
    uncovered = [True] * N
    code = []
    remaining = N

    while remaining > 0:
        if time.time() > _DEADLINE - 5:
            break
        best_cand = None
        best_count = -1
        # Candidate set = all words (a codeword may be any word in the space)
        for cand in words:
            if best_count >= remaining:
                break
            cnt = 0
            for i in range(N):
                if uncovered[i] and _covers(words[i], cand, radius):
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_cand = cand
                if cnt >= remaining:
                    break
        if best_count <= 0:
            # fallback: add any uncovered word
            for i in range(N):
                if uncovered[i]:
                    best_cand = words[i]
                    break
        code.append(best_cand)
        for i in range(N):
            if uncovered[i] and _covers(words[i], best_cand, radius):
                uncovered[i] = False
                remaining -= 1

    # Local search: try to remove redundant codewords, then try 1-for-1 swaps.
    def is_covered_all(code_list):
        for i in range(N):
            ok = False
            for c in code_list:
                if _covers(words[i], c, radius):
                    ok = True
                    break
            if not ok:
                return False
        return True

    # Removal pass
    changed = True
    while changed:
        if time.time() > _DEADLINE - 5:
            break
        changed = False
        for idx in range(len(code) - 1, -1, -1):
            new_code = code[:idx] + code[idx+1:]
            if is_covered_all(new_code):
                code = new_code
                changed = True
                break

    # 1-for-1 swap pass (replace one codeword with another to keep size but allow further removals)
    rng = random.Random(12345)
    improved = True
    while improved and time.time() < _DEADLINE - 10:
        improved = False
        for idx in range(len(code)):
            if time.time() > _DEADLINE - 10:
                break
            # try replacing code[idx] with a random word, then attempt removals
            trial = code[:]
            trial[idx] = rng.randrange(N)
            # removal pass on trial
            t_changed = True
            while t_changed:
                if time.time() > _DEADLINE - 10:
                    break
                t_changed = False
                for j in range(len(trial) - 1, -1, -1):
                    if is_covered_all(trial[:j] + trial[j+1:]):
                        trial = trial[:j] + trial[j+1:]
                        t_changed = True
                        break
            if len(trial) < len(code):
                code = trial
                improved = True
                break

    return code
