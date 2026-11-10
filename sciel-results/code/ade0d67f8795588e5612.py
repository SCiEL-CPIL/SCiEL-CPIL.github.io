import time
import random
import math

def _score(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def construct():
    deadline = time.time() + 85
    n = 20
    max_val = 10000

    # Start with AP (minimizes |A+A| to theoretical min 39)
    best = list(range(1, n + 1))
    best_score = _score(best)

    current = best[:]
    current_score = best_score
    temp = 100.0
    cooling = 0.9998
    min_temp = 0.5

    while time.time() < deadline - 1.0:
        idx = random.randint(0, n - 1)
        new_val = random.randint(1, max_val)
        if new_val == current[idx] or new_val in current:
            continue
        new_set = current[:]
        new_set[idx] = new_val
        new_score = _score(new_set)

        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current = new_set
            current_score = new_score
            if current_score < best_score:
                best = current[:]
                best_score = current_score

        temp = max(temp * cooling, min_temp)

    return best
