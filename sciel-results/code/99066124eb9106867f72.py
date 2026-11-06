"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    random.seed(12345)

    n = 34
    triples = list(combinations(range(n), 3))
    ntriples = len(triples)
    triple_index = {}
    for i, (a, b, c) in enumerate(triples):
        triple_index[(a, b, c)] = i

    red_k4s = []
    for quad in combinations(range(n), 4):
        idxs = [triple_index[tuple(sorted(q))] for q in combinations(quad, 3)]
        red_k4s.append(idxs)

    blue_k5s = []
    for quint in combinations(range(n), 5):
        idxs = [triple_index[tuple(sorted(q))] for q in combinations(quint, 3)]
        blue_k5s.append(idxs)

    triple_red_k4s = [[] for _ in range(ntriples)]
    for k4_idx, idxs in enumerate(red_k4s):
        for idx in idxs:
            triple_red_k4s[idx].append(k4_idx)
    triple_blue_k5s = [[] for _ in range(ntriples)]
    for k5_idx, idxs in enumerate(blue_k5s):
        for idx in idxs:
            triple_blue_k5s[idx].append(k5_idx)

    def count_violations(colors):
        red_viol = 0
        for idxs in red_k4s:
            if all(colors[i] == 0 for i in idxs):
                red_viol += 1
        blue_viol = 0
        for idxs in blue_k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_viol += 1
        return red_viol + blue_viol

    p_red = 0.6
    colors = [0 if random.random() < p_red else 1 for _ in range(ntriples)]
    current_score = count_violations(colors)
    best_colors = colors[:]
    best_score = current_score

    if best_score == 0:
        return {'n': n, 'colors': best_colors}

    temp = 1.0
    cool = 0.99
    min_temp = 0.001

    while time.time() < deadline - 5:
        i = random.randrange(ntriples)
        old = colors[i]
        new = 1 - old
        colors[i] = new

        delta = 0
        if old == 0:
            for k4 in triple_red_k4s[i]:
                idxs = red_k4s[k4]
                all_red = True
                for j in idxs:
                    if j != i and colors[j] != 0:
                        all_red = False
                        break
                if all_red:
                    delta -= 1
            for k5 in triple_blue_k5s[i]:
                idxs = blue_k5s[k5]
                all_blue = True
                for j in idxs:
                    if j != i and colors[j] != 1:
                        all_blue = False
                        break
                if all_blue:
                    delta += 1
        else:
            for k5 in triple_blue_k5s[i]:
                idxs = blue_k5s[k5]
                all_blue = True
                for j in idxs:
                    if j != i and colors[j] != 1:
                        all_blue = False
                        break
                if all_blue:
                    delta -= 1
            for k4 in triple_red_k4s[i]:
                idxs = red_k4s[k4]
                all_red = True
                for j in idxs:
                    if j != i and colors[j] != 0:
                        all_red = False
                        break
                if all_red:
                    delta += 1

        new_score = current_score + delta
        if delta <= 0 or random.random() < (temp / max(1, abs(delta))):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
                if best_score == 0:
                    return {'n': n, 'colors': best_colors}
        else:
            colors[i] = old

        temp *= cool
        if temp < min_temp:
            temp = min_temp

    # Final verification
    if count_violations(best_colors) == 0:
        return {'n': n, 'colors': best_colors}

    # Fallback
    n_small = 9
    return {'n': n_small, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_small), 3)]}
# EVOLVE-BLOCK-END
