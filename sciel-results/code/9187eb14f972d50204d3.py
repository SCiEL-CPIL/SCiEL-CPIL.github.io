import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 120s

    def check_valid(n, colors):
        """Verify a coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
        tidx = {t: i for i, t in enumerate(triples)}
        for k4 in combinations(range(n), 4):
            if all(colors[tidx[t]] == 0 for t in combinations(k4, 3)):
                return False
        for k5 in combinations(range(n), 5):
            if all(colors[tidx[t]] == 1 for t in combinations(k5, 3)):
                return False
        return True

    def solve_n(n, time_budget):
        """Find a valid coloring for n vertices using weighted local search."""
        triples = list(combinations(range(n), 3))
        num_t = len(triples)
        tidx = {t: i for i, t in enumerate(triples)}

        # Precompute incidence
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)

        k4_triples = []
        for k4 in k4s:
            k4_triples.append([tidx[t] for t in combinations(k4, 3)])
        k5_triples = []
        for k5 in k5s:
            k5_triples.append([tidx[t] for t in combinations(k5, 3)])

        triple_to_k4 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k4_triples):
            for t in ts:
                triple_to_k4[t].append(i)
        triple_to_k5 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k5_triples):
            for t in ts:
                triple_to_k5[t].append(i)

        # Energy: weighted sum of violations and near-violations
        # K4: red_count == 4 -> violation (weight 100), red_count == 3 -> near (weight 1)
        # K5: red_count == 0 -> violation (weight 100), red_count == 1 -> near (weight 1)
        W_VIOL = 100
        W_NEAR = 1

        def compute_energy(colors):
            rc4 = [0] * num_k4
            rc5 = [0] * num_k5
            for i, c in enumerate(colors):
                if c == 0:
                    for k in triple_to_k4[i]:
                        rc4[k] += 1
                    for k in triple_to_k5[i]:
                        rc5[k] += 1
            e = 0
            for r in rc4:
                if r == 4:
                    e += W_VIOL
                elif r == 3:
                    e += W_NEAR
            for r in rc5:
                if r == 0:
                    e += W_VIOL
                elif r == 1:
                    e += W_NEAR
            return e, rc4, rc5

        def delta_energy(t, old_color, rc4, rc5):
            """Compute change in energy when flipping triple t."""
            d = 0
            for k in triple_to_k4[t]:
                r = rc4[k]
                if old_color == 0:  # R->B, r decreases
                    if r == 4:
                        d -= W_VIOL
                    elif r == 3:
                        d -= W_NEAR
                    # after: r-1; if r-1==3 (r==4) handled; if r-1==... no new near
                else:  # B->R, r increases
                    if r == 3:
                        d += W_VIOL  # becomes 4
                    elif r == 2:
                        d += W_NEAR  # becomes 3
            for k in triple_to_k5[t]:
                r = rc5[k]
                if old_color == 0:  # R->B, r decreases
                    if r == 1:
                        d -= W_NEAR  # was near, now 0 -> but 0 is violation!
                        d += W_VIOL  # becomes violation
                        # net: +W_VIOL - W_NEAR
                    elif r == 2:
                        d += W_NEAR  # becomes 1 (near)
                    # r==0 impossible (this triple is red in it)
                else:  # B->R, r increases
                    if r == 0:
                        d -= W_VIOL  # was violation
                        d += W_NEAR  # becomes near
                    elif r == 1:
                        d -= W_NEAR  # was near, now 2
            return d

        # Generate diverse initial colorings
        def random_colors():
            return [random.randint(0, 1) for _ in range(num_t)]

        def pattern_colors(seed_pattern):
            """Structured initial coloring based on vertex patterns."""
            cols = []
            for (a, b, c) in triples:
                if seed_pattern == 0:
                    col = (a + b + c) % 2
                elif seed_pattern == 1:
                    col = (a * b + c) % 2
                elif seed_pattern == 2:
                    col = (a ^ b ^ c) & 1
                elif seed_pattern == 3:
                    col = (a + 2*b + 3*c) % 3 < 2
                else:
                    col = random.randint(0, 1)
                cols.append(col)
            return cols

        def local_search(colors, time_limit, T0=1.0, cooling=0.998, min_T=0.05):
            """Weighted simulated annealing with constraint-guided moves."""
            energy, rc4, rc5 = compute_energy(colors)
            if energy == 0:
                return colors, 0

            T = T0
            best_energy = energy
            best_colors = list(colors)
            t_end = time.time() + time_limit
            steps = 0
            max_steps = 200000

            while steps < max_steps and time.time() < t_end:
                if time.time() > deadline - 5:
                    break

                # Choose move: 70% random triple, 30% triple from a violating constraint
                if random.random() < 0.7:
                    t = random.randint(0, num_t - 1)
                else:
                    # Pick a random K4 or K5 that's violating/near-violating, then a random triple in it
                    if random.random() < 0.5:
                        # Find violating/near K4s
                        candidates = []
                        for k in range(num_k4):
                            if rc4[k] >= 3:
                                candidates.append(k)
                        if candidates:
                            k = random.choice(candidates)
                            t = random.choice(k4_triples[k])
                        else:
                            t = random.randint(0, num_t - 1)
                    else:
                        candidates = []
                        for k in range(num_k5):
                            if rc5[k] <= 1:
                                candidates.append(k)
                        if candidates:
                            k = random.choice(candidates)
                            t = random.choice(k5_triples[k])
                        else:
                            t = random.randint(0, num_t - 1)

                old_color = colors[t]
                new_color = 1 - old_color
                d = delta_energy(t, old_color, rc4, rc5)

                if d < 0 or (T > min_T and random.random() < math.exp(-d / T)):
                    colors[t] = new_color
                    for k in triple_to_k4[t]:
                        if old_color == 0:
                            rc4[k] -= 1
                        else:
                            rc4[k] += 1
                    for k in triple_to_k5[t]:
                        if old_color == 0:
                            rc5[k] -= 1
                        else:
                            rc5[k] += 1
                    energy += d
                    if energy < best_energy:
                        best_energy = energy
                        best_colors = list(colors)
                    if best_energy == 0:
                        return best_colors, 0
                T *= cooling

                steps += 1
                if steps % 5000 == 0:
                    # Periodic reheat to escape local minima
                    if T < 0.3:
                        T = 0.5

            return best_colors, best_energy

        # Try multiple initializations
        num_starts = 8
        start_budget = time_budget / num_starts

        for s in range(num_starts):
            if time.time() > deadline - 10:
                break
            if s < 5:
                colors = pattern_colors(s)
            else:
                colors = random_colors()

            result, e = local_search(colors, start_budget)
            if e == 0:
                return result

        # Final attempt with longer budget on best pattern
        if time.time() < deadline - 15:
            colors = pattern_colors(0)
            result, e = local_search(colors, time.time() and (deadline - 15 - time.time()), T0=2.0, cooling=0.9995, min_T=0.1)
            if e == 0:
                return result

        return None

    # Try n from 34 down
    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        budget = min(15.0, deadline - 20 - time.time())
        if budget < 3:
            break
        result = solve_n(n, budget)
        if result is not None:
            # Verify
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
