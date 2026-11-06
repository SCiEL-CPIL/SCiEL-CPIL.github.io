import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

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
                else:  # B->R, r increases
                    if r == 3:
                        d += W_VIOL  # becomes 4
                    elif r == 2:
                        d += W_NEAR  # becomes 3
            for k in triple_to_k5[t]:
                r = rc5[k]
                if old_color == 0:  # R->B, r decreases
                    if r == 1:
                        d -= W_NEAR  # was near
                        d += W_VIOL  # becomes violation
                    elif r == 2:
                        d += W_NEAR  # becomes 1 (near)
                else:  # B->R, r increases
                    if r == 0:
                        d -= W_VIOL  # was violation
                        d += W_NEAR  # becomes near
                    elif r == 1:
                        d -= W_NEAR  # was near, now 2
            return d

        # Generate diverse initial colorings using Algebraic Seeds
        def algebraic_colors(seed_id):
            """Structured initial coloring based on GF(2) algebraic structures."""
            cols = []
            for (a, b, c) in triples:
                if seed_id == 0:
                    # Simple XOR (Linear)
                    col = (a ^ b ^ c) & 1
                elif seed_id == 1:
                    # Quadratic form
                    col = ((a & b) ^ (b & c) ^ (c & a)) & 1
                elif seed_id == 2:
                    # Cubic form
                    col = (a & b & c) & 1
                elif seed_id == 3:
                    # Weighted Linear
                    col = ((a + 2*b + 4*c) & 1)
                elif seed_id == 4:
                    # Random Linear Map (Simulating GF(2)^d mapping)
                    # Pre-compute random bits for vertices
                    # This needs to be consistent, so we use a hash-like function
                    # or just fixed random bits generated once outside? 
                    # For simplicity in this function, we use a deterministic pseudo-random 
                    # based on vertex index to simulate a random linear map.
                    # u_v = (v * 12345) & 1
                    ua = (a * 12345) & 1
                    ub = (b * 12345) & 1
                    uc = (c * 12345) & 1
                    col = (ua ^ ub ^ uc) & 1
                elif seed_id == 5:
                    # Another linear variant
                    col = ((a ^ (b << 1)) & 1) ^ ((b ^ (c << 1)) & 1) ^ ((c ^ (a << 1)) & 1)
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

                # Choose move: 
                # 50% Random triple
                # 25% Triple from a random K4 (if it has violations/near)
                # 25% Triple from a random K5 (if it has violations/near)
                
                r_move = random.random()
                t = -1
                if r_move < 0.5:
                    t = random.randint(0, num_t - 1)
                elif r_move < 0.75:
                    # Pick random K4
                    k4_idx = random.randint(0, num_k4 - 1)
                    r = rc4[k4_idx]
                    if r >= 3: # Violating or Near
                        t = random.choice(k4_triples[k4_idx])
                    else:
                        t = random.randint(0, num_t - 1)
                else:
                    # Pick random K5
                    k5_idx = random.randint(0, num_k5 - 1)
                    r = rc5[k5_idx]
                    if r <= 1: # Violating or Near
                        t = random.choice(k5_triples[k5_idx])
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
        num_starts = 10
        start_budget = time_budget / num_starts

        # Prioritize algebraic seeds
        for s in range(num_starts):
            if time.time() > deadline - 10:
                break
            
            if s < 6:
                colors = algebraic_colors(s)
            else:
                colors = [random.randint(0, 1) for _ in range(num_t)]

            result, e = local_search(colors, start_budget)
            if e == 0:
                return result

        # Final attempt with longer budget on best pattern
        if time.time() < deadline - 15:
            colors = algebraic_colors(0)
            result, e = local_search(colors, time.time() and (deadline - 15 - time.time()), T0=2.0, cooling=0.9995, min_T=0.1)
            if e == 0:
                return result

        return None

    # Try n from 34 down
    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        budget = min(20.0, deadline - 20 - time.time())
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
