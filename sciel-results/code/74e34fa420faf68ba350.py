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
        """Find a valid coloring for n vertices using Algebraic Seeding + Simulated Annealing."""
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

        def init_state(colors):
            rc4 = [0] * num_k4
            rc5 = [0] * num_k5
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for k in triple_to_k4[i]:
                        rc4[k] += 1
                else: # Blue
                    for k in triple_to_k5[i]:
                        rc5[k] += 1
            viol_k4 = sum(1 for r in rc4 if r == 4)
            viol_k5 = sum(1 for r in rc5 if r == 10)
            near_k4 = sum(1 for r in rc4 if r == 3)
            near_k5 = sum(1 for r in rc5 if r == 9)
            return rc4, rc5, viol_k4, viol_k5, near_k4, near_k5

        def objective(viol, near):
            return viol * 1000 + near

        def algebraic_seed(rule_id):
            """Generate a coloring based on algebraic rules."""
            cols = []
            # Precompute QRs mod 17 for speed
            qr17 = set()
            for x in range(17):
                qr17.add((x*x) % 17)
            
            for (a, b, c) in triples:
                if rule_id == 0:
                    col = (a + b + c) % 2
                elif rule_id == 1:
                    col = (a * b + b * c + c * a) % 2
                elif rule_id == 2:
                    col = (a * b * c) % 2
                elif rule_id == 3:
                    col = (a + b + c + a * b + b * c + c * a) % 2
                elif rule_id == 4:
                    # Mod 17 QR check
                    d = ((a - b) * (b - c) * (c - a)) % 17
                    col = 0 if d in qr17 else 1
                elif rule_id == 5:
                    # Linear mod 3
                    col = 0 if (a + 2*b + 3*c) % 3 == 0 else 1
                elif rule_id == 6:
                    # Another linear mod 5
                    col = 0 if (2*a + 3*b + c) % 5 == 0 else 1
                elif rule_id == 7:
                    # Quadratic mod 5
                    col = 0 if (a*a + b*b + c*c) % 5 == 0 else 1
                else:
                    col = random.randint(0, 1)
                cols.append(col)
            return cols

        def simulated_annealing(colors, time_limit):
            """Simulated Annealing with incremental state updates."""
            rc4, rc5, viol_k4, viol_k5, near_k4, near_k5 = init_state(colors)
            best_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
            best_colors = list(colors)
            
            t_end = time.time() + time_limit
            steps = 0
            max_steps = 200000
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.9995
            
            # Pre-compute lists for faster access? No, dynamic is better for SA.
            
            while steps < max_steps and time.time() < t_end:
                if time.time() > deadline - 5:
                    break
                
                # Select a triple to flip
                # Bias towards triples in violating or near-violating clusters
                candidates = []
                if random.random() < 0.8:
                    viol_k4_indices = [i for i, r in enumerate(rc4) if r == 4]
                    viol_k5_indices = [i for i, r in enumerate(rc5) if r == 10]
                    
                    if viol_k4_indices or viol_k5_indices:
                        if viol_k4_indices and (not viol_k5_indices or random.random() < 0.5):
                            k = random.choice(viol_k4_indices)
                            candidates = [t for t in k4_triples[k] if colors[t] == 0]
                        else:
                            k = random.choice(viol_k5_indices)
                            candidates = [t for t in k5_triples[k] if colors[t] == 1]
                    else:
                        near_k4_indices = [i for i, r in enumerate(rc4) if r == 3]
                        near_k5_indices = [i for i, r in enumerate(rc5) if r == 9]
                        if near_k4_indices:
                            k = random.choice(near_k4_indices)
                            candidates = [t for t in k4_triples[k] if colors[t] == 0]
                        elif near_k5_indices:
                            k = random.choice(near_k5_indices)
                            candidates = [t for t in k5_triples[k] if colors[t] == 1]
                
                if not candidates:
                    candidates = list(range(num_t))
                
                t = random.choice(candidates)
                old_color = colors[t]
                new_color = 1 - old_color
                
                d_obj = 0
                d_viol_k4 = 0
                d_viol_k5 = 0
                d_near_k4 = 0
                d_near_k5 = 0
                
                for k in triple_to_k4[t]:
                    r = rc4[k]
                    if old_color == 0: # R -> B
                        if r == 4:
                            d_viol_k4 -= 1
                            d_near_k4 += 1
                            d_obj -= 999 # 1000 - 1
                        elif r == 3:
                            d_near_k4 -= 1
                            d_obj -= 1
                    else: # B -> R
                        if r == 3:
                            d_viol_k4 += 1
                            d_near_k4 -= 1
                            d_obj += 999
                        elif r == 2:
                            d_near_k4 += 1
                            d_obj += 1
                
                for k in triple_to_k5[t]:
                    r = rc5[k]
                    if old_color == 0: # R -> B
                        if r == 9:
                            d_viol_k5 += 1
                            d_near_k5 -= 1
                            d_obj += 999
                        elif r == 8:
                            d_near_k5 += 1
                            d_obj += 1
                    else: # B -> R
                        if r == 10:
                            d_viol_k5 -= 1
                            d_near_k5 += 1
                            d_obj -= 999
                        elif r == 9:
                            d_near_k5 -= 1
                            d_obj -= 1

                # Metropolis criterion
                if d_obj < 0:
                    accept = True
                else:
                    if T > T_min:
                        p = math.exp(-d_obj / T)
                        accept = random.random() < p
                    else:
                        accept = False

                if accept:
                    colors[t] = new_color
                    for k in triple_to_k4[t]:
                        if old_color == 0:
                            rc4[k] -= 1
                        else:
                            rc4[k] += 1
                    for k in triple_to_k5[t]:
                        if old_color == 0:
                            rc5[k] += 1
                        else:
                            rc5[k] -= 1
                    
                    viol_k4 += d_viol_k4
                    viol_k5 += d_viol_k5
                    near_k4 += d_near_k4
                    near_k5 += d_near_k5

                    current_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_colors = list(colors)
                        if best_obj == 0:
                            return best_colors, 0

                T *= cooling_rate
                steps += 1

            return best_colors, best_obj

        # 1. Try Algebraic Seeds
        for rule in range(8):
            if time.time() > deadline - 10:
                break
            colors = algebraic_seed(rule)
            if check_valid(n, colors):
                return colors

        # 2. Use best algebraic seed as start for SA
        # We don't know which is best without checking, so we just pick one or random
        # Let's pick rule 0 (simplest) or random
        start_colors = algebraic_seed(random.randint(0, 7))
        
        # 3. Run SA
        result, obj = simulated_annealing(start_colors, time_budget)
        if obj == 0:
            if check_valid(n, result):
                return result
        
        return None

    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        budget = min(20.0, deadline - 20 - time.time())
        if budget < 3:
            break
        result = solve_n(n, budget)
        if result is not None:
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
