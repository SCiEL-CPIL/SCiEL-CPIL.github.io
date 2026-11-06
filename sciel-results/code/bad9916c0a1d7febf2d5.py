import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

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
        """Find a valid coloring for n vertices using Greedy Violation Elimination."""
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
            return rc4, rc5, viol_k4, viol_k5

        def random_colors():
            return [random.randint(0, 1) for _ in range(num_t)]

        def structured_colors(mode):
            """Generate structured initial colorings."""
            cols = []
            for (a, b, c) in triples:
                if mode == 0:
                    col = (a + b + c) % 2
                elif mode == 1:
                    col = (a * b + c) % 2
                elif mode == 2:
                    col = (a ^ b ^ c) & 1
                elif mode == 3:
                    col = (a + 2*b + 3*c) % 3 < 2 
                else:
                    col = (a * 3 + b * 5 + c * 7) % 4 < 2
                cols.append(col)
            return cols

        def greedy_elimination(colors, time_limit):
            """
            Greedy strategy:
            1. Start with an initial coloring.
            2. While violations exist:
               a. Find all triples that are part of at least one violation.
               b. For each candidate triple, calculate the change in total violations if flipped.
               c. Pick the flip that minimizes total violations (or maximizes reduction).
               d. If no flip reduces violations, pick a random one from candidates to escape local minimum (stochastic hill climbing).
            """
            rc4, rc5, viol_k4, viol_k5 = init_state(colors)
            total_viol = viol_k4 + viol_k5
            best_colors = list(colors)
            best_viol = total_viol
            
            t_start = time.time()
            no_improve = 0
            max_no_improve = 500 # Escape threshold
            
            while time.time() - t_start < time_limit:
                if time.time() > deadline - 5:
                    break
                
                if total_viol == 0:
                    return colors, 0
                
                # Identify candidate triples: those involved in any violation
                # A triple is a candidate if flipping it might help.
                # Specifically, if a triple is Red and part of a Red K4 (rc4=4), flipping it to Blue breaks that K4.
                # If a triple is Blue and part of a Blue K5 (rc5=10), flipping it to Red breaks that K5.
                
                candidates = set()
                
                # Check Red violations
                for k in range(num_k4):
                    if rc4[k] == 4:
                        # All 4 triples in this K4 are Red. Flipping any of them to Blue reduces rc4[k] to 3.
                        for t in k4_triples[k]:
                            candidates.add(t)
                
                # Check Blue violations
                for k in range(num_k5):
                    if rc5[k] == 10:
                        # All 10 triples in this K5 are Blue. Flipping any of them to Red reduces rc5[k] to 9.
                        for t in k5_triples[k]:
                            candidates.add(t)
                
                if not candidates:
                    # No direct violations found? Should not happen if total_viol > 0 and logic is correct.
                    # But maybe we are in a state where violations exist but not counted?
                    # Or maybe we need to break ties differently.
                    # For safety, pick a random triple.
                    t = random.randint(0, num_t - 1)
                    candidates = {t}

                # Evaluate each candidate
                best_t = -1
                best_delta = 0 # We want to minimize new_viol, so maximize -delta (reduction)
                # Actually, let's calculate new_total_viol for each candidate.
                # To save time, we can estimate. But exact is better.
                # num_candidates is usually smallish (proportional to violations * 4 or 10).
                
                min_new_viol = float('inf')
                best_t = -1
                best_delta_viol = 0
                
                # To speed up, if there are too many candidates, sample them?
                # For n=34, max violations might be large initially, but as we progress, it should drop.
                # Let's just iterate through all candidates.
                
                for t in candidates:
                    old_color = colors[t]
                    
                    d_viol_k4 = 0
                    d_viol_k5 = 0
                    
                    # Calculate delta for flipping t
                    for k in triple_to_k4[t]:
                        r = rc4[k]
                        if old_color == 0: # R -> B
                            if r == 4:
                                d_viol_k4 -= 1
                            elif r == 3:
                                d_viol_k4 += 1
                        else: # B -> R
                            if r == 3:
                                d_viol_k4 += 1
                            elif r == 4:
                                d_viol_k4 -= 1
                    for k in triple_to_k5[t]:
                        r = rc5[k]
                        if old_color == 0: # R -> B
                            if r == 9:
                                d_viol_k5 += 1
                            elif r == 10:
                                d_viol_k5 -= 1
                        else: # B -> R
                            if r == 10:
                                d_viol_k5 -= 1
                            elif r == 9:
                                d_viol_k5 += 1
                    
                    new_total_viol = total_viol + d_viol_k4 + d_viol_k5
                    
                    if new_total_viol < min_new_viol:
                        min_new_viol = new_total_viol
                        best_t = t
                        best_delta_viol = d_viol_k4 + d_viol_k5

                if best_t == -1:
                    break

                # Apply best flip
                t = best_t
                old_color = colors[t]
                new_color = 1 - old_color
                
                # Update state
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
                
                colors[t] = new_color
                total_viol += best_delta_viol
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = list(colors)
                    no_improve = 0
                else:
                    no_improve += 1
                    # If stuck, do a random move to escape
                    if no_improve > 20:
                        # Random kick: flip a random triple
                        t_rand = random.randint(0, num_t - 1)
                        old_c = colors[t_rand]
                        new_c = 1 - old_c
                        
                        d_v4 = 0
                        d_v5 = 0
                        for k in triple_to_k4[t_rand]:
                            r = rc4[k]
                            if old_c == 0:
                                if r == 4: d_v4 -= 1
                                elif r == 3: d_v4 += 1
                            else:
                                if r == 3: d_v4 += 1
                                elif r == 4: d_v4 -= 1
                        for k in triple_to_k5[t_rand]:
                            r = rc5[k]
                            if old_c == 0:
                                if r == 9: d_v5 += 1
                                elif r == 10: d_v5 -= 1
                            else:
                                if r == 10: d_v5 -= 1
                                elif r == 9: d_v5 += 1
                        
                        for k in triple_to_k4[t_rand]:
                            if old_c == 0: rc4[k] -= 1
                            else: rc4[k] += 1
                        for k in triple_to_k5[t_rand]:
                            if old_c == 0: rc5[k] += 1
                            else: rc5[k] -= 1
                        
                        colors[t_rand] = new_c
                        total_viol += d_v4 + d_v5
                        no_improve = 0

            return best_colors, best_viol

        # Try multiple starts
        best_result = None
        best_obj = float('inf')
        
        t_start_solve = time.time()
        run_id = 0
        
        # Initial modes: 0,1,2,3,4 are structured, 5+ random
        while time.time() - t_start_solve < time_budget:
            if time.time() > deadline - 10:
                break
                
            # Choose initial coloring
            if run_id < 5:
                colors = structured_colors(run_id)
            else:
                colors = random_colors()
            
            time_left = time_budget - (time.time() - t_start_solve)
            if time_left < 1.0:
                break
            
            # Greedy is fast, so we can do many runs or one long run.
            # Let's give it a chunk of time, e.g., 2s per run.
            run_time = min(2.0, time_left)
            
            result, obj = greedy_elimination(colors, run_time)
            
            if obj < best_obj:
                best_obj = obj
                best_result = result
                
            if obj == 0:
                if check_valid(n, result):
                    return result
            
            run_id += 1
            
        return None

    # Try to find solution for largest n first
    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        
        if n >= 30:
            budget = min(60.0, deadline - 25 - time.time())
        else:
            budget = min(10.0, deadline - 25 - time.time())
            
        if budget < 2:
            break
            
        result = solve_n(n, budget)
        if result is not None:
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    # Fallback: return a small valid coloring
    n_fallback = 10
    if time.time() < deadline - 5:
        res = solve_n(n_fallback, 2.0)
        if res and check_valid(n_fallback, res):
            return {'n': n_fallback, 'colors': res}
            
    # Absolute fallback: n=9
    n = 9
    colors = []
    for a, b, c in combinations(range(n), 3):
        colors.append(0 if (a + b + c) % 3 == 0 else 1)
        
    return {'n': n, 'colors': colors}
