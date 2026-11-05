import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for the 150s limit

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_cliques(n):
        triples = get_triples(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def initial_counts(colors, k4s, k5s):
        """
        Returns k4_red_counts and k5_blue_counts arrays.
        k4_red_counts[i] = number of red triples in k4s[i]
        k5_blue_counts[i] = number of blue triples in k5s[i]
        """
        num_k4s = len(k4s)
        num_k5s = len(k5s)
        k4_red_counts = np.zeros(num_k4s, dtype=np.int32)
        k5_blue_counts = np.zeros(num_k5s, dtype=np.int32)
        
        # Vectorized update is tricky with list of tuples, so loop is fine for init
        for i, idxs in enumerate(k4s):
            k4_red_counts[i] = np.sum(colors[idxs] == 0)
        for i, idxs in enumerate(k5s):
            k5_blue_counts[i] = np.sum(colors[idxs] == 1)
            
        return k4_red_counts, k5_blue_counts

    def count_violations_from_counts(k4_red_counts, k5_blue_counts):
        red_viol = np.sum(k4_red_counts == 4)
        blue_viol = np.sum(k5_blue_counts == 10)
        return red_viol + blue_viol

    def local_search_refine(n, colors, time_limit, k4s, k5s, triple_to_k4s, triple_to_k5s, seed=0):
        """
        Local search using fast delta calculation.
        """
        np.random.seed(seed)
        colors = colors.copy().astype(np.int8)
        num_triples = len(colors)
        
        k4_red_counts, k5_blue_counts = initial_counts(colors, k4s, k5s)
        current_cost = count_violations_from_counts(k4_red_counts, k5_blue_counts)
        best_cost = current_cost
        best_colors = colors.copy()
        
        if current_cost == 0:
            return colors, True

        temp = 1.0
        cool_rate = 0.9995
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit:
            # Select triple to flip
            # Heuristic: pick from triples involved in violations if possible, else random
            # For simplicity and speed, just random, but we can weight it.
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            # Calculate delta cost
            delta = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[t]:
                count = k4_red_counts[k4_idx]
                if old_color == 0: # Red -> Blue
                    # Red count decreases by 1
                    if count == 4:
                        delta -= 1 # Violation removed
                    k4_red_counts[k4_idx] = count - 1
                else: # Blue -> Red
                    # Red count increases by 1
                    if count == 3:
                        delta += 1 # Violation created
                    k4_red_counts[k4_idx] = count + 1
                    
            # Update K5 counts
            for k5_idx in triple_to_k5s[t]:
                count = k5_blue_counts[k5_idx]
                if old_color == 1: # Blue -> Red
                    # Blue count decreases by 1
                    if count == 10:
                        delta -= 1 # Violation removed
                    k5_blue_counts[k5_idx] = count - 1
                else: # Red -> Blue
                    # Blue count increases by 1
                    if count == 9:
                        delta += 1 # Violation created
                    k5_blue_counts[k5_idx] = count + 1
                    
            new_cost = current_cost + delta
            
            # Acceptance criterion
            accept = False
            if delta <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-delta / temp)
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve += 1
            else:
                # Revert counts if not accepted? 
                # Wait, I updated the counts in-place. If not accepted, I must revert them.
                # This is a bug in the logic above. I need to revert if not accepted.
                
                # Revert K4
                for k4_idx in triple_to_k4s[t]:
                    if old_color == 0:
                        k4_red_counts[k4_idx] += 1
                    else:
                        k4_red_counts[k4_idx] -= 1
                # Revert K5
                for k5_idx in triple_to_k5s[t]:
                    if old_color == 1:
                        k5_blue_counts[k5_idx] += 1
                    else:
                        k5_blue_counts[k5_idx] -= 1
                        
                no_improve += 1

            temp *= cool_rate

            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.5, temp * 1.5)
                no_improve = 0

            if no_improve > max_no_improve * 3:
                # Restart from best
                colors = best_colors.copy()
                k4_red_counts, k5_blue_counts = initial_counts(colors, k4s, k5s)
                current_cost = best_cost
                temp = max(0.8, temp)
                no_improve = 0

        return best_colors, (best_cost == 0)

    def verify(n, colors):
        triples = get_triples(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    # Try n=35 first, then down
    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 5:
            break

        # Allocate time based on n
        if n == 35:
            time_budget = min(60.0, remaining * 0.8)
        elif n == 34:
            time_budget = min(40.0, remaining * 0.7)
        else:
            time_budget = min(20.0, remaining * 0.5)

        if time_budget < 3.0:
            break

        time_limit = time.time() + time_budget

        found = False
        # Multiple restarts with different seeds
        num_seeds = 10 if n >= 34 else 5
        
        # Pre-build cliques once per n
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_cliques(n)

        for seed in range(num_seeds):
            if time.time() > time_limit - 2:
                break

            try:
                # Initialization: Random
                np.random.seed(seed)
                # Bias towards red or blue? 
                # Red K4 is 4 triples. Blue K5 is 10 triples.
                # Random 0.5 is a good start.
                colors = np.random.randint(0, 2, num_triples).astype(np.int8)
                
                # Run local search
                colors, is_valid = local_search_refine(
                    n, colors, time_limit - 2, 
                    k4s, k5s, triple_to_k4s, triple_to_k5s, 
                    seed=seed
                )

                if is_valid and verify(n, colors.tolist()):
                    final_n = n
                    final_colors = colors.tolist()
                    found = True
                    break
                    
            except Exception:
                continue

        if found:
            break

    # Fallback for small n if large n failed
    if final_n == 0 or not final_colors:
        for n in [20, 15, 10, 5, 4, 3, 2, 1]:
            triples = get_triples(n)
            colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
            if verify(n, colors):
                final_n = n
                final_colors = colors
                break
            if final_n > 0:
                break

    if final_n == 0:
        final_n = 1
        final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
