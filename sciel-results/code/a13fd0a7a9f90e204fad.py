"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Precompute conflict matrix: conflict[a][b] = True if lcm(a,b) <= 256
    # i.e., a*b/gcd(a,b) <= 256
    N = 256
    # Use 0-indexed: index i represents number i+2
    size = N - 1  # numbers 2..256
    conflict = [[False] * size for _ in range(size)]
    
    for i in range(size):
        a = i + 2
        for j in range(i + 1, size):
            b = j + 2
            g = gcd(a, b)
            if a * b // g <= 256:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Precompute weights as Fractions for exact comparison
    weights = [Fraction(1, i + 2) for i in range(size)]
    
    def is_valid_set(selected_indices):
        """Check if all pairs in selected_indices have no conflict."""
        sel = list(selected_indices)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if conflict[sel[i]][sel[j]]:
                    return False
        return True
    
    def score_of(selected_indices):
        s = Fraction(0)
        for idx in selected_indices:
            s += weights[idx]
        return s
    
    def can_add(selected_set, idx):
        """Check if idx can be added to selected_set (set of indices)."""
        for other in selected_set:
            if conflict[idx][other]:
                return False
        return True
    
    def greedy(order):
        """Greedy: add elements in given order if compatible."""
        selected = set()
        for idx in order:
            if can_add(selected, idx):
                selected.add(idx)
        return selected
    
    def local_search(selected_set, max_iters=50000):
        """1-swap and 2-swap local search."""
        selected = set(selected_set)
        current_score = score_of(selected)
        best_score = current_score
        best_set = set(selected)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # 1-swap: remove one, add another
            for rem in list(selected):
                for add in range(size):
                    if add in selected:
                        continue
                    # Check if removing rem and adding add is valid
                    new_selected = (selected - {rem}) | {add}
                    # Check add against all others
                    valid = True
                    for other in new_selected:
                        if other != add and conflict[add][other]:
                            valid = False
                            break
                    if valid:
                        new_score = current_score - weights[rem] + weights[add]
                        if new_score > current_score:
                            current_score = new_score
                            selected = new_selected
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(selected)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # 2-swap: remove two, add two
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    if time.time() > deadline:
                        return best_set
                    for j in range(i + 1, len(sel_list)):
                        rem1, rem2 = sel_list[i], sel_list[j]
                        base_after_removal = selected - {rem1, rem2}
                        for add1 in range(size):
                            if add1 in selected:
                                continue
                            if not can_add(base_after_removal, add1):
                                continue
                            for add2 in range(add1 + 1, size):
                                if add2 in selected:
                                    continue
                                if not can_add(base_after_removal, add2):
                                    continue
                                if conflict[add1][add2]:
                                    continue
                                new_selected = base_after_removal | {add1, add2}
                                new_score = current_score - weights[rem1] - weights[rem2] + weights[add1] + weights[add2]
                                if new_score > current_score:
                                    current_score = new_score
                                    selected = new_selected
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_set = set(selected)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_set
    
    # Generate multiple diverse starting points
    start_points = []
    
    # Order 1: ascending (original greedy)
    start_points.append(list(range(size)))
    
    # Order 2: descending
    start_points.append(list(range(size - 1, -1, -1)))
    
    # Order 3: by weight descending (small numbers first)
    order3 = sorted(range(size), key=lambda i: -weights[i])
    start_points.append(order3)
    
    # Order 4: by number of conflicts ascending (least constrained first)
    conf_counts = [sum(conflict[i][j] for j in range(size)) for i in range(size)]
    order4 = sorted(range(size), key=lambda i: conf_counts[i])
    start_points.append(order4)
    
    # Order 5: by weight/conflict ratio
    order5 = sorted(range(size), key=lambda i: -weights[i] / (conf_counts[i] + 1))
    start_points.append(order5)
    
    # Order 6: random-ish (use a fixed pseudo-random permutation)
    import random
    rng = random.Random(42)
    order6 = list(range(size))
    rng.shuffle(order6)
    start_points.append(order6)
    
    # Order 7: another random
    rng2 = random.Random(123)
    order7 = list(range(size))
    rng2.shuffle(order7)
    start_points.append(order7)
    
    # Run greedy on each starting order, then local search
    best_overall = set()
    best_overall_score = Fraction(0)
    
    for order in start_points:
        if time.time() > deadline:
            break
        gs = greedy(order)
        if not gs:
            continue
        ls = local_search(gs, max_iters=30000)
        sc = score_of(ls)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = ls
    
    # Convert back to numbers
    result = [i + 2 for i in best_overall]
    if not result:
        result = [256]  # fallback
    
    return sorted(result)
# EVOLVE-BLOCK-END
