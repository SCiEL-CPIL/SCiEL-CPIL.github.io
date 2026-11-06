# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)
ZERO = Fraction(0, 1)

def _try_improve(best_idx, best_deficit, deadline=None):
    """Local search: 1-swap moves to reduce deficit while keeping sum < 1."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    changed = True
    while changed:
        if deadline and time.time() > deadline:
            break
        changed = False
        for i in best_idx[:]:
            if deadline and time.time() > deadline:
                break
            f_out = FRAC_DENOMS[i]
            sum_without = best_sum - f_out
            for j in range(N):
                if j in in_set:
                    continue
                f_in = FRAC_DENOMS[j]
                if sum_without + f_in >= ONE:
                    continue
                new_sum = sum_without + f_in
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.remove(i)
                    best_idx.append(j)
                    in_set.remove(i)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    changed = True
                    break
            if changed:
                break
    return best_idx, best_deficit

def _pairwise_swap_improve(best_idx, best_deficit, deadline=None):
    """Try pairwise swaps (remove two, add two) for larger jumps."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    k = len(best_idx)
    
    if k < 2:
        return best_idx, best_deficit
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        n_remove = len(best_idx)
        for a in range(n_remove):
            if deadline and time.time() > deadline:
                break
            for b in range(a+1, n_remove):
                if deadline and time.time() > deadline:
                    break
                i, j = best_idx[a], best_idx[b]
                sum_without_two = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j]
                for x in range(N):
                    if x in in_set:
                        continue
                    f_x = FRAC_DENOMS[x]
                    sum_after_x = sum_without_two + f_x
                    if sum_after_x >= ONE:
                        continue
                    for y in range(x+1, N):
                        if y in in_set:
                            continue
                        f_y = FRAC_DENOMS[y]
                        new_sum = sum_after_x + f_y
                        if new_sum >= ONE:
                            continue
                        new_def = ONE - new_sum
                        if new_def < best_deficit:
                            best_idx.remove(i)
                            best_idx.remove(j)
                            best_idx.append(x)
                            best_idx.append(y)
                            in_set.remove(i)
                            in_set.remove(j)
                            in_set.add(x)
                            in_set.add(y)
                            best_sum = new_sum
                            best_deficit = new_def
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_idx, best_deficit

def _add_remove_improve(best_idx, best_deficit, deadline=None):
    """Try adding one element or removing one element."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try adding one
        for j in range(N):
            if j in in_set:
                continue
            new_sum = best_sum + FRAC_DENOMS[j]
            if new_sum < ONE:
                new_def = ONE - new_sum
                if new_def < best_deficit:
                    best_idx.append(j)
                    in_set.add(j)
                    best_sum = new_sum
                    best_deficit = new_def
                    improved = True
                    break
        if improved:
            continue
        # Try removing one
        for i in best_idx[:]:
            if deadline and time.time() > deadline:
                break
            new_sum = best_sum - FRAC_DENOMS[i]
            new_def = ONE - new_sum
            if new_def < best_deficit:
                best_idx.remove(i)
                in_set.remove(i)
                best_sum = new_sum
                best_deficit = new_def
                improved = True
                break
    return best_idx, best_deficit

def _triple_swap_improve(best_idx, best_deficit, deadline=None):
    """Try triple swaps (remove three, add three)."""
    best_idx = list(best_idx)
    best_deficit = best_deficit
    best_sum = ONE - best_deficit
    in_set = set(best_idx)
    k = len(best_idx)
    
    if k < 3:
        return best_idx, best_deficit
    
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        n_remove = len(best_idx)
        for a in range(n_remove):
            if deadline and time.time() > deadline:
                break
            for b in range(a+1, n_remove):
                if deadline and time.time() > deadline:
                    break
                for c in range(b+1, n_remove):
                    if deadline and time.time() > deadline:
                        break
                    i, j, l = best_idx[a], best_idx[b], best_idx[c]
                    sum_without_three = best_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j] - FRAC_DENOMS[l]
                    for x in range(N):
                        if x in in_set:
                            continue
                        f_x = FRAC_DENOMS[x]
                        sum_after_x = sum_without_three + f_x
                        if sum_after_x >= ONE:
                            continue
                        for y in range(x+1, N):
                            if y in in_set:
                                continue
                            f_y = FRAC_DENOMS[y]
                            sum_after_y = sum_after_x + f_y
                            if sum_after_y >= ONE:
                                continue
                            for z in range(y+1, N):
                                if z in in_set:
                                    continue
                                f_z = FRAC_DENOMS[z]
                                new_sum = sum_after_y + f_z
                                if new_sum >= ONE:
                                    continue
                                new_def = ONE - new_sum
                                if new_def < best_deficit:
                                    best_idx.remove(i)
                                    best_idx.remove(j)
                                    best_idx.remove(l)
                                    best_idx.append(x)
                                    best_idx.append(y)
                                    best_idx.append(z)
                                    in_set.remove(i)
                                    in_set.remove(j)
                                    in_set.remove(l)
                                    in_set.add(x)
                                    in_set.add(y)
                                    in_set.add(z)
                                    best_sum = new_sum
                                    best_deficit = new_def
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_idx, best_deficit

def _simulated_annealing(start_idx, start_deficit, deadline, n_iter=2000):
    """Simulated annealing to escape local optima."""
    import math
    cur_idx = list(start_idx)
    cur_sum = sum(FRAC_DENOMS[i] for i in cur_idx)
    cur_def = ONE - cur_sum
    in_set = set(cur_idx)
    
    T_start = 0.05
    T_end = 0.0001
    T = T_start
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(n_iter, 1))
        
        # Random move: add, remove, or swap
        move_type = random.random()
        if move_type < 0.3 and len(cur_idx) < N:
            # Add
            candidates = [j for j in range(N) if j not in in_set]
            if not candidates:
                continue
            j = random.choice(candidates)
            new_sum = cur_sum + FRAC_DENOMS[j]
            if new_sum >= ONE:
                continue
            new_def = ONE - new_sum
            delta = new_def - cur_def
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_idx.append(j)
                in_set.add(j)
                cur_sum = new_sum
                cur_def = new_def
        elif move_type < 0.5 and len(cur_idx) > 1:
            # Remove
            i = random.choice(cur_idx)
            new_sum = cur_sum - FRAC_DENOMS[i]
            new_def = ONE - new_sum
            delta = new_def - cur_def
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_idx.remove(i)
                in_set.remove(i)
                cur_sum = new_sum
                cur_def = new_def
        else:
            # Swap
            if len(cur_idx) == 0:
                continue
            i = random.choice(cur_idx)
            candidates = [j for j in range(N) if j not in in_set]
            if not candidates:
                continue
            j = random.choice(candidates)
            new_sum = cur_sum - FRAC_DENOMS[i] + FRAC_DENOMS[j]
            if new_sum >= ONE:
                continue
            new_def = ONE - new_sum
            delta = new_def - cur_def
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_idx.remove(i)
                cur_idx.append(j)
                in_set.remove(i)
                in_set.add(j)
                cur_sum = new_sum
                cur_def = new_def
    
    return cur_idx, cur_def

def _greedy_seed():
    selected, total = [], Fraction()
    for i, d in enumerate(DENOMINATORS):
        if total + Fraction(1, d) < ONE:
            selected.append(i)
            total += Fraction(1, d)
    return selected, total

def _known_good_seeds():
    seeds = []
    # {2,3,7,43}
    s = [0, 1, 5, 41]
    sm = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,43)
    if sm < ONE:
        seeds.append((s, sm))
    # {2,3,8,24}
    s = [0, 1, 6, 22]
    sm = Fraction(1,2) + Fraction(1,3) + Fraction(1,8) + Fraction(1,24)
    if sm < ONE:
        seeds.append((s, sm))
    # {2,3,9,18}
    s = [0, 1, 7, 16]
    sm = Fraction(1,2) + Fraction(1,3) + Fraction(1,9) + Fraction(1,18)
    if sm < ONE:
        seeds.append((s, sm))
    # {2,3,10,15}
    s = [0, 1, 8, 13]
    sm = Fraction(1,2) + Fraction(1,3) + Fraction(1,10) + Fraction(1,15)
    if sm < ONE:
        seeds.append((s, sm))
    # {2,4,5,20}
    s = [0, 2, 3, 18]
    sm = Fraction(1,2) + Fraction(1,4) + Fraction(1,5) + Fraction(1,20)
    if sm < ONE:
        seeds.append((s, sm))
    return seeds

def construct():
    deadline = time.time() + 75  # Leave 15s safety margin
    
    best_idx = None
    best_deficit = None
    
    # Try multiple seeds
    seeds = []
    
    # Greedy seed
    g_idx, g_sum = _greedy_seed()
    seeds.append((g_idx, g_sum))
    
    # Known good seeds
    for s_idx, s_sum in _known_good_seeds():
        seeds.append((s_idx, s_sum))
    
    # Random seeds
    for _ in range(5):
        r_idx = random.sample(range(N), random.randint(3, 10))
        r_sum = sum(FRAC_DENOMS[i] for i in r_idx)
        if r_sum < ONE:
            seeds.append((r_idx, r_sum))
    
    for s_idx, s_sum in seeds:
        if time.time() > deadline:
            break
        s_def = ONE - s_sum
        if best_deficit is None or s_def < best_deficit:
            best_idx = list(s_idx)
            best_deficit = s_def
    
    if best_idx is None:
        # Fallback: single element
        best_idx = [0]
        best_deficit = ONE - Fraction(1, 2)
    
    # Improve with local search
    best_idx, best_deficit = _add_remove_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    best_idx, best_deficit = _triple_swap_improve(best_idx, best_deficit, deadline)
    
    # Simulated annealing from best solution
    if time.time() < deadline - 5:
        sa_idx, sa_def = _simulated_annealing(best_idx, best_deficit, deadline - 2, n_iter=3000)
        if sa_def < best_deficit:
            best_idx = sa_idx
            best_deficit = sa_def
            # Polish the SA result
            best_idx, best_deficit = _add_remove_improve(best_idx, best_deficit, deadline)
            best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
            best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    
    # More SA restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            # Random perturbation of best
            pert_idx = list(best_idx)
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5 and len(pert_idx) > 1:
                    i = random.choice(pert_idx)
                    pert_idx.remove(i)
                else:
                    candidates = [j for j in range(N) if j not in set(pert_idx)]
                    if candidates:
                        j = random.choice(candidates)
                        pert_idx.append(j)
            pert_sum = sum(FRAC_DENOMS[i] for i in pert_idx)
            if pert_sum >= ONE:
                continue
            pert_def = ONE - pert_sum
            sa_idx, sa_def = _simulated_annealing(pert_idx, pert_def, deadline - 1, n_iter=2000)
            if sa_def < best_deficit:
                best_idx = sa_idx
                best_deficit = sa_def
                best_idx, best_deficit = _add_remove_improve(best_idx, best_deficit, deadline)
                best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
    
    # Final polish
    if time.time() < deadline - 1:
        best_idx, best_deficit = _add_remove_improve(best_idx, best_deficit, deadline)
        best_idx, best_deficit = _try_improve(best_idx, best_deficit, deadline)
        best_idx, best_deficit = _pairwise_swap_improve(best_idx, best_deficit, deadline)
    
    return best_idx
# EVOLVE-BLOCK-END
