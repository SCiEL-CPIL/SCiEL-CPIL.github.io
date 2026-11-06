import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def get_score(s):
        return sum(1.0 / (512.0 - a) for a in s)

    # Precompute coprime relationships using bitmasks for speed
    # coprime_mask[a] is a bitmask where bit b is set if gcd(a, b) == 1
    coprime_mask = [0] * 512
    for a in range(1, 512):
        mask = 0
        for b in range(1, 512):
            if math.gcd(a, b) == 1:
                mask |= (1 << b)
        coprime_mask[a] = mask

    # Check if value v is coprime to all elements in chosen_set
    def is_coprime_to_set(v, chosen_set):
        for z in chosen_set:
            if not (coprime_mask[v] >> z) & 1:
                return False
        return True

    # Greedy initialization: pick largest available coprime numbers
    def greedy_init():
        chosen = []
        chosen_set = set()
        for a in range(511, 0, -1):
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    # 1-swap local search: find best single swap improvement
    def one_swap_improve(chosen, chosen_set):
        best_gain = 0
        best_swap = None
        for x in chosen:
            for y in range(511, x, -1):
                if y in chosen_set:
                    continue
                # Check if y is coprime to all z in chosen \ {x}
                ok = True
                for z in chosen:
                    if z == x:
                        continue
                    if not (coprime_mask[y] >> z) & 1:
                        ok = False
                        break
                if ok:
                    gain = (1.0 / (512.0 - y)) - (1.0 / (512.0 - x))
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
        if best_swap:
            x, y = best_swap
            chosen_set.remove(x)
            chosen_set.add(y)
            chosen = list(chosen_set)
        return chosen, chosen_set, best_gain

    # 2-swap local search: remove two, add two
    def two_swap_improve(chosen, chosen_set):
        best_gain = 0
        best_swap = None
        n = len(chosen)
        for i in range(n):
            for j in range(i + 1, n):
                x1, x2 = chosen[i], chosen[j]
                # Remaining set after removing x1, x2
                remaining = [z for z in chosen if z != x1 and z != x2]
                # Find best pair (y1, y2) not in chosen, y1 > x1 or x2, y2 > x1 or x2
                # that are coprime to remaining and to each other
                for y1 in range(511, 0, -1):
                    if y1 in chosen_set:
                        continue
                    if not (coprime_mask[y1] >> x1) & 1 and not (coprime_mask[y1] >> x2) & 1:
                        # y1 must be coprime to remaining
                        ok1 = True
                        for z in remaining:
                            if not (coprime_mask[y1] >> z) & 1:
                                ok1 = False
                                break
                        if not ok1:
                            continue
                    # Check y1 coprime to remaining
                    ok1 = True
                    for z in remaining:
                        if not (coprime_mask[y1] >> z) & 1:
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    for y2 in range(y1 - 1, 0, -1):
                        if y2 in chosen_set:
                            continue
                        # y2 must be coprime to remaining and to y1
                        ok2 = True
                        for z in remaining:
                            if not (coprime_mask[y2] >> z) & 1:
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        if not (coprime_mask[y1] >> y2) & 1:
                            continue
                        gain = ((1.0 / (512.0 - y1)) + (1.0 / (512.0 - y2))) - \
                               ((1.0 / (512.0 - x1)) + (1.0 / (512.0 - x2)))
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (x1, x2, y1, y2)
        if best_swap:
            x1, x2, y1, y2 = best_swap
            chosen_set.discard(x1)
            chosen_set.discard(x2)
            chosen_set.add(y1)
            chosen_set.add(y2)
            chosen = list(chosen_set)
        return chosen, chosen_set, best_gain

    # Full local search: alternate 1-swap and 2-swap until no improvement
    def local_search(chosen, chosen_set):
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            chosen, chosen_set, g1 = one_swap_improve(chosen, chosen_set)
            if g1 > 0:
                improved = True
            if time.time() > deadline - 10:
                break
            chosen, chosen_set, g2 = two_swap_improve(chosen, chosen_set)
            if g2 > 0:
                improved = True
        return chosen, chosen_set

    # Simulated annealing
    def simulated_annealing(chosen, chosen_set, max_iters=5000):
        T = 0.01  # initial temperature
        T_min = 1e-6
        cooling = 0.999
        best_chosen = list(chosen)
        best_set = set(chosen_set)
        best_score = get_score(chosen)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random move: swap one element
            if not chosen:
                break
            x = random.choice(chosen)
            # Try a few random y values
            found = False
            for _ in range(20):
                y = random.randint(1, 511)
                if y in chosen_set:
                    continue
                # Check y coprime to chosen \ {x}
                ok = True
                for z in chosen:
                    if z == x:
                        continue
                    if not (coprime_mask[y] >> z) & 1:
                        ok = False
                        break
                if ok:
                    gain = (1.0 / (512.0 - y)) - (1.0 / (512.0 - x))
                    # Accept with probability
                    if gain >= 0 or random.random() < math.exp(gain / T):
                        chosen_set.remove(x)
                        chosen_set.add(y)
                        chosen = list(chosen_set)
                        cur_score = get_score(chosen)
                        if cur_score > best_score:
                            best_score = cur_score
                            best_chosen = list(chosen)
                            best_set = set(chosen_set)
                        found = True
                        break
            if not found:
                # Try adding a new element (if possible)
                for _ in range(10):
                    y = random.randint(1, 511)
                    if y in chosen_set:
                        continue
                    if is_coprime_to_set(y, chosen_set):
                        chosen_set.add(y)
                        chosen = list(chosen_set)
                        cur_score = get_score(chosen)
                        if cur_score > best_score:
                            best_score = cur_score
                            best_chosen = list(chosen)
                            best_set = set(chosen_set)
                        found = True
                        break
            if not found:
                # Try removing an element
                if len(chosen) > 1:
                    x = random.choice(chosen)
                    chosen_set.remove(x)
                    chosen = list(chosen_set)
            T = max(T * cooling, T_min)
        
        return best_chosen, best_set, best_score

    # Main algorithm
    # Start with greedy
    chosen, chosen_set = greedy_init()
    chosen, chosen_set = local_search(chosen, chosen_set)
    best_score = get_score(chosen)
    best_chosen = list(chosen)
    best_set = set(chosen_set)

    # Multiple restarts with simulated annealing
    random.seed(42)
    while time.time() < deadline - 15:
        # Perturb the current best solution
        perturbed_chosen = list(best_chosen)
        perturbed_set = set(best_set)
        
        # Randomly remove 1-3 elements and re-greedy
        k = random.randint(1, min(3, len(perturbed_chosen)))
        to_remove = random.sample(perturbed_chosen, k)
        for x in to_remove:
            perturbed_set.discard(x)
            perturbed_chosen.remove(x)
        
        # Re-greedy fill
        for a in range(511, 0, -1):
            if a not in perturbed_set and is_coprime_to_set(a, perturbed_set):
                perturbed_set.add(a)
                perturbed_chosen.append(a)
        
        # Local search
        perturbed_chosen, perturbed_set = local_search(perturbed_chosen, perturbed_set)
        ps = get_score(perturbed_chosen)
        
        if ps > best_score:
            best_score = ps
            best_chosen = list(perturbed_chosen)
            best_set = set(perturbed_set)
        
        # Simulated annealing from current best
        if time.time() < deadline - 10:
            sa_chosen, sa_set, sa_score = simulated_annealing(list(best_chosen), set(best_set), max_iters=2000)
            if sa_score > best_score:
                best_score = sa_score
                best_chosen = list(sa_chosen)
                best_set = set(sa_set)

    # Final polish: exact local search with 1-swap and 2-swap
    best_chosen, best_set = local_search(best_chosen, best_set)

    return list(best_set)
