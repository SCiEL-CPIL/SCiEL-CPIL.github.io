import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def get_score(s):
        return sum(1.0 / (512.0 - a) for a in s)

    # Precompute prime factors for each number 1..511
    # prime_factors[a] = set of distinct prime factors of a
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if not prime_factors[p]:
            # p is prime
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)

    # Check if value v is coprime to all elements in chosen_set
    def is_coprime_to_set(v, chosen_set):
        if v == 1:
            return True
        vf = prime_factors[v]
        if not vf:
            return True
        for z in chosen_set:
            if z == 1:
                continue
            if vf & prime_factors[z]:
                return False
        return True

    # Precompute score for each a
    score_val = [0.0] * 512
    for a in range(1, 512):
        score_val[a] = 1.0 / (512.0 - a)

    # Greedy initialization: pick largest available coprime numbers
    def greedy_init():
        chosen = []
        chosen_set = set()
        for a in range(511, 0, -1):
            if is_coprime_to_set(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    # 1-swap local search: find best single swap improvement
    def one_swap_improve(chosen, chosen_set):
        best_gain = 0
        best_swap = None
        n = len(chosen)
        for i in range(n):
            x = chosen[i]
            sx = score_val[x]
            # Try candidates y from 511 down, but only those with score_val[y] > sx
            # to guarantee positive gain
            for y in range(511, 1, -1):
                if y in chosen_set:
                    continue
                if score_val[y] <= sx:
                    break  # since score_val is decreasing in y, no more improvement
                # Check if y is coprime to all z in chosen \ {x}
                ok = True
                if y != 1:
                    yf = prime_factors[y]
                    if yf:
                        for j in range(n):
                            if j == i:
                                continue
                            z = chosen[j]
                            if z != 1 and (yf & prime_factors[z]):
                                ok = False
                                break
                if ok:
                    gain = score_val[y] - sx
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, y)
        if best_swap:
            i, y = best_swap
            x = chosen[i]
            chosen_set.remove(x)
            chosen_set.add(y)
            chosen[i] = y
        return chosen, chosen_set, best_gain

    # 2-swap local search: remove two, add two
    def two_swap_improve(chosen, chosen_set):
        best_gain = 0
        best_swap = None
        n = len(chosen)
        for i in range(n):
            for j in range(i + 1, n):
                x1, x2 = chosen[i], chosen[j]
                s12 = score_val[x1] + score_val[x2]
                # Build list of remaining indices
                remaining_idx = [k for k in range(n) if k != i and k != j]
                # Find best pair (y1, y2) not in chosen
                # y1 > y2, both coprime to remaining and to each other
                # Iterate y1 from 511 down
                for y1 in range(511, 1, -1):
                    if y1 in chosen_set:
                        continue
                    if score_val[y1] + score_val[y1 - 1] <= s12:
                        break  # even best possible y2=y1-1 can't beat
                    # Check y1 coprime to remaining
                    ok1 = True
                    if y1 != 1:
                        y1f = prime_factors[y1]
                        if y1f:
                            for k in remaining_idx:
                                z = chosen[k]
                                if z != 1 and (y1f & prime_factors[z]):
                                    ok1 = False
                                    break
                    if not ok1:
                        continue
                    # Now find best y2 < y1, not in chosen, coprime to remaining and y1
                    for y2 in range(y1 - 1, 1, -1):
                        if y2 in chosen_set:
                            continue
                        gain = score_val[y1] + score_val[y2] - s12
                        if gain <= best_gain:
                            break  # no more improvement possible for this y1
                        # Check y2 coprime to remaining
                        ok2 = True
                        if y2 != 1:
                            y2f = prime_factors[y2]
                            if y2f:
                                for k in remaining_idx:
                                    z = chosen[k]
                                    if z != 1 and (y2f & prime_factors[z]):
                                        ok2 = False
                                        break
                        if not ok2:
                            continue
                        # Check y2 coprime to y1
                        if y2 != 1 and y1 != 1:
                            if prime_factors[y2] & prime_factors[y1]:
                                continue
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, j, y1, y2)
        if best_swap:
            i, j, y1, y2 = best_swap
            x1, x2 = chosen[i], chosen[j]
            chosen_set.discard(x1)
            chosen_set.discard(x2)
            chosen_set.add(y1)
            chosen_set.add(y2)
            chosen[i] = y1
            chosen[j] = y2
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

    # Simulated annealing with factor-based checks
    def simulated_annealing(chosen, chosen_set, max_iters=5000):
        T = 0.01
        T_min = 1e-6
        cooling = 0.999
        best_chosen = list(chosen)
        best_set = set(chosen_set)
        best_score = get_score(chosen)
        cur_score = best_score

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if not chosen:
                break
            x = random.choice(chosen)
            sx = score_val[x]
            found = False
            for _ in range(20):
                y = random.randint(1, 511)
                if y in chosen_set:
                    continue
                # Check y coprime to chosen \ {x}
                ok = True
                if y != 1:
                    yf = prime_factors[y]
                    if yf:
                        for z in chosen:
                            if z == x:
                                continue
                            if z != 1 and (yf & prime_factors[z]):
                                ok = False
                                break
                if ok:
                    gain = score_val[y] - sx
                    if gain >= 0 or random.random() < math.exp(gain / T):
                        chosen_set.remove(x)
                        chosen_set.add(y)
                        chosen[chosen.index(x)] = y
                        cur_score += gain
                        if cur_score > best_score:
                            best_score = cur_score
                            best_chosen = list(chosen)
                            best_set = set(chosen_set)
                        found = True
                        break
            if not found:
                # Try adding a new element
                for _ in range(10):
                    y = random.randint(1, 511)
                    if y in chosen_set:
                        continue
                    if is_coprime_to_set(y, chosen_set):
                        chosen_set.add(y)
                        chosen.append(y)
                        cur_score += score_val[y]
                        if cur_score > best_score:
                            best_score = cur_score
                            best_chosen = list(chosen)
                            best_set = set(chosen_set)
                        found = True
                        break
            if not found and len(chosen) > 1:
                x = random.choice(chosen)
                chosen_set.remove(x)
                chosen.remove(x)
                cur_score -= score_val[x]
            T = max(T * cooling, T_min)

        return best_chosen, best_set, best_score

    # Main algorithm
    chosen, chosen_set = greedy_init()
    chosen, chosen_set = local_search(chosen, chosen_set)
    best_score = get_score(chosen)
    best_chosen = list(chosen)
    best_set = set(chosen_set)

    random.seed(42)
    while time.time() < deadline - 15:
        # Perturb the current best solution
        perturbed_chosen = list(best_chosen)
        perturbed_set = set(best_set)

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

        perturbed_chosen, perturbed_set = local_search(perturbed_chosen, perturbed_set)
        ps = get_score(perturbed_chosen)

        if ps > best_score:
            best_score = ps
            best_chosen = list(perturbed_chosen)
            best_set = set(perturbed_set)

        if time.time() < deadline - 10:
            sa_chosen, sa_set, sa_score = simulated_annealing(list(best_chosen), set(best_set), max_iters=2000)
            if sa_score > best_score:
                best_score = sa_score
                best_chosen = list(sa_chosen)
                best_set = set(sa_set)

    # Final polish: multiple passes of local search
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        best_chosen, best_set = local_search(best_chosen, best_set)

    return list(best_set)
