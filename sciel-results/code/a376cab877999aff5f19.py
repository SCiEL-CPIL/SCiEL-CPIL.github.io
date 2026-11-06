import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    N = 512
    MAX_A = 511

    # 1. Precomputation
    # Prime factors
    prime_factors = [set() for _ in range(MAX_A + 1)]
    for p in range(2, MAX_A + 1):
        if not prime_factors[p]:
            for multiple in range(p, MAX_A + 1, p):
                prime_factors[multiple].add(p)

    # Weights
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)

    # Conflict masks
    # conflict_mask[a] has bit b set if gcd(a, b) > 1
    conflict_mask = [0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        mask = 0
        for b in range(1, MAX_A + 1):
            if a == b:
                continue
            if prime_factors[a] & prime_factors[b]:
                mask |= (1 << b)
        conflict_mask[a] = mask

    def is_coprime_to_set(a, chosen_set):
        if a == 1:
            return True
        af = prime_factors[a]
        if not af:
            return True
        for b in chosen_set:
            if b == 1:
                continue
            if af & prime_factors[b]:
                return False
        return True

    def get_score(chosen):
        return sum(weights[a] for a in chosen)

    def greedy_init(order=None):
        if order is None:
            order = list(range(MAX_A, 0, -1))
        chosen = []
        chosen_set = set()
        for a in order:
            if is_coprime_to_set(a, chosen_set):
                chosen.append(a)
                chosen_set.add(a)
        return chosen, chosen_set

    def local_search(chosen, chosen_set):
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # 1-swap: replace one element with a better one
            for i in range(len(chosen)):
                x = chosen[i]
                sx = weights[x]
                for y in range(MAX_A, 1, -1):
                    if y in chosen_set:
                        continue
                    if weights[y] <= sx:
                        continue
                    # Check if y is coprime to all in chosen except x
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
                        chosen_set.remove(x)
                        chosen_set.add(y)
                        chosen[i] = y
                        improved = True
                        break
                if improved:
                    break
            # 2-swap: replace two elements with two better ones
            if not improved and len(chosen) >= 2:
                n = len(chosen)
                for i in range(n):
                    for j in range(i + 1, n):
                        x1, x2 = chosen[i], chosen[j]
                        s12 = weights[x1] + weights[x2]
                        best_pair = None
                        best_gain = 0
                        for y1 in range(MAX_A, 1, -1):
                            if y1 in chosen_set:
                                continue
                            for y2 in range(y1 - 1, 1, -1):
                                if y2 in chosen_set:
                                    continue
                                if y1 == y2:
                                    continue
                                gain = weights[y1] + weights[y2] - s12
                                if gain <= best_gain:
                                    continue
                                # Check coprimality
                                ok = True
                                for z in chosen:
                                    if z == x1 or z == x2:
                                        continue
                                    if z == 1:
                                        continue
                                    zf = prime_factors[z]
                                    if zf:
                                        if (prime_factors[y1] & zf) or (prime_factors[y2] & zf):
                                            ok = False
                                            break
                                if not ok:
                                    continue
                                # Check y1 and y2 are coprime
                                if y1 != 1 and y2 != 1 and (prime_factors[y1] & prime_factors[y2]):
                                    continue
                                best_pair = (y1, y2)
                                best_gain = gain
                        if best_pair:
                            y1, y2 = best_pair
                            chosen_set.remove(x1)
                            chosen_set.remove(x2)
                            chosen_set.add(y1)
                            chosen_set.add(y2)
                            # Replace in list
                            chosen[i] = y1
                            chosen[j] = y2
                            improved = True
                            break
                    if improved:
                        break
        return chosen, chosen_set

    # Multi-start with different orderings
    best_chosen = None
    best_set = None
    best_score = 0.0

    # Start with standard greedy
    chosen, chosen_set = greedy_init()
    chosen, chosen_set = local_search(chosen, chosen_set)
    score = get_score(chosen)
    if score > best_score:
        best_score = score
        best_chosen = chosen[:]
        best_set = chosen_set.copy()

    # Try multiple random orderings
    random.seed(42)
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        # Create a random ordering with bias towards high values
        order = list(range(1, MAX_A + 1))
        random.shuffle(order)
        # Sort with some randomness but bias high values
        order.sort(key=lambda x: random.random() * 0.3 + weights[x] * 0.7, reverse=True)
        chosen, chosen_set = greedy_init(order)
        chosen, chosen_set = local_search(chosen, chosen_set)
        score = get_score(chosen)
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
            best_set = chosen_set.copy()

    # Final local search on best solution
    if best_chosen is not None:
        best_chosen, best_set = local_search(best_chosen, best_set)
        best_score = get_score(best_chosen)

    return list(best_set)
