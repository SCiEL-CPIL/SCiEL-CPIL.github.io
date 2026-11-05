import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80

    N = 256
    squares = set(s * s for s in range(2, 23) if s * s <= 512)

    # Build conflict graph
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if (a + b) in squares:
                conflict[a].add(b)
                if a != b:
                    conflict[b].add(a)

    def greedy(order):
        selected = set()
        for a in order:
            if not (conflict[a] & selected):
                selected.add(a)
        return selected

    def local_search(selected, max_time=1.0):
        selected = set(selected)
        start = time.time()
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            # Try adding
            for a in range(1, N + 1):
                if a not in selected and not (conflict[a] & selected):
                    selected.add(a)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps (remove one, add one)
            for a in range(1, N + 1):
                if a not in selected:
                    conf = conflict[a] & selected
                    if len(conf) == 1:
                        b = next(iter(conf))
                        selected.remove(b)
                        selected.add(a)
                        improved = True
                        break
                    elif len(conf) == 0:
                        selected.add(a)
                        improved = True
                        break
        return selected

    best_set = set()
    best_size = 0

    # Multiple random greedy + local search
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy(order)
        s = local_search(s, max_time=0.3)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)

    # Simulated annealing
    if time.time() < deadline - 10:
        sa_start = time.time()
        selected = set(best_set)
        temp = 1.0
        cooling = 0.99995
        best_sa = set(selected)
        best_sa_size = len(selected)

        while time.time() - sa_start < 40 and temp > 0.005:
            r = random.random()
            if r < 0.4:
                candidates = [a for a in range(1, N + 1) if a not in selected]
                if candidates:
                    a = random.choice(candidates)
                    if not (conflict[a] & selected):
                        selected.add(a)
                        delta = 1
                        if len(selected) > best_sa_size:
                            best_sa = set(selected)
                            best_sa_size = len(selected)
                    else:
                        temp *= cooling
                        continue
            elif r < 0.7:
                if not selected:
                    temp *= cooling
                    continue
                a = random.choice(list(selected))
                selected.remove(a)
                delta = -1
            else:
                if not selected:
                    temp *= cooling
                    continue
                a = random.choice(list(selected))
                selected.remove(a)
                candidates = [x for x in range(1, N + 1) if x not in selected]
                if candidates:
                    b = random.choice(candidates)
                    if not (conflict[b] & selected):
                        selected.add(b)
                        delta = 0
                    else:
                        selected.add(a)
                        temp *= cooling
                        continue
                else:
                    selected.add(a)
                    temp *= cooling
                    continue

            if delta >= 0 or random.random() < math.exp(delta / max(temp, 0.001)):
                pass
            else:
                if r < 0.4:
                    if a in selected:
                        selected.remove(a)
                elif r < 0.7:
                    selected.add(a)

            temp *= cooling

        if best_sa_size > best_size:
            best_set = best_sa
            best_size = best_sa_size

    # Final local search
    if time.time() < deadline - 2:
        best_set = local_search(best_set, max_time=5.0)

    return sorted(best_set)
