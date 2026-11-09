import time
import math


def construct():
    t0 = time.time()
    DEADLINE = 100.0
    MIN_M, MAX_M = 20, 200

    moduli = list(range(MIN_M, MAX_M + 1))
    n_m = len(moduli)

    # Precompute incompatibility: gcd == 1 means always conflict
    incomp = [0] * n_m
    for i in range(n_m):
        for j in range(i + 1, n_m):
            if math.gcd(moduli[i], moduli[j]) == 1:
                incomp[i] |= (1 << j)
                incomp[j] |= (1 << i)

    weights = [1.0 / m for m in moduli]
    order = sorted(range(n_m), key=lambda i: weights[i], reverse=True)

    # Greedy initial solution
    greedy_set = set()
    remaining = (1 << n_m) - 1
    for i in order:
        if (remaining >> i) & 1:
            greedy_set.add(i)
            remaining &= ~incomp[i]
            remaining &= ~(1 << i)
    best_weight = sum(weights[i] for i in greedy_set)
    best_set = set(greedy_set)

    # B&B for maximum weight set with no two coprime
    def bbsolve(vertex_set, current_weight, current_set):
        nonlocal best_weight, best_set
        if time.time() > DEADLINE:
            return
        if not vertex_set:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return

        # Upper bound: sum of all weights in vertex_set
        ub = current_weight
        for i in range(n_m):
            if (vertex_set >> i) & 1:
                ub += weights[i]
        if ub <= best_weight:
            return

        # Pick highest weight vertex
        v = None
        for i in order:
            if (vertex_set >> i) & 1:
                v = i
                break
        if v is None:
            return

        # Branch 1: include v
        new_set = vertex_set & ~incomp[v] & ~(1 << v)
        bbsolve(new_set, current_weight + weights[v], current_set | {v})

        # Branch 2: exclude v
        new_set2 = vertex_set & ~(1 << v)
        bbsolve(new_set2, current_weight, current_set)

    bbsolve((1 << n_m) - 1, 0.0, set())

    # Assign residues to selected moduli
    selected = sorted(best_set)
    assign = {}
    for idx in selected:
        m = moduli[idx]
        placed = False
        for r in range(m):
            ok = True
            for idx2 in assign:
                m2 = moduli[idx2]
                g = math.gcd(m, m2)
                if r % g == assign[idx2] % g:
                    ok = False
                    break
            if ok:
                assign[idx] = r
                placed = True
                break
        if not placed:
            best_set.discard(idx)

    result = []
    for idx in sorted(best_set):
        m = moduli[idx]
        r = assign[idx]
        result.append([m, r])
    return result
