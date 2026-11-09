import time
import random

# ---- Precompute all nonconstant 4-term APs (0-based indices) ----
N = 128
ap_members = []          # list of 4-tuples of indices
aps_of = [[] for _ in range(N)]  # for each position, list of (ap_idx, slot)
for x in range(1, N + 1):
    maxd = (N - x) // 3
    for d in range(1, maxd + 1 + 1):
        if x + 3 * d > N:
            break
        i0 = x - 1
        i1 = i0 + d
        i2 = i0 + 2 * d
        i3 = i0 + 3 * d
        ap = (i0, i1, i2, i3)
        idx = len(ap_members)
        ap_members.append(ap)
        for slot, m in enumerate(ap):
            aps_of[m].append((idx, slot))

NUM_APS = len(ap_members)


def distinct4(a, b, c, d):
    s = 1
    if b != a:
        s += 1
    if c != a and c != b:
        s += 1
    if d != a and d != b and d != c:
        s += 1
    return s


def compute_V(a):
    V = 0
    for (i0, i1, i2, i3) in ap_members:
        if distinct4(a[i0], a[i1], a[i2], a[i3]) <= 2:
            V += 1
    return V


def distinct_count(color_count):
    return sum(1 for c in color_count if c > 0)


def eval_dV(a, i, o, n):
    dV = 0
    for (ap, slot) in aps_of[i]:
        m0, m1, m2, m3 = ap_members[ap]
        before = distinct4(a[m0], a[m1], a[m2], a[m3])
        if slot == 0:
            after = distinct4(n, a[m1], a[m2], a[m3])
        elif slot == 1:
            after = distinct4(a[m0], n, a[m2], a[m3])
        elif slot == 2:
            after = distinct4(a[m0], a[m1], n, a[m3])
        else:
            after = distinct4(a[m0], a[m1], a[m2], n)
        if (after <= 2) != (before <= 2):
            dV += 1 if after <= 2 else -1
    return dV


def local_search(init, K, deadline):
    a = init[:]
    color_count = [0] * K
    for v in a:
        color_count[v] += 1
    V = compute_V(a)

    # ---- Phase 1: minimize V ----
    kicks = 0
    plateau = 0
    while V > 0 and time.time() < deadline:
        improved = False
        for _ in range(400):
            if time.time() >= deadline:
                break
            i = random.randrange(N)
            o = a[i]
            best_dV = 0
            best_n = o
            for n in range(K):
                if n == o:
                    continue
                dV = eval_dV(a, i, o, n)
                if dV < best_dV:
                    best_dV = dV
                    best_n = n
            if best_dV < 0:
                color_count[o] -= 1
                color_count[best_n] += 1
                a[i] = best_n
                V += best_dV
                improved = True
        if V == 0:
            break
        if not improved:
            plateau += 1
            if plateau > 4:
                # kick: random perturbations to escape local minimum
                for _ in range(20):
                    i = random.randrange(N)
                    n = random.randrange(K)
                    o = a[i]
                    if n == o:
                        continue
                    dV = eval_dV(a, i, o, n)
                    color_count[o] -= 1
                    color_count[n] += 1
                    a[i] = n
                    V += dV
                plateau = 0
                kicks += 1
                if kicks > 40:
                    break
        else:
            plateau = 0

    # ---- Phase 2: minimize distinct colors while V==0 ----
    if V == 0:
        for _ in range(300):
            if time.time() >= deadline:
                break
            improved = False
            for _ in range(300):
                if time.time() >= deadline:
                    break
                i = random.randrange(N)
                o = a[i]
                co = color_count[o]
                best_dd = 0
                best_n = o
                for n in range(K):
                    if n == o:
                        continue
                    dV = eval_dV(a, i, o, n)
                    if dV != 0:
                        continue
                    cn = color_count[n]
                    dd = 0
                    if co == 1:
                        dd -= 1
                    if cn == 0:
                        dd += 1
                    if dd < best_dd:
                        best_dd = dd
                        best_n = n
                if best_dd < 0:
                    color_count[o] -= 1
                    color_count[best_n] += 1
                    a[i] = best_n
                    improved = True
            if not improved:
                break

    # ---- Merge pass: try merging colors to reduce distinct count ----
    if V == 0:
        for _ in range(50):
            if time.time() >= deadline:
                break
            used = [c for c in range(K) if color_count[c] > 0]
            merged = False
            for A in used:
                if color_count[A] == 0:
                    continue
                for B in used:
                    if B == A or color_count[B] == 0:
                        continue
                    # try merge A into B
                    changed = []
                    for i in range(N):
                        if a[i] == A:
                            changed.append(i)
                    for i in changed:
                        a[i] = B
                    if compute_V(a) == 0:
                        for i in changed:
                            color_count[A] -= 1
                            color_count[B] += 1
                        merged = True
                        break
                    else:
                        for i in changed:
                            a[i] = A
                if merged:
                    break
            if not merged:
                break

    return a, V, distinct_count(color_count)


def construct():
    start = time.time()
    deadline = start + 70.0  # safety margin before 90s kill

    best = None  # (distinct, colors_list)

    init_strategies = [
        lambda K: [random.randrange(K) for _ in range(N)],
        lambda K: [i % K for i in range(N)],
        lambda K: [random.randrange(3) for _ in range(N)] if K >= 3 else [random.randrange(K) for _ in range(N)],
        lambda K: [(i // 4) % K for i in range(N)],
    ]

    K_choices = [4, 5, 6, 8, 5, 6, 4, 6]

    restart = 0
    while time.time() < deadline - 3.0:
        K = K_choices[restart % len(K_choices)]
        init = init_strategies[restart % len(init_strategies)](K)
        a, V, dc = local_search(init, K, deadline - 3.0)
        if V == 0:
            if best is None or dc < best[0]:
                best = (dc, a[:])
        restart += 1

    if best is not None:
        return best[1]

    # Fallback: trivially valid (each position unique color)
    return list(range(N))
