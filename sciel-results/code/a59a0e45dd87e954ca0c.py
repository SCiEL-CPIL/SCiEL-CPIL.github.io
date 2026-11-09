import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 118.0  # safety margin before the 150s kill

    N = 51975  # 3^3 * 5^2 * 7 * 11

    # All divisors > 1 of N
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    all_divisors_set = set(divisors)

    def omega(n):
        c = 0
        t = n
        for p in (3, 5, 7, 11):
            while t % p == 0:
                c += 1
                t //= p
        return c

    # ---- coverage machinery (counts / uncovered), vectorized ----
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N

    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        dec = np.sum(counts[idx] == 0)
        counts[idx] += 1
        uncovered -= dec

    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        inc = np.sum(counts[idx] == 1)
        counts[idx] -= 1
        uncovered += inc

    def find_best_residue(m, counts_arr):
        Z = (counts_arr == 0).astype(np.int32)
        best_r = 0
        best_gain = -1
        max_possible = N // m
        for r in range(m):
            gain = np.sum(Z[r::m])
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if best_gain == max_possible:
                    break
        return best_r, best_gain

    def reset_counts():
        nonlocal counts, uncovered
        counts = np.zeros(N, dtype=np.int32)
        uncovered = N

    # ---- greedy with a given ordering ----
    def greedy(order):
        reset_counts()
        classes = []
        used = set()
        for m in order:
            if m in used:
                continue
            best_r, best_gain = find_best_residue(m, counts)
            if best_gain > 0:
                classes.append([m, best_r])
                used.add(m)
                update_add(m, best_r)
        return classes, uncovered

    # ---- SA over the current classes ----
    def sa(classes, T0, T_min, cooling, time_budget):
        nonlocal counts, uncovered
        reset_counts()
        for m, r in classes:
            update_add(m, r)
        current_uncovered = uncovered
        best_uncovered = uncovered
        best_classes = [c.copy() for c in classes]
        T = T0
        t_end = start_time + time_budget
        while time.time() < t_end and T >= T_min:
            move_type = random.random()
            if move_type < 0.5 and classes:
                i = random.randint(0, len(classes) - 1)
                m, r = classes[i]
                new_r = random.randint(0, m - 1)
                if new_r == r:
                    continue
                old_r = r
                update_remove(m, old_r)
                update_add(m, new_r)
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                classes[i][1] = new_r
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        update_remove(m, new_r)
                        update_add(m, old_r)
                        classes[i][1] = old_r
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]
            elif move_type < 0.7 and len(classes) >= 2:
                i, j = random.sample(range(len(classes)), 2)
                m_i, r_i = classes[i]
                m_j, r_j = classes[j]
                update_remove(m_i, r_i)
                update_remove(m_j, r_j)
                update_add(m_i, r_j)
                update_add(m_j, r_i)
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                classes[i][1] = r_j
                classes[j][1] = r_i
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        update_remove(m_i, r_j)
                        update_remove(m_j, r_i)
                        update_add(m_i, r_i)
                        update_add(m_j, r_j)
                        classes[i][1] = r_i
                        classes[j][1] = r_j
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]
            elif move_type < 0.85:
                unused = [m for m in divisors if m not in used_moduli_of(classes)]
                if unused:
                    m = random.choice(unused)
                    r = random.randint(0, m - 1)
                    update_add(m, r)
                    classes.append([m, r])
                    delta = uncovered - current_uncovered
                    current_uncovered = uncovered
                    if delta > 0:
                        if random.random() < np.exp(-delta / T):
                            pass
                        else:
                            update_remove(m, r)
                            classes.pop()
                            current_uncovered = uncovered
                    else:
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_classes = [c.copy() for c in classes]
            else:
                if classes:
                    i = random.randint(0, len(classes) - 1)
                    m, r = classes[i]
                    update_remove(m, r)
                    classes.pop(i)
                    delta = uncovered - current_uncovered
                    current_uncovered = uncovered
                    if delta > 0:
                        if random.random() < np.exp(-delta / T):
                            pass
                        else:
                            classes.insert(i, [m, r])
                            update_add(m, r)
                            current_uncovered = uncovered
                    else:
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_classes = [c.copy() for c in classes]
            T *= cooling
        return best_uncovered, best_classes

    def used_moduli_of(classes):
        return set(m for m, _ in classes)

    # ---- smooth-surrogate polish: log-sum-exp soft-min over residues ----
    # c_r = sum_x 1/(1+counts_x) for residue r of modulus m.
    # Minimizing the soft-min t = logsumexp(-c_r) pushes all residues toward
    # high coverage; the argmax of c_r is the exact best residue, so the
    # surrogate only guides which residue to try, while the exact check decides.
    def smooth_polish(classes, time_budget):
        nonlocal counts, uncovered
        reset_counts()
        for m, r in classes:
            update_add(m, r)
        t_end = start_time + time_budget
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for i in range(len(classes)):
                if time.time() > t_end:
                    break
                m, r = classes[i]
                update_remove(m, r)
                # soft surrogate over all residues
                inv = 1.0 / (1.0 + counts.astype(np.float64))
                c = np.array([np.sum(inv[rr::m]) for rr in range(m)])
                # logsumexp(-c)
                mc = np.max(-c)
                soft = mc + np.log(np.sum(np.exp(-c - mc)))
                best_r = int(np.argmax(c))
                # exact gain for the surrogate-chosen residue
                Z = (counts == 0).astype(np.int32)
                gain_best = int(np.sum(Z[best_r::m]))
                # exact gain for current residue
                gain_cur = int(np.sum(Z[r::m]))
                if best_r != r and gain_best > gain_cur:
                    update_add(m, best_r)
                    classes[i][1] = best_r
                    improved = True
                    break
                else:
                    update_add(m, r)
        return classes, uncovered

    # ---- exact local polish (fast exact argmax) ----
    def local_polish(classes, time_budget):
        nonlocal counts, uncovered
        reset_counts()
        for m, r in classes:
            update_add(m, r)
        t_end = start_time + time_budget
        improved = True
        while improved and time.time() < t_end:
            improved = False
            for i in range(len(classes)):
                if time.time() > t_end:
                    break
                m, r = classes[i]
                update_remove(m, r)
                best_r, _ = find_best_residue(m, counts)
                if best_r != r:
                    update_add(m, best_r)
                    classes[i][1] = best_r
                    improved = True
                    break
                else:
                    update_add(m, r)
        return classes, uncovered

    # ---- build candidate orderings ----
    orderings = []
    orderings.append(divisors.copy())
    orderings.append(divisors[::-1].copy())
    orderings.append(sorted(divisors, key=lambda m: (N // m, m)))
    orderings.append(sorted(divisors, key=lambda m: (-omega(m), m)))
    for _ in range(6):
        perm = divisors.copy()
        random.shuffle(perm)
        orderings.append(perm)

    best_uncovered = N
    best_classes = [[3, 0]]

    # Phase 1: multi-ordering greedy, keep best
    for order in orderings:
        if time.time() > deadline - 50:
            break
        classes, sc = greedy(order)
        if sc < best_uncovered:
            best_uncovered = sc
            best_classes = [c.copy() for c in classes]

    # Phase 2: SA from the best greedy, then smooth polish, then local polish
    if time.time() < deadline - 25:
        sa_classes = [c.copy() for c in best_classes]
        sa_uncovered, sa_classes = sa(sa_classes, 10.0, 0.01, 0.999, deadline - 45)
        if sa_uncovered < best_uncovered:
            best_uncovered = sa_uncovered
            best_classes = [c.copy() for c in sa_classes]

    # Phase 3: smooth-surrogate polish (guides exact moves via soft-min)
    if time.time() < deadline - 12:
        pc = [c.copy() for c in best_classes]
        pc, sc = smooth_polish(pc, deadline - 15)
        if sc < best_uncovered:
            best_uncovered = sc
            best_classes = [c.copy() for c in pc]

    # Phase 4: exact local polish (final high-precision pass)
    if time.time() < deadline - 2:
        pc = [c.copy() for c in best_classes]
        pc, sc = local_polish(pc, deadline - 2)
        if sc < best_uncovered:
            best_uncovered = sc
            best_classes = [c.copy() for c in pc]

    return best_classes
