import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    n = 256

    # Precompute canonical 3-APs and 4-APs (0-indexed internally)
    canonical_aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                canonical_aps3.append((a, b, c))

    canonical_aps4 = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                canonical_aps4.append((a, b, c, e))

    # Precompute partners for fast delta computation
    partners3 = [[] for _ in range(n)]
    for a, b, c in canonical_aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    partners4 = [[] for _ in range(n)]
    for a, b, c, d in canonical_aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    # Degree scores for greedy ordering
    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    def count_3aps_mask(A):
        count = 0
        for a, b, c in canonical_aps3:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1:
                count += 1
        return count

    def gain_add(A, v):
        gain = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                gain += 1
        return gain

    def loss_remove(A, v):
        loss = 0
        for u, w in partners3[v]:
            if ((A >> u) & 1) and ((A >> w) & 1):
                loss += 1
        return loss

    def can_add(A, v):
        for u, w, x in partners4[v]:
            if ((A >> u) & 1) and ((A >> w) & 1) and ((A >> x) & 1):
                return False
        return True

    def greedy_construct(order):
        A = 0
        for v in order:
            if can_add(A, v):
                A |= (1 << v)
        return A

    def hill_climb(A, time_limit, max_iters=1000):
        current_score = count_3aps_mask(A)
        best_score = current_score
        best_A = A
        iters = 0

        improved = True
        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            best_delta = 0
            best_move = None
            iters += 1

            # Try all adds
            for v in range(n):
                if (A >> v) & 1:
                    continue
                if not can_add(A, v):
                    continue
                g = gain_add(A, v)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', v, -1)

            # Try all removes
            for v in range(n):
                if not ((A >> v) & 1):
                    continue
                l = loss_remove(A, v)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', v, -1)

            # Try all swaps (sampled for speed)
            in_s = [v for v in range(n) if (A >> v) & 1]
            not_s = [v for v in range(n) if not ((A >> v) & 1)]
            
            # Sample a subset of swaps to keep it fast
            sample_size = min(len(in_s) * len(not_s), 500)
            if len(in_s) > 0 and len(not_s) > 0:
                for _ in range(sample_size):
                    u = random.choice(in_s)
                    v = random.choice(not_s)
                    if u == v:
                        continue

                    # Check if swap is valid
                    A_wo_u = A & ~(1 << u)
                    if not can_add(A_wo_u, v):
                        continue

                    loss_u = loss_remove(A, u)
                    gain_v = gain_add(A_wo_u, v)
                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)

            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    A |= (1 << v)
                elif mtype == 'remove':
                    A &= ~(1 << v)
                else:  # swap
                    A &= ~(1 << u)
                    A |= (1 << v)
                current_score += best_delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
                improved = True

        return best_A, best_score

    def sa_search(A, sc, max_iters=1000, time_limit=None):
        T = 5.0
        T_min = 0.1
        cool = 0.98
        for _ in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            r = random.random()
            if r < 0.40:
                cands = [x for x in range(n) if not ((A >> x) & 1) and can_add(A, x)]
                if not cands:
                    continue
                x = random.choice(cands)
                A_new = A | (1 << x)
                delta = gain_add(A, x)
            elif r < 0.60:
                in_s = [x for x in range(n) if (A >> x) & 1]
                if not in_s:
                    continue
                y = random.choice(in_s)
                A_new = A & ~(1 << y)
                delta = -loss_remove(A, y)
            elif r < 0.85:
                in_s = [x for x in range(n) if (A >> x) & 1]
                not_s = [x for x in range(n) if not ((A >> x) & 1)]
                if not in_s or not not_s:
                    continue
                y = random.choice(in_s)
                A_wo = A & ~(1 << y)
                valid = [x for x in not_s if can_add(A_wo, x)]
                if not valid:
                    continue
                x = random.choice(valid)
                A_new = A_wo | (1 << x)
                delta = gain_add(A_wo, x) - loss_remove(A, y)
            else:
                in_s = [x for x in range(n) if (A >> x) & 1]
                not_s = [x for x in range(n) if not ((A >> x) & 1)]
                if len(in_s) < 2 or len(not_s) < 2:
                    continue
                y1, y2 = random.sample(in_s, 2)
                A_wo = A & ~(1 << y1) & ~(1 << y2)
                valid = [x for x in not_s if can_add(A_wo, x)]
                if len(valid) < 2:
                    continue
                x1, x2 = random.sample(valid, 2)
                A_new = A_wo | (1 << x1) | (1 << x2)
                delta = gain_add(x1, A_wo) + gain_add(x2, A_wo) - loss_remove(A, y1) - loss_remove(A, y2)

            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                A = A_new
                sc += delta
            T = max(T_min, T * cool)

        # Final polish
        for x in range(n):
            if time_limit is not None and time.time() > time_limit:
                break
            if (A >> x) & 1:
                continue
            if not can_add(A, x):
                continue
            g = gain_add(A, x)
            if g > 0:
                A |= (1 << x)
                sc += g
        return A, sc

    def repair(A):
        # Remove elements to break any 4-APs
        for ap in canonical_aps4:
            if all((A >> e) & 1 for e in ap):
                # Remove the element with the highest loss
                best_loss = -1
                best_e = ap[0]
                for e in ap:
                    l = loss_remove(A, e)
                    if l > best_loss:
                        best_loss = l
                        best_e = e
                A &= ~(1 << best_e)
        return A

    random.seed(42)
    best_A = 0
    best_score = 0

    # Strategy 1: Multiple greedy constructions with different orderings + hill climb
    orderings = []
    orderings.append(sorted(range(n), key=lambda v: (-deg3[v], v)))
    orderings.append(sorted(range(n), key=lambda v: (-(deg3[v] - deg4[v]), v)))
    orderings.append(list(range(n)))
    orderings.append(list(range(n - 1, -1, -1)))
    for s in range(6):
        rng = random.Random(100 + s)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)

    for i, order in enumerate(orderings):
        if time.time() > deadline - 10:
            break
        A = greedy_construct(order)
        sc = count_3aps_mask(A)
        # Hill climb with time budget
        remaining = deadline - time.time() - 10
        time_per_hill = remaining / (len(orderings) - i)
        A, sc = hill_climb(A, time.time() + max(0.1, time_per_hill))
        if sc > best_score:
            best_score = sc
            best_A = A

    # Strategy 2: SA from best
    if time.time() < deadline - 15 and best_A:
        A = best_A
        sc = best_score
        A, sc = sa_search(A, sc, max_iters=800, time_limit=deadline - 10)
        A, sc = hill_climb(A, deadline - 5)
        if sc > best_score:
            best_score = sc
            best_A = A

    # Strategy 3: Random restarts with SA
    for r in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        A = greedy_construct(order)
        sc = count_3aps_mask(A)
        A, sc = sa_search(A, sc, max_iters=200, time_limit=deadline - 8)
        A, sc = hill_climb(A, deadline - 5)
        if sc > best_score:
            best_score = sc
            best_A = A

    # Strategy 4: Random density + repair + hill climb
    for p in [0.4, 0.5, 0.6, 0.7]:
        if time.time() > deadline - 8:
            break
        A = 0
        for x in range(n):
            if random.random() < p:
                A |= (1 << x)
        A = repair(A)
        sc = count_3aps_mask(A)
        A, sc = hill_climb(A, deadline - 5)
        if sc > best_score:
            best_score = sc
            best_A = A

    # Final polish
    if time.time() < deadline - 2 and best_A:
        best_A, best_score = hill_climb(best_A, deadline - 0.5)

    # Convert to 1-indexed list
    result = [x + 1 for x in range(n) if (best_A >> x) & 1]
    if not result:
        result = [1]

    return result
