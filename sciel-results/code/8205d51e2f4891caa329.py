import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    # ---------- Precomputation (0-indexed) ----------
    aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            c = a + 2 * d
            if c < n:
                aps3.append((a, a + d, c))

    aps4 = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            e = a + 3 * d
            if e < n:
                aps4.append((a, a + d, a + 2 * d, e))

    # partners3[x] = list of (u, w) such that {x, u, w} forms a 3-AP
    partners3 = [[] for _ in range(n)]
    for a, b, c in aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    # partners4[x] = list of (u, v, w) such that {x, u, v, w} forms a 4-AP
    partners4 = [[] for _ in range(n)]
    for a, b, c, d in aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    deg3 = [len(partners3[v]) for v in range(n)]
    deg4 = [len(partners4[v]) for v in range(n)]

    # ---------- Core operations ----------
    def count_3aps(S):
        cnt = 0
        for a, b, c in aps3:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def can_add(S, x):
        for u, v, w in partners4[x]:
            if (S >> u) & 1 and (S >> v) & 1 and (S >> w) & 1:
                return False
        return True

    def count_4aps(S):
        cnt = 0
        for a, b, c, d in aps4:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def gain_add(S, x):
        g = 0
        for u, w in partners3[x]:
            if (S >> u) & 1 and (S >> w) & 1:
                g += 1
        return g

    def loss_remove(S, x):
        g = 0
        for u, w in partners3[x]:
            if (S >> u) & 1 and (S >> w) & 1:
                g += 1
        return g

    # ---------- QUBO-style penalty evaluation ----------
    PENALTY = 5000

    def qubo_value(S):
        """Objective: 3-AP count - PENALTY * 4-AP count."""
        cnt3 = 0
        for a, b, c in aps3:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt3 += 1
        cnt4 = 0
        for a, b, c, d in aps4:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt4 += 1
        return cnt3 - PENALTY * cnt4

    def qubo_delta_add(S, x):
        """Delta in qubo value from adding x."""
        g3 = gain_add(S, x)
        # Count 4-APs that would be completed
        g4 = 0
        for u, v, w in partners4[x]:
            if (S >> u) & 1 and (S >> v) & 1 and (S >> w) & 1:
                g4 += 1
        return g3 - PENALTY * g4

    def qubo_delta_remove(S, x):
        """Delta in qubo value from removing x."""
        l3 = loss_remove(S, x)
        # Count 4-APs that would be broken
        b4 = 0
        for u, v, w in partners4[x]:
            if (S >> u) & 1 and (S >> v) & 1 and (S >> w) & 1:
                b4 += 1
        return -l3 + PENALTY * b4

    # ---------- Greedy construction ----------
    def greedy_construct(order, use_score=True, min_size=15):
        S = 0
        for x in order:
            if not can_add(S, x):
                continue
            if use_score:
                g = gain_add(S, x)
                if g > 0 or bin(S).count('1') < min_size:
                    S |= (1 << x)
            else:
                S |= (1 << x)
        return S

    # ---------- Best-improving hill climb ----------
    def hill_climb(S, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None

            # Try all adds
            for x in range(n):
                if (S >> x) & 1:
                    continue
                if not can_add(S, x):
                    continue
                g = gain_add(S, x)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', x, -1)

            # Try all removes
            for x in range(n):
                if not ((S >> x) & 1):
                    continue
                l = loss_remove(S, x)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', x, -1)

            # Try all swaps
            in_set = [x for x in range(n) if (S >> x) & 1]
            not_in_set = [x for x in range(n) if not ((S >> x) & 1)]
            for u in in_set:
                S_wo_u = S & ~(1 << u)
                for v in not_in_set:
                    # Check validity of swap
                    valid = True
                    for w, x2, y in partners4[v]:
                        if ((S >> w) & 1) and ((S >> x2) & 1) and ((S >> y) & 1):
                            if w != u and x2 != u and y != u:
                                valid = False
                                break
                    if not valid:
                        continue
                    loss_u = 0
                    for w, x2 in partners3[u]:
                        if ((S >> w) & 1) and ((S >> x2) & 1) and w != v and x2 != v:
                            loss_u += 1
                    gain_v = 0
                    for w, x2 in partners3[v]:
                        if ((S >> w) & 1) and ((S >> x2) & 1) and w != u and x2 != u:
                            gain_v += 1
                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)

            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    S |= (1 << v)
                elif mtype == 'remove':
                    S &= ~(1 << v)
                else:
                    S &= ~(1 << u)
                    S |= (1 << v)
                current_score += best_delta
                improved = True

        return S, current_score

    # ---------- QUBO Simulated Annealing ----------
    def qubo_sa(S, time_limit):
        """SA with penalty-based 4-AP handling."""
        T = 200.0
        T_min = 0.5
        cooling = 0.995
        best_S = S
        best_val = qubo_value(S)

        while time.time() < time_limit:
            T = max(T_min, T * cooling)

            move_type = random.random()
            if move_type < 0.35:
                # Add
                not_in = [x for x in range(n) if not ((S >> x) & 1)]
                if not not_in:
                    continue
                x = random.choice(not_in)
                S_new = S | (1 << x)
                delta = qubo_delta_add(S, x)
            elif move_type < 0.65:
                # Remove
                in_set = [x for x in range(n) if (S >> x) & 1]
                if not in_set:
                    continue
                x = random.choice(in_set)
                S_new = S & ~(1 << x)
                delta = qubo_delta_remove(S, x)
            else:
                # Swap
                in_set = [x for x in range(n) if (S >> x) & 1]
                not_in = [x for x in range(n) if not ((S >> x) & 1)]
                if not in_set or not not_in:
                    continue
                u = random.choice(in_set)
                v = random.choice(not_in)
                S_wo_u = S & ~(1 << u)
                # Check if adding v is valid after removing u
                valid = True
                for w, x2, y in partners4[v]:
                    if ((S >> w) & 1) and ((S >> x2) & 1) and ((S >> y) & 1):
                        if w != u and x2 != u and y != u:
                            valid = False
                            break
                if not valid:
                    continue
                S_new = S_wo_u | (1 << v)
                # Compute delta
                loss_u = 0
                for w, x2 in partners3[u]:
                    if ((S >> w) & 1) and ((S >> x2) & 1) and w != v and x2 != v:
                        loss_u += 1
                gain_v = 0
                for w, x2 in partners3[v]:
                    if ((S >> w) & 1) and ((S >> x2) & 1) and w != u and x2 != u:
                        gain_v += 1
                # 4-AP changes
                b4_u = 0
                for w, x2, y in partners4[u]:
                    if ((S >> w) & 1) and ((S >> x2) & 1) and ((S >> y) & 1) and w != v and x2 != v and y != v:
                        b4_u += 1
                g4_v = 0
                for w, x2, y in partners4[v]:
                    if ((S >> w) & 1) and ((S >> x2) & 1) and ((S >> y) & 1) and w != u and x2 != u and y != u:
                        g4_v += 1
                delta = (gain_v - loss_u) - PENALTY * (g4_v - b4_u)

            if delta > 0 or random.random() < math.exp(delta / T):
                S = S_new
                val = qubo_value(S)
                if val > best_val:
                    best_val = val
                    best_S = S

        # Verify best_S has no 4-APs; if it does, repair
        if count_4aps(best_S) > 0:
            # Greedily remove to fix
            for ap in aps4:
                if all((best_S >> elem) & 1 for elem in ap):
                    min_loss = float('inf')
                    min_elem = ap[0]
                    for elem in ap:
                        l = loss_remove(best_S, elem)
                        if l < min_loss:
                            min_loss = l
                            min_elem = elem
                    best_S &= ~(1 << min_elem)

        return best_S

    # ---------- Prune from full ----------
    def prune_from_full():
        S = (1 << n) - 1
        for ap in aps4:
            if all((S >> elem) & 1 for elem in ap):
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(S, elem)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << min_elem)
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    random.seed(42)

    # Generate orderings
    orderings = []
    order1 = sorted(range(n), key=lambda v: (-deg3[v], v))
    orderings.append(order1)
    order2 = sorted(range(n), key=lambda v: (-(deg3[v] - deg4[v]), v))
    orderings.append(order2)
    orderings.append(list(range(n)))
    orderings.append(list(range(n - 1, -1, -1)))
    for s in range(10):
        rng = random.Random(200 + s)
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)

    # Phase 1: Greedy + hill climb for each ordering
    for i, order in enumerate(orderings):
        if time.time() > deadline - 15:
            break
        S = greedy_construct(order, use_score=True, min_size=15)
        sc = count_3aps(S)
        remaining = deadline - time.time() - 15
        time_per = max(0.5, remaining / max(1, len(orderings) - i))
        S, sc = hill_climb(S, sc, time.time() + time_per)
        if sc > best_score:
            best_score = sc
            best_S = S

    # Phase 2: Prune from full
    if time.time() < deadline - 15:
        S_p = prune_from_full()
        sc_p = count_3aps(S_p)
        S_p, sc_p = hill_climb(S_p, sc_p, deadline - 10)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Phase 3: QUBO SA from best
    if time.time() < deadline - 10 and best_S:
        S_sa = qubo_sa(int(best_S), deadline - 8)
        sc_sa = count_3aps(S_sa)
        # Verify no 4-APs
        if count_4aps(S_sa) == 0:
            S_sa, sc_sa = hill_climb(S_sa, sc_sa, deadline - 5)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Phase 4: QUBO SA from random starts
    for r in range(5):
        if time.time() > deadline - 10:
            break
        order = list(range(n))
        random.shuffle(order)
        S_r = greedy_construct(order, use_score=True, min_size=10)
        S_sa = qubo_sa(S_r, deadline - 8)
        sc_sa = count_3aps(S_sa)
        if count_4aps(S_sa) == 0:
            S_sa, sc_sa = hill_climb(S_sa, sc_sa, deadline - 5)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Phase 5: Final hill climb polish
    if time.time() < deadline - 3 and best_S:
        best_S, best_score = hill_climb(best_S, best_score, deadline - 1)

    # Convert to 1-indexed sorted list
    result = [x + 1 for x in range(n) if (best_S >> x) & 1]
    if not result:
        result = [1]

    # Final safety check: remove elements that create 4-APs
    if count_4aps(best_S) > 0:
        for ap in aps4:
            if all((best_S >> elem) & 1 for elem in ap):
                best_S &= ~(1 << ap[0])
        result = [x + 1 for x in range(n) if (best_S >> x) & 1]
        if not result:
            result = [1]

    return result
