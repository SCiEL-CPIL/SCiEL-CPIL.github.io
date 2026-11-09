import time, random, math

def _precompute(n=180):
    triples = []
    for x in range(1, n + 1):
        x2 = x * x
        d = 1
        while d * d <= x2:
            if x2 % d == 0:
                e = x2 // d
                if d < e:
                    y = x + d
                    z = x + e
                    if y <= n and z <= n and y != z:
                        triples.append((x - 1, y - 1, z - 1))
            d += 1
    pairs = [[] for _ in range(n)]
    for (i, j, k) in triples:
        pairs[i].append((j, k))
        pairs[j].append((i, k))
        pairs[k].append((i, j))
    return triples, pairs


def _count_violations(S, triples):
    c = 0
    for (i, j, k) in triples:
        if S[i] and S[j] and S[k]:
            c += 1
    return c


def _greedy_add(S, pairs, n):
    """Greedy add with priority: prefer elements with fewer conflicts first."""
    improved = True
    while improved:
        improved = False
        # Collect candidates and score them by conflict count
        candidates = []
        for v in range(n):
            if not S[v]:
                conflicts = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        conflicts += 1
                candidates.append((conflicts, v))
        
        if not candidates:
            break
        
        # Sort by conflict count (ascending) - add least-conflicting first
        candidates.sort()
        
        added_any = False
        for conflicts, v in candidates:
            if not S[v]:  # Re-check in case earlier additions changed state
                # Re-verify it's still valid
                ok = True
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        ok = False
                        break
                if ok:
                    S[v] = 1
                    added_any = True
        
        improved = added_any
    return S, sum(S)


def _repair_best(S, triples, pairs, n):
    """Remove elements to fix violations, choosing removal that minimizes degree loss."""
    S = S[:]
    changed = True
    while changed:
        changed = False
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                di = sum(1 for (u, w) in pairs[i] if S[u] and S[w])
                dj = sum(1 for (u, w) in pairs[j] if S[u] and S[w])
                dk = sum(1 for (u, w) in pairs[k] if S[u] and S[w])
                # Remove the one with fewest active conflicts (least damage)
                if di <= dj and di <= dk:
                    S[i] = 0
                elif dj <= di and dj <= dk:
                    S[j] = 0
                else:
                    S[k] = 0
                changed = True
                break
    return S, sum(S)


def _two_swap_search(S, triples, pairs, n, deadline, rng):
    """Try 2-swap moves: remove one, add another. Uses adaptive temperature."""
    cur_v = _count_violations(S, triples)
    cur_size = sum(S)
    best_S = S[:]
    best_size = cur_size
    best_v = cur_v
    
    T = 0.8  # Starting temperature for swap acceptance
    T_min = 0.05
    
    while time.time() < deadline:
        # Pick random element to remove
        if cur_size == 0:
            break
        v_remove = rng.randrange(n)
        if not S[v_remove]:
            continue
        
        # Pick random element to add
        v_add = rng.randrange(n)
        if S[v_add]:
            continue
        
        # Compute effect of removing v_remove and adding v_add
        rem_v = 0
        for (u, w) in pairs[v_remove]:
            if S[u] and S[w]:
                rem_v += 1
        
        add_v = 0
        for (u, w) in pairs[v_add]:
            if u == v_remove:
                if S[w] and w != v_remove:
                    add_v += 1
            elif w == v_remove:
                if S[u] and u != v_remove:
                    add_v += 1
            else:
                if S[u] and S[w]:
                    add_v += 1
        
        new_v = cur_v - rem_v + add_v
        new_size = cur_size
        
        # Score change: prefer size increase, penalize violations
        dE = (new_size - cur_size) - 10.0 * (new_v - cur_v)
        
        if dE >= 0 or rng.random() < math.exp(dE / T):
            S[v_remove] = 0
            S[v_add] = 1
            cur_v = new_v
            cur_size = new_size
            if cur_v == 0 and cur_size > best_size:
                best_S = S[:]
                best_size = cur_size
        
        # Cool down slowly
        T *= 0.99995
        if T < T_min:
            T = T_min
    
    return best_S, best_size


def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    triples, pairs = _precompute(n)
    rng = random.Random(12345)
    
    best = [0] * n
    best_size = 0
    
    # Multiple restarts with different initial states
    num_restarts = 7
    restart_time = (DEADLINE - (time.time() - t0)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        # Different initial seeds - more diverse
        if restart == 0:
            # Upper half
            S = [0] * n
            for i in range(90, 180):
                S[i] = 1
        elif restart == 1:
            # Middle to upper
            S = [0] * n
            for i in range(60, 180):
                S[i] = 1
        elif restart == 2:
            # Upper third
            S = [0] * n
            for i in range(120, 180):
                S[i] = 1
        elif restart == 3:
            # Random 50%
            S = [0] * n
            for i in range(n):
                if rng.random() < 0.5:
                    S[i] = 1
        elif restart == 4:
            # Random 40%
            S = [0] * n
            for i in range(n):
                if rng.random() < 0.4:
                    S[i] = 1
        elif restart == 5:
            # Start from best and perturb
            S = best[:]
            for _ in range(15):
                v = rng.randrange(n)
                S[v] = 1 - S[v]
        else:
            # Weighted random: prefer larger numbers
            S = [0] * n
            for i in range(n):
                # Higher probability for larger indices
                p = 0.3 + 0.4 * (i / n)
                if rng.random() < p:
                    S[i] = 1
        
        # Repair initial state
        S, cur_size = _repair_best(S, triples, pairs, n)
        cur_v = _count_violations(S, triples)
        
        # SA phase with adaptive penalty
        T = 1.2
        penalty = 300.0
        sa_deadline = min(time.time() - t0 + restart_time * 0.65, DEADLINE - 10)
        
        while time.time() - t0 < sa_deadline:
            v = rng.randrange(n)
            old = S[v]
            if old == 0:
                addc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        addc += 1
                new_v = cur_v + addc
                new_size = cur_size + 1
            else:
                remc = 0
                for (u, w) in pairs[v]:
                    if S[u] and S[w]:
                        remc += 1
                new_v = cur_v - remc
                new_size = cur_size - 1
            
            new_score = new_size - penalty * new_v
            cur_score = cur_size - penalty * cur_v
            dE = new_score - cur_score
            if dE >= 0 or rng.random() < math.exp(dE / T):
                S[v] = 1 - old
                cur_v = new_v
                cur_size = new_size
                if new_v == 0 and cur_size > best_size:
                    best = S[:]
                    best_size = cur_size
            T *= 0.99997
            if T < 0.05:
                T = 0.05
            # Slowly reduce penalty to allow more exploration
            penalty *= 0.99999
            if penalty < 50.0:
                penalty = 50.0
        
        # Two-swap search phase
        swap_deadline = min(time.time() - t0 + restart_time * 0.25, DEADLINE - 8)
        S2, sz2 = _two_swap_search(S, triples, pairs, n, swap_deadline, rng)
        if sz2 > best_size:
            v2 = _count_violations(S2, triples)
            if v2 == 0:
                best = S2[:]
                best_size = sz2
        
        # Greedy add from current S
        S, cur_size = _greedy_add(S, pairs, n)
        if cur_size > best_size:
            feasible = True
            for (i, j, k) in triples:
                if S[i] and S[j] and S[k]:
                    feasible = False
                    break
            if feasible:
                best = S[:]
                best_size = cur_size
        
        # Greedy add from best
        Sb = best[:]
        Sb, szb = _greedy_add(Sb, pairs, n)
        if szb > best_size:
            feasible = True
            for (i, j, k) in triples:
                if Sb[i] and S[j] and S[k]:
                    feasible = False
                    break
            if feasible:
                best = Sb
                best_size = szb
    
    # Final polish: try greedy add multiple times from best with slight perturbations
    for _ in range(20):
        if time.time() - t0 > DEADLINE - 2:
            break
        Sp = best[:]
        # Remove 1-2 random elements
        for _ in range(rng.randint(1, 2)):
            v = rng.randrange(n)
            if Sp[v]:
                Sp[v] = 0
        Sp, szp = _greedy_add(Sp, pairs, n)
        if szp > best_size:
            feasible = True
            for (i, j, k) in triples:
                if Sp[i] and Sp[j] and Sp[k]:
                    feasible = False
                    break
            if feasible:
                best = Sp
                best_size = szp
    
    out = [i + 1 for i in range(n) if best[i]]
    if not out:
        out = [180]
    return out
