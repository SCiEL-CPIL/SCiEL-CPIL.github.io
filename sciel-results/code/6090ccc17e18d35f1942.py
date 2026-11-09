import time, random, math

def _precompute(n=180):
    """Precompute all reciprocal triples."""
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
    return triples


def _count_violations(S, triples):
    """Count how many triples are fully contained in S."""
    c = 0
    for (i, j, k) in triples:
        if S[i] and S[j] and S[k]:
            c += 1
    return c


def _greedy_add(S, triples, n):
    """Greedy add: repeatedly add the vertex with fewest new violations."""
    improved = True
    while improved:
        improved = False
        cur_v = _count_violations(S, triples)
        best_v = None
        best_gain = 0
        
        for v in range(n):
            if not S[v]:
                # Count new violations if we add v
                new_v = 0
                for (i, j, k) in triples:
                    if v in (i, j, k):
                        other1, other2 = (i, j) if v == k else ((i, k) if v == j else (j, k))
                        if S[other1] and S[other2]:
                            new_v += 1
                gain = 1 - 10 * new_v  # Prefer adding if no new violations
                if new_v == 0 and best_v is None:
                    best_v = v
                    best_gain = gain
                elif new_v == 0 and gain > best_gain:
                    best_v = v
                    best_gain = gain
        
        if best_v is not None:
            S[best_v] = 1
            improved = True
    
    return S, sum(S)


def _repair(S, triples, n):
    """Remove vertices to eliminate all violations."""
    S = S[:]
    changed = True
    while changed:
        changed = False
        for (i, j, k) in triples:
            if S[i] and S[j] and S[k]:
                # Remove the one that appears in fewest triples with active neighbors
                di = sum(1 for (a, b, c) in triples if i in (a, b, c) and S[a] and S[b] and S[c])
                dj = sum(1 for (a, b, c) in triples if j in (a, b, c) and S[a] and S[b] and S[c])
                dk = sum(1 for (a, b, c) in triples if k in (a, b, c) and S[a] and S[b] and S[c])
                if di <= dj and di <= dk:
                    S[i] = 0
                elif dj <= di and dj <= dk:
                    S[j] = 0
                else:
                    S[k] = 0
                changed = True
                break
    return S, sum(S)


def _sa_search(S, triples, n, deadline, rng):
    """Simulated annealing with triple-aware moves."""
    cur_v = _count_violations(S, triples)
    cur_size = sum(S)
    best_S = S[:]
    best_size = cur_size
    best_v = cur_v
    
    T = 1.5
    T_min = 0.1
    
    while time.time() < deadline:
        # Random flip
        v = rng.randrange(n)
        old = S[v]
        
        if old == 0:
            addc = 0
            for (i, j, k) in triples:
                if v in (i, j, k):
                    other1, other2 = (i, j) if v == k else ((i, k) if v == j else (j, k))
                    if S[other1] and S[other2]:
                        addc += 1
            new_v = cur_v + addc
            new_size = cur_size + 1
        else:
            remc = 0
            for (i, j, k) in triples:
                if v in (i, j, k) and S[i] and S[j] and S[k]:
                    remc += 1
            new_v = cur_v - remc
            new_size = cur_size - 1
        
        dE = (new_size - cur_size) - 50.0 * (new_v - cur_v)
        
        if dE >= 0 or rng.random() < math.exp(dE / T):
            S[v] = 1 - old
            cur_v = new_v
            cur_size = new_size
            if cur_v == 0 and cur_size > best_size:
                best_S = S[:]
                best_size = cur_size
        
        T *= 0.99998
        if T < T_min:
            T = T_min
    
    return best_S, best_size


def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    triples = _precompute(n)
    rng = random.Random(12345)
    
    best = [0] * n
    best_size = 0
    
    num_restarts = 6
    restart_budget = (DEADLINE - 10) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 8:
            break
        
        # Diverse initializations
        if restart == 0:
            S = [0] * n
            for i in range(90, 180):
                S[i] = 1
        elif restart == 1:
            S = [0] * n
            for i in range(60, 180):
                S[i] = 1
        elif restart == 2:
            S = [0] * n
            for i in range(120, 180):
                S[i] = 1
        elif restart == 3:
            S = [0] * n
            for i in range(n):
                if rng.random() < 0.45:
                    S[i] = 1
        elif restart == 4:
            # Weighted toward larger values
            S = [0] * n
            for i in range(n):
                p = 0.2 + 0.5 * (i / n)
                if rng.random() < p:
                    S[i] = 1
        else:
            # Perturb best
            S = best[:]
            for _ in range(10):
                v = rng.randrange(n)
                S[v] = 1 - S[v]
        
        # Repair
        S, _ = _repair(S, triples, n)
        
        # SA
        sa_deadline = min(time.time() - t0 + restart_budget * 0.6, DEADLINE - 15)
        S, sz = _sa_search(S, triples, n, sa_deadline, rng)
        if sz > best_size:
            best = S[:]
            best_size = sz
        
        # Greedy add
        S2, sz2 = _greedy_add(S, triples, n)
        if sz2 > best_size:
            if _count_violations(S2, triples) == 0:
                best = S2[:]
                best_size = sz2
        
        # Greedy add from best
        Sb, szb = _greedy_add(best[:], triples, n)
        if szb > best_size:
            if _count_violations(Sb, triples) == 0:
                best = Sb[:]
                best_size = szb
    
    # Final polish
    for _ in range(15):
        if time.time() - t0 > DEADLINE - 2:
            break
        Sp = best[:]
        for _ in range(rng.randint(1, 2)):
            v = rng.randrange(n)
            if Sp[v]:
                Sp[v] = 0
        Sp, szp = _greedy_add(Sp, triples, n)
        if szp > best_size and _count_violations(Sp, triples) == 0:
            best = Sp[:]
            best_size = szp
    
    out = [i + 1 for i in range(n) if best[i]]
    if not out:
        out = [180]
    return out
