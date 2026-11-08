import time
from math import log, gcd, exp
from itertools import combinations
import random

def _score_raw(U):
    """Score an already-normalized sorted set (contains 0, len>=2)."""
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    sc = len(sums)
    dc = len(diffs)
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1:
        return -1.0
    D = 2 * U[-1] + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def normalize(U):
    """Return sorted distinct nonnegative ints containing 0, divided by gcd."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    g = 0
    for x in U:
        g = gcd(g, x)
    if g > 1:
        U = sorted(set(x // g for x in U))
    return U

def score_set(U):
    U = normalize(list(U))
    if len(U) < 2:
        return -1.0
    return _score_raw(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition/replacement based on sums and differences.
    """
    U_set = set(U)
    candidates = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            s = ui + U[j]
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if i != j:
                d = abs(ui - U[j])
                if 0 < d <= limit_val and d not in U_set:
                    candidates.add(d)
    return list(candidates)

def local_search(U, deadline, max_val=500, max_n=30, T0=0.02):
    """
    Local search with dynamic candidate generation.
    """
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 300
    step = 0
    
    # Initial candidate pool
    cur_limit = max(max_val, max(U) * 2) if U else max_val
    candidates = generate_candidates(U, cur_limit)
    
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        step += 1
        r = random.random()
        n = len(U)
        new_U = None
        
        # Periodically refresh candidates
        if step % 10 == 0 or not candidates:
            cur_limit = max(max_val, max(U) * 2) if U else max_val
            candidates = generate_candidates(U, cur_limit)
            if not candidates:
                # Fallback to random if no structural candidates
                candidates = [random.randint(1, cur_limit) for _ in range(10)]

        if r < 0.40 and n < max_n:
            # Add from candidates
            if candidates:
                v = random.choice(candidates)
                if v not in U:
                    new_U = normalize(U + [v])
            else:
                v = random.randint(1, cur_limit)
                if v not in U:
                    new_U = normalize(U + [v])
        elif r < 0.60 and n > 2:
            # Remove
            idx = random.randint(1, n - 1)
            new_U = normalize([x for j, x in enumerate(U) if j != idx])
        elif r < 0.80:
            # Replace with candidate
            if candidates:
                idx = random.randint(1, n - 1)
                v = random.choice(candidates)
                if v not in U:
                    cand = list(U)
                    cand[idx] = v
                    new_U = normalize(cand)
            else:
                idx = random.randint(1, n - 1)
                v = random.randint(1, cur_limit)
                if v not in U:
                    cand = list(U)
                    cand[idx] = v
                    new_U = normalize(cand)
        elif r < 0.90:
            # Perturb nearby
            idx = random.randint(1, n - 1)
            v = U[idx] + random.randint(-15, 15)
            if v >= 1 and v not in U:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        else:
            # Swap
            if n > 2:
                i = random.randint(1, n - 1)
                j = random.randint(1, n - 1)
                if i != j:
                    cand = list(U)
                    cand[i], cand[j] = cand[j], cand[i]
                    new_U = normalize(cand)
        
        if new_U is None or len(new_U) < 2:
            no_improve += 1
            continue
            
        new_score = _score_raw(new_U)
        
        if new_score > cur_score:
            U = new_U
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(U)
            # Refresh candidates immediately after improvement
            cur_limit = max(max_val, max(U) * 2) if U else max_val
            candidates = generate_candidates(U, cur_limit)
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                U = new_U
                cur_score = new_score
            no_improve += 1
            
        T = max(T * 0.998, 1e-5)
        
    return best_score, best_set

def greedy_build(deadline, maxv=100, maxn=15):
    """Constructively add the element that most improves the score."""
    U = [0]
    while time.time() < deadline - 4:
        cur = score_set(U)
        best_v = None
        best_sc = cur
        # Generate candidates for greedy step too
        cands = generate_candidates(U, maxv)
        if not cands:
            cands = list(range(1, maxv + 1))
        else:
            cands = cands[:50] # Limit candidates for speed
        
        for v in cands:
            if v in U:
                continue
            sc = _score_raw(normalize(U + [v]))
            if sc > best_sc:
                best_sc = sc
                best_v = v
        if best_v is None or len(U) >= maxn:
            break
        U = normalize(U + [best_v])
    return U

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    best_set = [0, 1, 3, 4]
    best_score = score_set(best_set)

    # --- Structured seeds ---
    structured = [
        [0,1,3,4],
        [0,1,3,4,9,10,12,13],
        [0,1,4,9,11],
        [0,1,4,10,20,35],
        [0,1,2,4,8,16],
        [0,1,3,7,15,31],
        [0,1,5,13,25,41],
        [0,1,3,9,27,81],
        [0,1,4,10,18,28],
        [0,1,2,5,13,34],
        [0,1,2,4,7,9],
        [0,1,3,6,7,11],
        [0, 1, 4, 9, 16, 25, 36, 49], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64], # Powers of 2
    ]
    for s in structured:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- Random + greedy seeds ---
    seeds = [list(best_set)]
    for _ in range(20):
        n = random.randint(3, 8)
        vals = random.sample(range(1, 100), n)
        s = normalize([0] + vals)
        seeds.append(s)
    
    for mv in (60, 100, 150):
        if time.time() < deadline - 100:
            g = greedy_build(deadline, maxv=mv, maxn=12)
            if len(g) >= 2:
                seeds.append(g)
                
    for s in seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)

    # --- Exhaustive small-n (normalized) ---
    def exhaustive(n, maxv):
        bs = best_score
        bc = best_set
        for comb in combinations(range(1, maxv + 1), n - 1):
            U = normalize([0] + list(comb))
            if len(U) != n:
                continue
            sc = _score_raw(U)
            if sc > bs:
                bs = sc
                bc = list(U)
        return bs, bc

    if time.time() < deadline - 75:
        bs, bc = exhaustive(3, 80)
        if bs > best_score: best_score, best_set = bs, bc
    if time.time() < deadline - 55:
        bs, bc = exhaustive(4, 50)
        if bs > best_score: best_score, best_set = bs, bc
    if time.time() < deadline - 40:
        bs, bc = exhaustive(5, 35)
        if bs > best_score: best_score, best_set = bs, bc

    # --- Local search restarts ---
    restart_seeds = [list(best_set)] + seeds[:10]
    attempts = 0
    while time.time() < deadline - 20:
        base = restart_seeds[attempts % len(restart_seeds)]
        sc, s = local_search(base, deadline, max_val=300, max_n=28,
                             T0=random.uniform(0.01, 0.03))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            if len(restart_seeds) < 16:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1

    # --- Final high-precision polish ---
    if time.time() < deadline - 3:
        cur = list(best_set)
        cur_score = _score_raw(cur)
        improved = True
        it = 0
        while improved and time.time() < deadline - 2 and it < 100:
            improved = False
            it += 1
            n = len(cur)
            # Replace with small delta window
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old = cur[i]
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    v = old + delta
                    if v < 1 or v in cur:
                        continue
                    cand = list(cur)
                    cand[i] = v
                    newU = normalize(cand)
                    if len(newU) != n:
                        continue
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        break
                if improved:
                    break
            if improved:
                continue
            # Add
            if n < 25:
                hi = min(200, cur[-1] + 50)
                cands = generate_candidates(cur, hi)
                if not cands:
                    cands = list(range(1, hi + 1))
                for v in cands:
                    if v in cur:
                        continue
                    newU = normalize(cur + [v])
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        break
            if improved:
                continue
            # Remove
            for i in range(1, n):
                newU = normalize([x for j, x in enumerate(cur) if j != i])
                if len(newU) < 2:
                    continue
                sc = _score_raw(newU)
                if sc > cur_score:
                    cur, cur_score, improved = newU, sc, True
                    break
        if cur_score > best_score:
            best_score = cur_score
            best_set = cur

    return normalize(list(best_set))

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
