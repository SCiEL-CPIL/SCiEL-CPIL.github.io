import time
import random
import math

N = 128

def build():
    aps = []
    for d in range(1, (N-1)//3 + 1):
        max_a = N - 1 - 3*d
        for a in range(0, max_a + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    return aps

APS = build()
OTHER3 = [[] for _ in range(N)]
for ap in APS:
    for p in ap:
        others = [q for q in ap if q != p]
        OTHER3[p].append(tuple(others))

def distinct_count(c, ap):
    a = c[ap[0]]; b = c[ap[1]]; cc = c[ap[2]]; d = c[ap[3]]
    s = 1
    if b != a: s += 1
    if cc != a and cc != b: s += 1
    if d != a and d != b and d != cc: s += 1
    return s

def total_cost(c):
    tot = 0
    for ap in APS:
        if distinct_count(c, ap) < 3:
            tot += 1
    return tot

def verify(c):
    for ap in APS:
        if distinct_count(c, ap) < 3:
            return False
    return True

def solve_k(k, budget, seed):
    rng = random.Random(seed)
    start = time.time()
    while time.time() - start < budget:
        c = [rng.randrange(k) for _ in range(N)]
        cost = total_cost(c)
        if cost == 0:
            return c
        T = 1.5
        T_min = 0.02
        no_improve = 0
        best_cost = cost
        while time.time() - start < budget:
            # choose position
            # find a violated AP for directed move
            # (recompute violated list occasionally)
            p = rng.randrange(N)
            oc = c[p]
            best_nc = oc
            best_delta = 0
            for nc in range(k):
                if nc == oc:
                    continue
                ov = 0; nv = 0
                for (o1, o2, o3) in OTHER3[p]:
                    a = c[o1]; b = c[o2]; cc = c[o3]
                    s3 = 1
                    if b != a: s3 += 1
                    if cc != a and cc != b: s3 += 1
                    db = s3 + (0 if (oc == a or oc == b or oc == cc) else 1)
                    da = s3 + (0 if (nc == a or nc == b or nc == cc) else 1)
                    if db < 3: ov += 1
                    if da < 3: nv += 1
                delta = nv - ov
                if delta < best_delta or (delta == best_delta and rng.random() < 0.5):
                    best_delta = delta; best_nc = nc
            if best_nc != oc:
                if best_delta <= 0 or rng.random() < math.exp(-best_delta / T):
                    c[p] = best_nc
                    cost += best_delta
                    if cost == 0:
                        return c
                    if cost < best_cost:
                        best_cost = cost
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
            # cooling
            if no_improve > 400:
                # restart this run
                break
            T = max(T_min, T * 0.9995)
        # restart outer
    return None

def construct():
    overall_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s
    # try k=3,4,5,...
    k = 3
    seed = 12345
    while time.time() - overall_start < DEADLINE - 5:
        remaining = DEADLINE - 5 - (time.time() - overall_start)
        if remaining < 3:
            break
        # allocate time for this k
        alloc = min(remaining, 25.0 if k == 3 else 15.0)
        res = solve_k(k, alloc, seed)
        seed += 1000
        if res is not None and verify(res):
            return res
        k += 1
        if k > 8:
            break
    # fallback: guaranteed valid coloring using n mod 49
    c = [ (i % 49) for i in range(1, N+1) ]
    # verify (should be valid)
    if verify(c):
        return c
    # ultimate fallback: unique colors
    return list(range(N))
