import time
from math import gcd
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0  # explicit wall-clock guard (evaluator kills at 90s, buffer for overhead)

    N = 512
    vals = np.arange(1, N + 1, dtype=np.int64)

    # ---- Precompute the full compatibility matrix (lcm(a,b) <= N) ----
    G = np.empty((N, N), dtype=bool)
    for i in range(N):
        a = vals[i]
        row = (a // np.gcd(a, vals)) * vals
        G[i] = row <= N
    # diagonal (self) is always True; keep it, harmless.

    # ---- Divisor-set seeds (all divisors of L form a valid clique) ----
    def get_divisors(n):
        d = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                d.add(i)
                d.add(n // i)
            i += 1
        return d

    div_sets = []
    for L in range(1, N + 1):
        divs = get_divisors(L)
        div_sets.append((len(divs), L, divs))
    div_sets.sort(key=lambda t: (-t[0], t[1]))

    # ---- Deterministic greedy expansion using the compatibility matrix ----
    # Optimized: maintains a mask of compatible candidates
    def expand(s):
        s = set(s)
        if not s:
            return s
        
        # Initialize compatibility mask: candidate c is compatible if G[c-1, x-1] is True for all x in s
        # We can compute this as the logical AND of rows for all x in s
        arr = np.fromiter(s, dtype=np.int64, count=len(s))
        compatible = np.all(G[arr - 1], axis=0)
        # Exclude elements already in s
        compatible[arr - 1] = False
        
        while time.time() < deadline:
            if len(s) == N:
                break
            
            cand = np.flatnonzero(compatible)
            if cand.size == 0:
                break
            
            # Score = number of compatible partners in s
            # We can compute this by summing G[cand, arr-1]
            scores = G[cand, arr - 1].sum(axis=1)
            order = np.argsort(-scores, kind="stable")
            best_idx = order[0]
            best = int(cand[best_idx])
            
            s.add(best)
            
            # Update compatibility mask: new candidate must be compatible with 'best'
            # compatible = compatible & G[best-1]
            # But we also need to ensure 'best' itself is removed from candidates (already done by mask logic if we update carefully)
            # Actually, simpler: just AND with the row of the new element.
            compatible &= G[best - 1]
            compatible[best - 1] = False # Ensure best is not picked again
            
        return s

    # ---- Local search: try removals, re-expand, keep improvement ----
    def local_search(s):
        s = set(s)
        best = len(s)
        improved = True
        # Limit iterations to prevent timeout
        max_iters = 20
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            lst = list(s)
            m = len(lst)
            
            # 1-exchange: remove one element, re-expand
            # Heuristic: try removing elements that are "least connected" or just random sample if m is large
            # For simplicity and speed, we'll try a limited number of removals
            # If m is small, try all. If m is large, try a random subset.
            if m <= 50:
                candidates_to_remove = lst
            else:
                candidates_to_remove = random.sample(lst, 20)
                
            for rc in candidates_to_remove:
                if time.time() > deadline:
                    break
                t = set(s)
                t.discard(rc)
                e = expand(t)
                if len(e) > best:
                    s, best = e, len(e)
                    improved = True
                    break
            
            if improved or time.time() > deadline:
                continue
                
            # 2-exchange: remove two elements, re-expand
            # Only do this if 1-exchange failed and we have time
            if m <= 30:
                # Try all pairs for small sets
                for i in range(m):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, m):
                        t = set(s)
                        t.discard(lst[i])
                        t.discard(lst[j])
                        e = expand(t)
                        if len(e) > best:
                            s, best = e, len(e)
                            improved = True
                            break
                    if improved:
                        break
            else:
                # Try random pairs for large sets
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    if m < 2:
                        break
                    rc1, rc2 = random.sample(lst, 2)
                    t = set(s)
                    t.discard(rc1)
                    t.discard(rc2)
                    e = expand(t)
                    if len(e) > best:
                        s, best = e, len(e)
                        improved = True
                        break
                        
        return s

    # ---- Build candidate seeds: top divisor sets + a few random starts ----
    seeds = []
    # Use top divisor sets as seeds
    for _, L, divs in div_sets[:5]:
        seeds.append(set(divs))
        
    # Add a few random starts to diversify
    for _ in range(3):
        rs = set()
        # Start with a random number and expand
        start_val = random.randint(1, N)
        rs.add(start_val)
        rs = expand(rs)
        seeds.append(rs)

    # ---- Evaluate each seed: expand -> local search, keep the best ----
    best_set = None
    best_size = -1
    for seed in seeds:
        if time.time() > deadline - 2.0:
            break
        e = expand(set(seed))
        ls = local_search(e)
        if len(ls) > best_size:
            best_size, best_set = len(ls), ls

    # ---- Final polish on the best set ----
    if best_set is not None and time.time() < deadline - 1.0:
        best_set = expand(set(best_set))

    if best_set is None or len(best_set) == 0:
        # guaranteed-valid fallback: divisors of 512
        best_set = get_divisors(512)

    return list(best_set)
