"""Rewritten program: SEARCH approach for unique-triple-sum sets."""
import itertools
import random
import time

from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0   # safety margin under the 90s kill

    N = 4096

    # ---------- helper: validate a set ----------
    def is_valid(vals):
        # O(k^3) sum check with early exit
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # ---------- helper: greedy extend from a base set ----------
    def greedy_extend(base, candidates, limit):
        chosen = list(base)
        # build triple sums for current chosen
        def rebuild_sums():
            st = set()
            for x in itertools.combinations_with_replacement(chosen, 3):
                st.add(sum(x))
            return st
        sums = rebuild_sums()
        chosen_set = set(chosen)
        # Precompute all pair sums (u,w) with u<=w for fast incremental test
        def get_pair_sums(cl):
            ps = set()
            for ui in range(len(cl)):
                u = cl[ui]
                for wi in range(ui, len(cl)):
                    w = cl[wi]
                    ps.add(u + w)
            return ps
        pair_sums = get_pair_sums(chosen)
        
        for v in candidates:
            if time.time() > limit:
                break
            if v in chosen_set:
                continue
            # Check new triple sums: (v,v,v), (v,v,u), (v,u,u), (v,u,w)
            # (v,v,v): 3v
            if 3 * v in sums:
                continue
            # (v,v,u): 2v+u for u in chosen
            # (v,u,u): v+2u for u in chosen
            # (v,u,w): v+(u+w) for (u,w) in pair_sums
            new_sums = set()
            new_sums.add(3 * v)
            for u in chosen:
                s1 = 2 * v + u
                if s1 in sums:
                    break
                s2 = v + 2 * u
                if s2 in sums:
                    break
                new_sums.add(s1)
                new_sums.add(s2)
            else:
                # Check (v,u,w) sums
                for ps in pair_sums:
                    s3 = v + ps
                    if s3 in sums:
                        break
                    new_sums.add(s3)
                else:
                    # All new sums are valid
                    sums |= new_sums
                    chosen.append(v)
                    chosen_set.add(v)
                    # Update pair_sums with new pairs involving v
                    for u in chosen[:-1]:  # all previously chosen
                        pair_sums.add(v + u)
                    pair_sums.add(2 * v)
        return chosen

    best = []

    # ---------- 1) power-of-2 core (provable) ----------
    pow2 = [2 ** i for i in range(0, 12) if 2 ** i <= 2048]
    if is_valid(pow2):
        best = pow2
    # extend the power-2 core greedily with ascending order
    try:
        ext = greedy_extend(pow2, list(range(1, N + 1)), DEADLINE)
        if is_valid(ext) and len(ext) > len(best):
            best = ext
    except Exception:
        pass
    # Also try extending with descending order
    try:
        ext_desc = greedy_extend(pow2, list(range(N, 0, -1)), DEADLINE)
        if is_valid(ext_desc) and len(ext_desc) > len(best):
            best = ext_desc
    except Exception:
        pass

    # ---------- 2) randomized greedy constructions ----------
    # Strategy: try multiple base spacings, both ascending and shuffled orders
    random.seed(12345)
    best_size = len(best)
    for it in range(120):  # More iterations with faster greedy
        if time.time() > DEADLINE - 2:
            break
        # Vary the base spacing
        base = random.randint(2, 12)
        # Create candidate pool with some density
        density = random.choice([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        pool = []
        for x in range(1, N + 1):
            if (x - 1) % base == 0 or random.random() < density:
                pool.append(x)
        
        # Alternate between ascending, descending, and shuffled order
        r = it % 3
        if r == 0:
            # Ascending order (deterministic greedy)
            candidates = sorted(pool)
        elif r == 1:
            # Descending order
            candidates = sorted(pool, reverse=True)
        else:
            # Shuffled order
            candidates = pool[:]
            random.shuffle(candidates)
        
        # Use the optimized greedy_extend for empty base
        chosen = greedy_extend([], candidates, DEADLINE - 2)
        
        if len(chosen) > best_size:
            if is_valid(chosen):
                best = chosen
                best_size = len(chosen)

    # ---------- 3) local improvement: multiple passes to add missing numbers ----------
    if time.time() < DEADLINE - 5:
        # Use the optimized greedy_extend with best as base
        for pass_num in range(5):
            if time.time() > DEADLINE - 3:
                break
            # Different candidate orderings per pass
            if pass_num == 0:
                candidates = list(range(1, N + 1))
            elif pass_num == 1:
                candidates = list(range(N, 0, -1))
            elif pass_num == 2:
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
            elif pass_num == 3:
                # Focus on gaps: numbers not too close to existing ones
                candidates = [x for x in range(1, N + 1) if x not in set(best)]
                random.shuffle(candidates)
            else:
                candidates = list(range(1, N + 1, 2))  # Odd numbers
                random.shuffle(candidates)
            
            ext = greedy_extend(list(best), candidates, DEADLINE - 3)
            if is_valid(ext) and len(ext) > len(best):
                best = ext
            else:
                break  # No progress, stop

    # ---------- 4) sanity: ensure within range, distinct, positive ----------
    best = [v for v in best if 1 <= v <= N]
    best = sorted(set(best))

    return best
