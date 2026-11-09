"""Rewritten program: Maximum Independent Set on conflict structure with greedy + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(vals, seed_for_rng):
        n = len(vals)
        if n == 0:
            return []
        rng = random.Random(seed_for_rng * 1000003 + 7)
        val_set = set(vals)
        index = {v: i for i, v in enumerate(vals)}

        # Conflict: a triple (a,b,c) with a+b=c. For a given element x, it is "bad"
        # if there exist a,b in current subset with a+b=x, or a,b in current with x+a=b etc.
        # We track conflicts via counting.

        # Greedy initialization: start with all, then remove highest-degree vertices.
        # degree(x) = number of pairs (a,b) in S with a+b = x  PLUS  number of pairs (a,b) in S with x+a=b or x+b=a.
        # Simpler: for each x in S, count how many other elements y in S have (x+y) in S or (y-x) in S or (x-y) in S.
        # Actually a conflict triple containing x: 
        #   x + y = z  (x,y,z in S)  -> x is a "summand"
        #   y + x = z  same
        #   y + z = x  -> x is the "sum"
        # So x is in a conflict if: exists y,z in S, y!=z not required, such that x+y=z or y+z=x.

        def conflict_count(S):
            Sset = set(S)
            cnt = 0
            for i in range(len(S)):
                a = S[i]
                for j in range(len(S)):
                    b = S[j]
                    if (a + b) in Sset:
                        cnt += 1
            return cnt

        # Greedy removal
        S = list(vals)
        Sset = set(S)
        # Remove elements with highest involvement until no conflict
        max_iter = 5000
        it = 0
        while it < max_iter:
            it += 1
            # find element with max conflict involvement
            Sset = set(S)
            best_x = None
            best_cnt = 0
            for x in S:
                c = 0
                for y in S:
                    if (x + y) in Sset:
                        c += 1
                    # y + x = z same as above
                # also x as sum: y+z=x
                for y in S:
                    if y != x and (x - y) in Sset:
                        c += 1
                if c > best_cnt:
                    best_cnt = c
                    best_x = x
            if best_cnt == 0:
                break
            S.remove(best_x)
            Sset.discard(best_x)

        # Now S is sum-free. Try to add back removed elements via simulated annealing.
        removed = [v for v in vals if v not in Sset]
        # Simulated annealing: try to add a removed element if it doesn't break sum-freeness,
        # or swap.
        def is_sum_free(Sset):
            for a in Sset:
                for b in Sset:
                    if (a + b) in Sset:
                        return False
            return True

        # We'll do local search: maintain S as a set. Objective = 0 conflicts, maximize |S|.
        # Use simulated annealing on "add/remove/swap" moves.
        cur = set(S)
        best = set(cur)
        best_size = len(cur)
        T = 1.0
        cooling = 0.9995
        steps = 20000
        for step in range(steps):
            if T < 1e-6:
                break
            # propose a move
            move = rng.random()
            if move < 0.5 and removed:
                # add a random removed element
                x = rng.choice(removed)
                if x not in cur:
                    new = cur | {x}
                    # count conflicts introduced
                    conf = 0
                    for y in new:
                        if (x + y) in new:
                            conf += 1
                        if (y - x) in new and y != x:
                            conf += 1
                    delta_conf = conf  # since cur was conflict-free
                    if delta_conf == 0:
                        cur = new
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                    else:
                        # accept with probability exp(-delta/T)
                        if rng.random() < math.exp(-delta_conf / max(T, 1e-9)):
                            cur = new
            elif move < 0.8:
                # remove a random element from cur (to allow better additions)
                if cur:
                    x = rng.choice(list(cur))
                    new = cur - {x}
                    cur = new
            else:
                # swap: remove one from cur, add one from removed
                if cur and removed:
                    x = rng.choice(list(cur))
                    y = rng.choice(removed)
                    new = (cur - {x}) | {y}
                    # check if new is sum-free
                    ok = True
                    for a in new:
                        for b in new:
                            if (a + b) in new:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        cur = new
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                    else:
                        # accept with small probability
                        if rng.random() < math.exp(-2.0 / max(T, 1e-9)):
                            cur = new
            T *= cooling

        # Final: ensure best is sum-free (it should be), return it
        # Re-verify best
        bs = set(best)
        for a in bs:
            for b in bs:
                if (a + b) in bs:
                    # fallback to greedy S
                    return sorted(S)
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
