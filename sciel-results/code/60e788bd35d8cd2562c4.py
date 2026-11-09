"""Solver for large sum-free subsets of three fixed sets of integers."""
# EVOLVE-BLOCK-START
import time
import random
import math
from random import Random

def construct():
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def build_triples(vals):
        vset = set(vals)
        conflict = {c: [] for c in vals}
        for c in vals:
            for a in vals:
                b = c - a
                if b in vset and a <= b:
                    conflict[c].append((a, b))
        return conflict

    def is_sum_free(subset):
        s = set(subset)
        for c in subset:
            for a in subset:
                b = c - a
                if b in s and a <= b:
                    return False
        return True

    def greedy_initial(vals, conflict, ordering_seed):
        items = [(x, Random(ordering_seed * 1000003 + x).random()) for x in vals]
        items.sort(key=lambda t: (len(conflict[t[0]]), t[1]))
        ordered = [t[0] for t in items]
        selected = set()
        for c in ordered:
            ok = True
            for (a, b) in conflict[c]:
                if a in selected and b in selected:
                    ok = False
                    break
            if ok:
                for a in selected:
                    if (c + a) in selected:
                        ok = False
                        break
            if ok:
                selected.add(c)
        return selected

    def size_conflict(c, sel, conflict):
        cnt = 0
        for (a, b) in conflict[c]:
            if a in sel and b in sel:
                cnt += 1
        for a in sel:
            if (c + a) in sel:
                cnt += 1
        return cnt

    def try_add(c, sel, conflict):
        if c in sel:
            return False
        if size_conflict(c, sel, conflict) == 0:
            sel.add(c)
            return True
        return False

    def local_search(vals, conflict, start_selected, deadline, rng):
        selected = set(start_selected)
        best = set(selected)
        best_size = len(selected)

        # Phase 1: greedy add
        changed = True
        while changed:
            changed = False
            for c in vals:
                if c not in selected:
                    if try_add(c, selected, conflict):
                        changed = True
                        if len(selected) > best_size:
                            best = set(selected)
                            best_size = len(selected)

        # Phase 2: remove-and-retry
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = list(selected)
            rng.shuffle(order)
            for e in order:
                if time.time() >= deadline:
                    break
                saved = set(selected)
                selected.discard(e)
                for c in vals:
                    if c not in selected:
                        try_add(c, selected, conflict)
                if len(selected) > len(saved):
                    improved = True
                    if len(selected) > best_size:
                        best = set(selected)
                        best_size = len(selected)
                else:
                    selected = saved

        # Phase 3: simulated annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        cur = set(selected)
        if len(cur) > best_size:
            best = set(cur)
            best_size = len(cur)
        steps = 0
        while T > T_min and time.time() < deadline:
            move = rng.choice(['add', 'remove', 'swap'])
            if move == 'add':
                c = rng.choice(vals)
                if c not in cur:
                    conf = size_conflict(c, cur, conflict)
                    if conf == 0:
                        cur.add(c)
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                    elif rng.random() < math.exp(-conf / T):
                        cur.add(c)
            elif move == 'remove':
                if cur:
                    cur.discard(rng.choice(list(cur)))
            else:
                if cur:
                    e = rng.choice(list(cur))
                    c = rng.choice(vals)
                    if c not in cur:
                        cur.discard(e)
                        conf = size_conflict(c, cur, conflict)
                        if conf == 0:
                            cur.add(c)
                            if len(cur) > best_size:
                                best = set(cur)
                                best_size = len(cur)
                        elif rng.random() < math.exp(-conf / T):
                            cur.add(c)
                        else:
                            cur.add(e)
            T *= cooling
            steps += 1

        if len(cur) > best_size:
            best = set(cur)
        return best

    def solve_instance(vals, deadline):
        conflict = build_triples(vals)
        best = set()
        best_size = 0
        n_starts = 8
        for i in range(n_starts):
            if time.time() >= deadline:
                break
            start_sel = greedy_initial(vals, conflict, ordering_seed=1000 + i * 7919)
            rng = random.Random(2000 + i * 104729)
            res = local_search(vals, conflict, start_sel, deadline, rng)
            if len(res) > best_size:
                best = res
                best_size = len(res)
        odds = set(x for x in vals if x % 2)
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)
        if time.time() < deadline:
            rng = random.Random(99999)
            res = local_search(vals, conflict, best, deadline, rng)
            if len(res) > best_size:
                best = res
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        per_deadline = min(DEADLINE, time.time() + 40.0)
        results.append(solve_instance(vals, per_deadline))

    return results

# EVOLVE-BLOCK-END
