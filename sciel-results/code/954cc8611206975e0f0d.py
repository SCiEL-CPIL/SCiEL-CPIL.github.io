import time
from math import gcd
import random

def construct():
    N = 128
    start = time.time()
    deadline = start + 80.0

    try:
        # Precompute, for each x, the set of pairs {y,z} such that {x,y,z} is a bad triple
        bad = [set() for _ in range(N + 1)]
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                for c in range(b + 1, N + 1):
                    if gcd(a, b) == gcd(a, c) == gcd(b, c):
                        bad[a].add(frozenset((b, c)))
                        bad[b].add(frozenset((a, c)))
                        bad[c].add(frozenset((a, b)))

        all_numbers = list(range(1, N + 1))
        deg = [len(bad[x]) for x in range(N + 1)]

        def can_add(x, ps):
            return not (ps & bad[x])

        def add_elem(x, S, ps):
            S.add(x)
            for u in S:
                if u != x:
                    ps.add(frozenset((x, u)))

        def remove_elem(y, S, ps):
            for u in S:
                if u != y:
                    ps.discard(frozenset((y, u)))
            S.discard(y)

        def build_ps(S):
            ps = set()
            lst = list(S)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    ps.add(frozenset((lst[i], lst[j])))
            return ps

        def greedy(order):
            S = set()
            ps = set()
            for x in order:
                if can_add(x, ps):
                    add_elem(x, S, ps)
            return S, ps

        def saturate(S, ps):
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in all_numbers:
                    if x not in S and can_add(x, ps):
                        add_elem(x, S, ps)
                        changed = True
            return S, ps

        def swap_check(x, y, S, ps):
            # Is (S - {y}) + {x} valid?
            pairs_y = set()
            for u in S:
                if u != y:
                    pairs_y.add(frozenset((y, u)))
            remaining = ps - pairs_y
            if remaining & bad[x]:
                return False
            for u in S:
                if u != y and frozenset((x, u)) in bad[x]:
                    return False
            return True

        def do_swap(x, y, S, ps):
            remove_elem(y, S, ps)
            add_elem(x, S, ps)

        best_S = set()
        best_size = 0
        rng = random.Random(987654321)

        def update_best(S):
            nonlocal best_S, best_size
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        orderings = [
            all_numbers[:],
            sorted(all_numbers, key=lambda x: deg[x]),
            sorted(all_numbers, key=lambda x: -deg[x]),
            sorted(all_numbers),
            sorted(all_numbers, reverse=True),
        ]

        # Multiple restarts: diverse greedy + saturate + directed swap local search
        while time.time() < deadline - 3.0:
            if orderings:
                order = orderings.pop(0)
            else:
                order = all_numbers[:]
                rng.shuffle(order)

            S, ps = greedy(order)
            S, ps = saturate(S, ps)
            update_best(S)

            ls_start = time.time()
            while time.time() - ls_start < 3.0 and time.time() < deadline - 3.0:
                progress = False
                outside = [x for x in all_numbers if x not in S]
                rng.shuffle(outside)
                for x in outside:
                    if time.time() < deadline - 3.0:
                        ys = list(S)
                        rng.shuffle(ys)
                        for y in ys:
                            if swap_check(x, y, S, ps):
                                do_swap(x, y, S, ps)
                                S, ps = saturate(S, ps)
                                update_best(S)
                                progress = True
                                break  # continue scanning other outside x's
                if not progress:
                    break

        # Final exhaustive single-swap polish on the best set
        if best_S and time.time() < deadline - 1.0:
            S = set(best_S)
            ps = build_ps(S)
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                outside = [x for x in all_numbers if x not in S]
                for x in outside:
                    if time.time() < deadline - 1.0:
                        for y in list(S):
                            if swap_check(x, y, S, ps):
                                do_swap(x, y, S, ps)
                                S, ps = saturate(S, ps)
                                update_best(S)
                                improved = True
                                break
                    if improved:
                        break

        result = list(best_S)
        if not result:
            result = [1]
        return result

    except Exception:
        # Safe fallback: pure greedy
        selected = []
        for x in range(1, 129):
            ok = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    a, b = selected[i], selected[j]
                    if gcd(a, b) == gcd(a, x) == gcd(b, x):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected if selected else [1]
