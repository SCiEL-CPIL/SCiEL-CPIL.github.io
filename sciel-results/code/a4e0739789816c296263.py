import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0

    N = 180

    # Precompute conflict structure.
    # 1/a = 1/b + 1/c  <=>  bc = a(b+c), with a the smallest.
    # For given a<b: c = a*b/(b-a), integer, c>b, c<=N.
    triples = []
    x_pairs = {x: [] for x in range(1, N + 1)}   # x -> list of (o1,o2) other two in a triple
    conflict_pairs = {x: set() for x in range(1, N + 1)}  # x -> set of partners

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c > b and c <= N:
                triples.append((a, b, c))
                conflict_pairs[a].add(b); conflict_pairs[a].add(c)
                conflict_pairs[b].add(a); conflict_pairs[b].add(c)
                conflict_pairs[c].add(a); conflict_pairs[c].add(b)
                x_pairs[a].append((b, c))
                x_pairs[b].append((a, c))
                x_pairs[c].append((a, b))

    def is_valid(S):
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                return False
        return True

    def can_add(x, present):
        """present: list/bool of length N+1, present[y] true if y in set."""
        for (o1, o2) in x_pairs[x]:
            if present[o1] and present[o2]:
                return False
        return True

    def greedy_add_order(order, present):
        S = set()
        for x in order:
            if can_add(x, present):
                S.add(x)
                present[x] = True
        return S

    def greedy_remove():
        S = set(range(1, N + 1))
        present = [False] * (N + 1)
        for x in S:
            present[x] = True
        # conflict_count[x] = number of triples in which x participates and other two present
        conf = [0] * (N + 1)
        for x in S:
            for (o1, o2) in x_pairs[x]:
                if present[o1] and present[o2]:
                    conf[x] += 1
        while True:
            if time.time() - start_time > deadline - 5:
                break
            # pick x with max conf
            best_elem = None
            best_count = 0
            for x in S:
                if conf[x] > best_count:
                    best_count = conf[x]
                    best_elem = x
            if best_elem is None or best_count == 0:
                break
            x = best_elem
            S.remove(x)
            present[x] = False
            # update conf for neighbors
            for (o1, o2) in x_pairs[x]:
                # x was the "other" for triples; recompute affected
                # For each neighbor n, a triple (x, o1, o2) contributed to conf[o1] and conf[o2]
                # when x present. Now x removed.
                if present[o1]:
                    conf[o1] -= 1
                if present[o2]:
                    conf[o2] -= 1
            conf[x] = 0
        return S

    def local_search(S, max_iter=3000):
        S = set(S)
        present = [False] * (N + 1)
        for x in S:
            present[x] = True
        best = set(S)
        best_size = len(S)
        not_in = [x for x in range(1, N + 1) if x not in S]

        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            improved = False

            # Phase 1: add any element that fits
            for x in list(not_in):
                if can_add(x, present):
                    S.add(x)
                    present[x] = True
                    not_in.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best = set(S)

            # Phase 2: 1-swap (remove x, add y) and 2-swap (remove x, add y, add z)
            if not improved:
                # try single remove+add
                found = False
                for x in list(S):
                    # tentatively remove x
                    S.remove(x)
                    present[x] = False
                    # try adding each y not in S
                    for y in list(not_in):
                        if can_add(y, present):
                            S.add(y)
                            present[y] = True
                            not_in.remove(y)
                            # now try to add more (cascade)
                            for z in list(not_in):
                                if can_add(z, present):
                                    S.add(z)
                                    present[z] = True
                                    not_in.remove(z)
                            if len(S) > best_size:
                                best_size = len(S)
                                best = set(S)
                            found = True
                            break
                    if found:
                        improved = True
                        break
                    # restore x
                    S.add(x)
                    present[x] = True
                    if y in not_in:
                        # y was not added, it's still in not_in (we didn't remove it)
                        pass
                if not found and not improved:
                    break

            if not improved:
                break

        return best

    # Strategy 1: greedy removal
    best = greedy_remove()
    best_size = len(best)

    # Strategy 2: greedy add in several deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -len(conflict_pairs[x])),
        sorted(range(1, N + 1), key=lambda x: len(conflict_pairs[x])),
    ]
    for order in orders:
        present = [False] * (N + 1)
        S2 = greedy_add_order(order, present)
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)

    # Strategy 3: random restarts with local search
    random.seed(12345)
    count = 0
    while time.time() - start_time < deadline - 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        present = [False] * (N + 1)
        S_r = greedy_add_order(order, present)
        S_improved = local_search(S_r, max_iter=1500)
        if len(S_improved) > best_size:
            best = S_improved
            best_size = len(S_improved)
        count += 1
        if count > 400:
            break

    # Final high-precision polish
    if time.time() - start_time < deadline - 3:
        best_final = local_search(best, max_iter=10000)
        if len(best_final) > best_size:
            best = best_final
            best_size = len(best_final)

    if not is_valid(best):
        best = greedy_remove()

    return list(best)
