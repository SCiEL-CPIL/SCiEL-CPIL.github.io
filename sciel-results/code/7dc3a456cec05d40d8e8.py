import time
import random
import itertools

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def get_lengths(S):
        lengths = {N}
        for k in S:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        for u, v in itertools.combinations(S, 2):
            lengths.add(abs(u - v) + 2)
        return lengths

    def is_pancyclic(S):
        lengths = get_lengths(S)
        for L in range(3, N + 1):
            if L not in lengths:
                return False
        return True

    def greedy_find_S(m):
        candidates = list(range(2, N - 1))

        seeds = [
            [2, 38, 5, 35, 10, 30, 15, 25],
            [2, 38, 7, 33, 10, 30, 15, 25],
            [2, 38, 5, 35, 12, 28, 17, 23],
            [2, 38, 4, 36, 9, 31, 14, 26],
            [2, 38, 6, 34, 11, 29, 16, 24],
            [2, 38, 5, 35, 8, 32, 13, 27],
            [3, 37, 6, 34, 11, 29, 16, 24],
            [2, 38, 5, 35, 10, 20, 15, 25],
            [2, 38, 4, 36, 9, 21, 14, 26],
            [2, 38, 6, 34, 8, 22, 13, 27],
        ]

        for seed in seeds:
            if not set(seed).issubset(set(candidates)):
                continue
            S = set(seed)
            if len(S) > m:
                continue
            if is_pancyclic(S):
                return S

        # Greedy from best anchor
        for anchor in [[2, 38], [3, 37], [2, 3, 37, 38]]:
            S = set(anchor)
            if len(S) > m:
                continue

            while len(S) < m:
                current_lengths = get_lengths(S)
                if is_pancyclic(S):
                    return S

                best_k = None
                best_score = -1

                for k in candidates:
                    if k in S:
                        continue
                    new_S = S | {k}
                    new_lengths = get_lengths(new_S)
                    gain = len(new_lengths - current_lengths)
                    score = gain
                    if score > best_score:
                        best_score = score
                        best_k = k

                if best_k is None:
                    break
                S.add(best_k)

            if is_pancyclic(S):
                return S

        # Random with local search
        rng = random.Random(42)
        for restart in range(20000):
            if time.time() > deadline:
                return None

            S = set(rng.sample(candidates, m))

            for step in range(500):
                if time.time() > deadline:
                    return None
                if is_pancyclic(S):
                    return S

                in_s = list(S)
                out_s = [x for x in candidates if x not in S]
                if not in_s or not out_s:
                    break

                # Try all swaps, pick best
                best_swap = None
                best_missing = len([L for L in range(3, N + 1) if L not in get_lengths(S)])

                for u in in_s:
                    for v in out_s:
                        S.remove(u)
                        S.add(v)
                        new_missing = len([L for L in range(3, N + 1) if L not in get_lengths(S)])
                        S.remove(v)
                        S.add(u)
                        if new_missing < best_missing:
                            best_missing = new_missing
                            best_swap = (u, v)

                if best_swap is None:
                    break

                u, v = best_swap
                S.remove(u)
                S.add(v)

        return None

    def find_S(m):
        S = greedy_find_S(m)
        if S:
            return S

        candidates = list(range(2, N - 1))
        rng = random.Random(123 + m)
        attempts = 0

        while time.time() < deadline and attempts < 1000000:
            attempts += 1
            S = set(rng.sample(candidates, m))
            if is_pancyclic(S):
                return S

        return None

    def build_result(S):
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
        for k in sorted(S):
            edges.append((0, k))

        cycles = []
        length_map = {}

        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)

        for u, v in itertools.combinations(sorted(S), 2):
            l = abs(u - v) + 2
            if l not in length_map:
                length_map[l] = ('chord_pair', min(u, v), max(u, v))

        length_map[N] = ('rim',)

        for L in range(3, N + 1):
            if L not in length_map:
                return None

            typ = length_map[L]
            if typ[0] == 'rim':
                cycles.append(list(range(N)))
            elif typ[0] == 'chord_short':
                k = typ[1]
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                cycles.append([0, u] + list(range(u + 1, v + 1)))

        return {'edges': edges, 'cycles': cycles}

    for m in [8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        S = find_S(m)
        if S:
            result = build_result(S)
            if result:
                return result

    # Fallback
    edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
    cycles = []
    for L in range(3, N + 1):
        cycles.append(list(range(L)))

    return {'edges': edges, 'cycles': cycles}
