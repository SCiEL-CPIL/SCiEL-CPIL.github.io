import time
import random
import heapq
import itertools

def construct(q, n, radius):
    start = time.time()
    deadline = 148.0

    # Generic fallback for non-(3,6,1) parameters
    if not (q == 3 and n == 6 and radius == 1):
        total = q ** n
        def to_digits(num):
            r = []
            for _ in range(n):
                r.append(num % q)
                num //= q
            return r
        def from_digits(d):
            v = 0
            for x in reversed(d):
                v = v * q + x
            return v
        nbrs = {}
        for w in range(total):
            d = to_digits(w)
            s = {w}
            for i in range(n):
                for nv in range(q):
                    if nv != d[i]:
                        nd = d[:]; nd[i] = nv
                        s.add(from_digits(nd))
            nbrs[w] = list(s)
        # Greedy cover
        counts = [0] * total
        chosen = []
        for w in range(total):
            add = 0
            for v in nbrs[w]:
                if counts[v] == 0:
                    add += 1
            if add > 0:
                chosen.append(w)
                for v in nbrs[w]:
                    counts[v] += 1
        return [to_digits(w) for w in chosen]

    # ---- Specialised, fast solver for q=3, n=6, radius=1 ----
    N = 729
    pow3 = [1, 3, 9, 27, 81, 243, 729]

    def to_base3(num):
        return [num // pow3[i] % 3 for i in range(n)]

    def from_base3(digits):
        return sum(digits[i] * pow3[i] for i in range(n))

    # Precompute all distance-1 neighbours (12 per word)
    nbr1 = [[] for _ in range(N)]
    for w in range(N):
        d = to_base3(w)
        lst = []
        for i in range(n):
            o = d[i]
            for nv in range(3):
                if nv != o:
                    nd = d[:]
                    nd[i] = nv
                    lst.append(from_base3(nd))
        nbr1[w] = lst

    # Precompute distance-2 neighbour masks (bitsets over 729 words)
    dist2_mask = [0] * N
    for w in range(N):
        d = to_base3(w)
        m = 0
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = d[i], d[j]
                for ai in range(3):
                    if ai == oi:
                        continue
                    for aj in range(3):
                        if aj == oj:
                            continue
                        nd = d[:]
                        nd[i] = ai
                        nd[j] = aj
                        m |= 1 << from_base3(nd)
        dist2_mask[w] = m

    # Greedy maximal independent set on distance-2 graph -> good initial cover
    random.seed(12345)
    order = list(range(N))
    random.shuffle(order)
    chosen = []
    chosen_mask = 0
    for w in order:
        if dist2_mask[w] & chosen_mask:
            continue
        chosen.append(w)
        chosen_mask |= 1 << w
    initial_size = len(chosen)

    def coverage_counts(cset):
        cnt = [0] * N
        for c in cset:
            for v in nbr1[c]:
                cnt[v] += 1
        return cnt

    def remove_redundants(cset, cnt):
        """Max-heap (lazy) pass: remove codewords whose ball is fully covered by others."""
        heap = []
        for c in cset:
            mn = min(cnt[v] for v in nbr1[c])
            heapq.heappush(heap, (-mn, c))
        while heap:
            if time.time() > deadline - 1.0:
                break
            neg, c = heapq.heappop(heap)
            if c not in cset:
                continue
            if (-neg) != min(cnt[v] for v in nbr1[c]):
                continue
            if (-neg) > 1:
                # c is redundant
                cset.remove(c)
                for v in nbr1[c]:
                    cnt[v] -= 1
                # update heap for neighbours of c
                for nb in nbr1[c]:
                    if nb in cset:
                        mn = min(cnt[v] for v in nbr1[nb])
                        heapq.heappush(heap, (-mn, nb))

    best_set = set(initial_size and chosen or [])
    best_counts = coverage_counts(best_set)
    best_size = len(best_set)

    cur_set = set(chosen)
    cur_counts = coverage_counts(cur_set)
    remove_redundants(cur_set, cur_counts)
    if len(cur_set) < best_size:
        best_size = len(cur_set)
        best_set = set(cur_set)
        best_counts = cur_counts

    def rollback(add, removed, cur_set, cur_counts):
        cur_set.discard(add)
        for v in nbr1[add]:
            cur_counts[v] -= 1
        for c in removed:
            cur_set.add(c)
            for v in nbr1[c]:
                cur_counts[v] += 1

    # Local search: prefer swaps (move a codeword to a new location), then additions.
    while time.time() < deadline - 2.0:
        # --- try a swap: replace a random codeword c with a random word w ---
        if cur_set:
            c = random.choice(list(cur_set))
            w = random.randint(0, N - 1)
            if w not in cur_set and w != c:
                # apply
                removed = []
                for v in nbr1[c]:
                    cur_counts[v] -= 1
                cur_set.discard(c)
                cur_set.add(w)
                for v in nbr1[w]:
                    cur_counts[v] += 1
                # remove redundants
                heap = []
                for cc in cur_set:
                    mn = min(cur_counts[v] for v in nbr1[cc])
                    heapq.heappush(heap, (-mn, cc))
                while heap:
                    if time.time() > deadline - 1.0:
                        break
                    neg, cc = heapq.heappop(heap)
                    if cc not in cur_set:
                        continue
                    if (-neg) != min(cur_counts[v] for v in nbr1[cc]):
                        continue
                    if (-neg) > 1:
                        cur_set.remove(cc)
                        for v in nbr1[cc]:
                            cur_counts[v] -= 1
                        removed.append(cc)
                        for nb in nbr1[cc]:
                            if nb in cur_set:
                                mn = min(cur_counts[v] for v in nbr1[nb])
                                heapq.heappush(heap, (-mn, nb))
                new_size = len(cur_set)
                if new_size < best_size:
                    best_size = new_size
                    best_set = set(cur_set)
                    best_counts = cur_counts
                elif new_size == best_size:
                    if random.random() < 0.3:
                        pass  # accept, wander
                    else:
                        rollback(w, [c] + removed, cur_set, cur_counts)
                else:
                    rollback(w, [c] + removed, cur_set, cur_counts)
            continue

        # --- try an addition ---
        w = random.randint(0, N - 1)
        if w in cur_set:
            continue
        cur_set.add(w)
        for v in nbr1[w]:
            cur_counts[v] += 1
        removed = []
        heap = []
        for cc in cur_set:
            mn = min(cur_counts[v] for v in nbr1[cc])
            heapq.heappush(heap, (-mn, cc))
        while heap:
            if time.time() > deadline - 1.0:
                break
            neg, cc = heapq.heappop(heap)
            if cc not in cur_set:
                continue
            if (-neg) != min(cur_counts[v] for v in nbr1[cc]):
                continue
            if (-neg) > 1:
                cur_set.remove(cc)
                for v in nbr1[cc]:
                    cur_counts[v] -= 1
                removed.append(cc)
                for nb in nbr1[cc]:
                    if nb in cur_set:
                        mn = min(cur_counts[v] for v in nbr1[nb])
                        heapq.heappush(heap, (-mn, nb))
        new_size = len(cur_set)
        if new_size < best_size:
            best_size = new_size
            best_set = set(cur_set)
            best_counts = cur_counts
        elif new_size == best_size:
            if random.random() < 0.2:
                pass  # accept to wander
            else:
                rollback(w, removed, cur_set, cur_counts)
        else:
            rollback(w, removed, cur_set, cur_counts)

    return [to_base3(w) for w in best_set]
