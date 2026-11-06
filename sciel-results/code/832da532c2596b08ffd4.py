import time
import random

N = 256
MAX_SUM = 2 * N  # 512


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s

    # ---------- fast helpers ----------

    def is_sidon_set(S):
        used = set()
        for a in S:
            for b in S:
                if a <= b:
                    s = a + b
                    if s in used:
                        return False
                    used.add(s)
        return True

    def compute_sums(S):
        sums = set()
        for a in S:
            for b in S:
                if a <= b:
                    sums.add(a + b)
        return sums

    def can_add(x, S, sums):
        # x can be added iff no sum x+y (y in S) and 2x is in sums
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def new_sums_after_add(x, S, sums):
        ns = set(sums)
        for y in S:
            ns.add(x + y)
        ns.add(2 * x)
        return ns

    def count_unaddable(S, sums):
        # count elements in [1,N] not in S that are addable
        Sset = set(S)
        cnt = 0
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if can_add(x, S, sums):
                cnt += 1
        return cnt

    def is_maximal(S, sums):
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if can_add(x, S, sums):
                return False
        return True

    def make_maximal(S, sums):
        # Add elements (greedy, most-blocking-first) until maximal
        S = list(S)
        sums = set(sums)
        Sset = set(S)
        while True:
            # find all addable candidates
            addable = []
            for x in range(1, N + 1):
                if x in Sset:
                    continue
                if can_add(x, S, sums):
                    addable.append(x)
            if not addable:
                break
            # pick the one that blocks the most other addable candidates
            best_x = None
            best_blocks = -1
            for x in addable:
                ns = new_sums_after_add(x, S, sums)
                blocks = 0
                for y in addable:
                    if y == x:
                        continue
                    if not can_add(y, S + [x], ns):
                        blocks += 1
                if blocks > best_blocks:
                    best_blocks = blocks
                    best_x = x
            if best_x is None:
                break
            S.append(best_x)
            sums = new_sums_after_add(best_x, S[:-1], sums)
            Sset.add(best_x)
        return S, sums

    # ---------- build initial maximal set ----------

    def build_random_maximal():
        S = []
        sums = set()
        # random permutation of [1,N]
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        for x in perm:
            if can_add(x, S, sums):
                S.append(x)
                sums = new_sums_after_add(x, S[:-1], sums)
        S, sums = make_maximal(S, sums)
        return S, sums

    # ---------- lazy deletion / minimum-vertex-cover-like reduction ----------

    def lazy_delete_reduce(S, sums, time_limit):
        S = list(S)
        sums = set(sums)
        Sset = set(S)
        while time.time() < time_limit:
            if len(S) <= 1:
                break
            best_x = None
            best_new_unaddable = None
            best_cost = None
            for x in S:
                # remove x
                new_S = [y for y in S if y != x]
                new_sums = compute_sums(new_S)
                # count unaddable (elements not in new_S that are addable)
                new_Sset = set(new_S)
                unaddable = 0
                for z in range(1, N + 1):
                    if z in new_Sset:
                        continue
                    if can_add(z, new_S, new_sums):
                        unaddable += 1
                # cost: number of elements that become addable after removal
                # (these are the "new" unaddable). We want to minimize this.
                # Actually we want to remove the element that, when removed,
                # makes the FEWEST new elements addable (so maximality is
                # most preserved), i.e. minimize unaddable.
                if best_cost is None or unaddable < best_cost:
                    best_cost = unaddable
                    best_x = x
                    best_new_unaddable = unaddable
            if best_x is None:
                break
            # Remove best_x and re-maximize
            S = [y for y in S if y != best_x]
            sums = compute_sums(S)
            S, sums = make_maximal(S, sums)
        return S, sums

    # ---------- local search: swap to reduce size ----------

    def local_search_swap(S, sums, time_limit):
        S = list(S)
        sums = set(sums)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            Sset = set(S)
            for i in range(len(S)):
                if time.time() > time_limit:
                    break
                for j in range(1, N + 1):
                    if j in Sset:
                        continue
                    # try swap S[i] -> j
                    new_S = list(S)
                    new_S[i] = j
                    if len(set(new_S)) != len(new_S):
                        continue
                    if not is_sidon_set(new_S):
                        continue
                    new_sums = compute_sums(new_S)
                    # check maximality
                    if not is_maximal(new_S, new_sums):
                        # make maximal and see if size decreases
                        new_S2, new_sums2 = make_maximal(new_S, new_sums)
                        if len(new_S2) < len(S):
                            S = new_S2
                            sums = new_sums2
                            improved = True
                            break
                if improved:
                    break
        return S, sums

    # ---------- main search loop ----------

    best_size = N + 1
    best_set = None

    # First build a few initial maximal sets and reduce them
    n_initial = 3
    for k in range(n_initial):
        if time.time() > deadline:
            break
        S, sums = build_random_maximal()
        S, sums = lazy_delete_reduce(S, sums, deadline - 2.0)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]

    # Continue with more random builds + reductions
    while time.time() < deadline:
        S, sums = build_random_maximal()
        S, sums = lazy_delete_reduce(S, sums, deadline - 2.0)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
        # occasional swap improvement on the best
        if best_set is not None and time.time() < deadline - 3.0:
            S2, sums2 = local_search_swap(best_set, compute_sums(best_set), deadline - 2.0)
            if len(S2) < best_size:
                best_size = len(S2)
                best_set = S2[:]
        if best_size <= 14:
            break

    if best_set is None:
        # fallback
        S, sums = build_random_maximal()
        best_set = S

    return best_set
