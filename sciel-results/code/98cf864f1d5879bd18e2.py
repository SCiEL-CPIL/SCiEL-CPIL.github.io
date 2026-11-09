import random
import time

def construct():
    start = time.time()
    DEADLINE = start + 115.0

    N = 1000
    rng = random.Random(12345)

    # ---- Efficient B[2] bookkeeping ----
    # cnt[s] = number of pairs (a,b), a<=b, a+b==s, with both in set.
    # cap[s] = 2 - cnt[s]; valid iff all cap[s] >= 0.
    # neg = number of sums with cap < 0 (i.e. cnt > 2).
    def build_state(A):
        cnt = {}
        for i, a in enumerate(A):
            for b in A[i:]:
                s = a + b
                cnt[s] = cnt.get(s, 0) + 1
        neg = 0
        for v in cnt.values():
            if v > 2:
                neg += 1
        return cnt, neg

    def try_add(A, cnt, neg, x):
        # returns (new_cnt, new_neg) if adding x keeps neg unchanged-ish, else None
        # We do a tentative add and count how many sums newly exceed 2.
        if x in A:
            return None
        touched = {}
        ok = True
        # pairs (x, a) for a in A with a <= x, and (x,x)
        # Actually all pairs involving x: for each a in A with a<=x: (a,x); plus (x,x)
        for a in A:
            if a > x:
                s = x + a
            else:
                s = x + a
            # both cases same sum
            c = cnt.get(s, 0)
            if c >= 2:
                ok = False
                break
            touched[s] = c
        s2 = 2 * x
        c2 = cnt.get(s2, 0)
        if c2 >= 2:
            ok = False
        if not ok:
            return None
        # apply
        new_cnt = dict(cnt)
        for s, c in touched.items():
            new_cnt[s] = c + 1
        new_cnt[s2] = c2 + 1
        new_neg = neg  # adding only increases counts; could create new neg
        # recompute neg incrementally: a sum becomes neg if its count goes 2->3
        for s in list(touched.keys()) + [s2]:
            # count after add
            # we need to know if it crossed 2
            pass
        # Simpler: just recompute neg from scratch is too slow; do incremental:
        new_neg = neg
        for s in list(touched.keys()):
            if touched[s] == 2:
                new_neg += 1
        if c2 == 2:
            new_neg += 1
        A.append(x)
        return new_cnt, new_neg

    def try_remove(A, cnt, neg, x):
        if x not in A:
            return None
        touched = {}
        for a in A:
            if a == x:
                continue
            s = x + a
            touched[s] = cnt.get(s, 0)
        s2 = 2 * x
        c2 = cnt.get(s2, 0)
        new_cnt = dict(cnt)
        new_neg = neg
        for s, c in touched.items():
            new_cnt[s] = c - 1
            if c == 3:
                new_neg -= 1
        new_cnt[s2] = c2 - 1
        if c2 == 3:
            new_neg -= 1
        A.remove(x)
        return new_cnt, new_neg

    # ---- Initial solution: greedy ----
    A = []
    cnt = {}
    neg = 0
    for x in range(1, N + 1):
        # check
        bad = False
        for a in A:
            s = x + a
            if cnt.get(s, 0) >= 2:
                bad = True
                break
        if not bad and cnt.get(2 * x, 0) >= 2:
            bad = True
        if not bad:
            for a in A:
                s = x + a
                cnt[s] = cnt.get(s, 0) + 1
            cnt[2 * x] = cnt.get(2 * x, 0) + 1
            A.append(x)

    best_set = A[:]
    best_size = len(A)

    # ---- Simulated annealing: random add/remove, accept based on neg and size ----
    # Objective: maximize size, but must end with neg == 0.
    # During search we allow temporary neg>0 but penalize.
    # Accept move if it improves (size up with neg same, or neg down), else with prob based on temperature.

    A_set = set(A)
    cur_size = len(A)
    cur_neg = neg
    T0 = 5.0
    Tend = 0.05
    steps_total = 0

    # Precompute a list of candidates = 1..N
    candidates = list(range(1, N + 1))

    while time.time() < DEADLINE:
        steps_total += 1
        # temperature schedule (linear in steps, capped by deadline)
        frac = min(1.0, steps_total / 200000.0)
        T = T0 * (Tend / T0) ** frac

        # choose a move: add or remove
        if rng.random() < 0.5 and cur_size < N:
            # add
            x = candidates[rng.randrange(N)]
            if x in A_set:
                continue
            res = try_add(A, cnt, cur_neg, x)
            if res is None:
                continue
            new_cnt, new_neg = res
            new_size = cur_size + 1
            # score = size - penalty*neg
            penalty = 3.0
            old_score = cur_size - penalty * cur_neg
            new_score = new_size - penalty * new_neg
            d = new_score - old_score
            if d >= 0 or rng.random() < (2.718281828 ** (d / T)):
                # accept
                cnt = new_cnt
                cur_neg = new_neg
                cur_size = new_size
                A_set.add(x)
                if cur_neg == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_set = A[:]
            else:
                # reject: undo
                A.pop()
        else:
            # remove
            if cur_size == 0:
                continue
            x = A[rng.randrange(cur_size)]
            res = try_remove(A, cnt, cur_neg, x)
            if res is None:
                continue
            new_cnt, new_neg = res
            new_size = cur_size - 1
            penalty = 3.0
            old_score = cur_size - penalty * cur_neg
            new_score = new_size - penalty * new_neg
            d = new_score - old_score
            if d >= 0 or rng.random() < (2.718281828 ** (d / T)):
                cnt = new_cnt
                cur_neg = new_neg
                cur_size = new_size
                A_set.discard(x)
                if cur_neg == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_set = A[:]
            else:
                # undo: re-add x (we already removed it)
                A.append(x)
                # restore cnt and neg
                # recompute touched
                for a in A:
                    if a == x:
                        continue
                    s = x + a
                    cnt[s] = cnt.get(s, 0) + 1
                s2 = 2 * x
                cnt[s2] = cnt.get(s2, 0) + 1
                # recompute neg fully (cheap-ish) to be safe
                cur_neg = 0
                for v in cnt.values():
                    if v > 2:
                        cur_neg += 1
                cur_size = len(A)
                A_set.add(x)

    return best_set
