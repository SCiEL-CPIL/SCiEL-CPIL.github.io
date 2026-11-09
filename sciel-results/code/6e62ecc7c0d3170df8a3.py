import time
import random
import sys
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill
    N = 32
    MAXV = 512

    def fitness(arr):
        """Count distinct d>0 with a, a+d, a+2d in set. arr: list of N ints."""
        mask = 0
        for x in arr:
            mask |= 1 << x
        diffs = 0
        for i in range(N):
            a = arr[i]
            for j in range(i + 1, N):
                c = arr[j]
                s = a + c
                if (s & 1) == 0:
                    b = s >> 1
                    if (mask >> b) & 1:
                        d = c - b
                        if d > 0:
                            diffs += 1
        return diffs

    def valid(arr):
        if len(arr) != N:
            return False
        if len(set(arr)) != N:
            return False
        for x in arr:
            if x < 1 or x > MAXV:
                return False
        return True

    def repair(arr):
        """Ensure N distinct ints in [1,MAXV]."""
        arr = list(arr)
        if len(arr) != N:
            # pad or trim
            if len(arr) < N:
                used = set(arr)
                for x in range(1, MAXV + 1):
                    if len(arr) >= N:
                        break
                    if x not in used:
                        arr.append(x)
                        used.add(x)
            else:
                arr = arr[:N]
        # fix duplicates / out-of-range
        used = set()
        for i, x in enumerate(arr):
            if x < 1 or x > MAXV or x in used:
                # find a fresh value
                cand = random.randint(1, MAXV)
                tries = 0
                while cand in used and tries < 200:
                    cand = random.randint(1, MAXV)
                    tries += 1
                if cand in used:
                    # scan
                    for c in range(1, MAXV + 1):
                        if c not in used:
                            cand = c
                            break
                arr[i] = cand
                used.add(cand)
            else:
                used.add(x)
        return tuple(sorted(arr))

    def random_individual():
        return tuple(sorted(random.sample(range(1, MAXV + 1), N)))

    def structured_candidates():
        cands = []
        # 1) Full arithmetic progressions (max d = (MAXV-1)//2 = 255)
        for step in range(1, 256):
            base = 1
            arr = [base + i * step for i in range(N)]
            if arr[-1] <= MAXV:
                cands.append(tuple(arr))
        # 2) Grid: rows x cols
        for rows in range(2, N + 1):
            cols = N // rows
            if rows * cols != N:
                continue
            for m in range(1, MAXV // rows + 1):
                arr = []
                ok = True
                for r in range(rows):
                    for c in range(cols):
                        v = r * m + c + 1
                        if v > MAXV:
                            ok = False
                            break
                        arr.append(v)
                    if not ok:
                        break
                if ok and len(set(arr)) == N:
                    cands.append(tuple(sorted(arr)))
        # 3) Two-segment / block constructions
        for a in range(1, 10):
            for b in range(1, 10):
                arr = []
                for i in range(N):
                    arr.append(a * (i % b) + b * (i // b) + 1)
                if max(arr) <= MAXV and len(set(arr)) == N:
                    cands.append(tuple(sorted(arr)))
        # 4) Random structured: random step + random base, keep if fits
        for _ in range(200):
            step = random.randint(1, 250)
            base = random.randint(1, 5)
            arr = [base + i * step for i in range(N)]
            if arr[-1] <= MAXV:
                cands.append(tuple(arr))
        # 5) Random sets
        for _ in range(200):
            cands.append(random_individual())
        # dedup
        seen = set()
        uniq = []
        for c in cands:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def local_search(ind, time_limit):
        """Hill-climb with swap and replace moves. Returns best tuple found."""
        cur = list(ind)
        cur_score = fitness(cur)
        best = tuple(cur)
        best_score = cur_score
        t0 = time.time()
        no_improve = 0
        max_no_improve = 4000
        while time.time() - t0 < time_limit:
            # choose move type
            if random.random() < 0.7:
                # swap two positions
                i, j = random.sample(range(N), 2)
                cur[i], cur[j] = cur[j], cur[i]
            else:
                # replace one element with a random value
                i = random.randrange(N)
                old = cur[i]
                new = random.randint(1, MAXV)
                tries = 0
                while new == old and tries < 50:
                    new = random.randint(1, MAXV)
                    tries += 1
                cur[i] = new
            ns = fitness(cur)
            if ns > cur_score:
                cur_score = ns
                no_improve = 0
                if ns > best_score:
                    best_score = ns
                    best = tuple(cur)
            else:
                # revert
                if random.random() < 0.7:
                    i, j = random.sample(range(N), 2)
                    cur[i], cur[j] = cur[j], cur[i]
                else:
                    i = random.randrange(N)
                    # we don't track old; just resample to keep valid
                    # simpler: revert by re-picking? Instead, accept a small probability of accepting worse to escape? 
                    # For pure hill climb, revert: we need old value. Let's store it.
                    pass
                # To properly revert replace, we need old value; handle below.
                no_improve += 1
                if no_improve > max_no_improve:
                    # perturb: do a few random changes
                    for _ in range(3):
                        k = random.randrange(N)
                        cur[k] = random.randint(1, MAXV)
                    # repair duplicates
                    used = set()
                    for k in range(N):
                        if cur[k] in used or cur[k] < 1 or cur[k] > MAXV:
                            c = random.randint(1, MAXV)
                            while c in used:
                                c = random.randint(1, MAXV)
                            cur[k] = c
                            used.add(c)
                        else:
                            used.add(cur[k])
                    cur_score = fitness(cur)
                    no_improve = 0
        return best, best_score

    # --- Main search ---
    best_overall = None
    best_overall_score = -1

    # Phase 1: evaluate structured candidates, keep top K
    cands = structured_candidates()
    scored = []
    for c in cands:
        if not valid(c):
            c = repair(c)
        s = fitness(c)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:100]]
    for s, c in scored[:100]:
        if s > best_overall_score:
            best_overall_score = s
            best_overall = c

    # Phase 2: multi-restart local search from diverse starts
    t0 = time.time()
    # Allocate time: use most of the budget for local search
    # Number of restarts adaptive
    restarts_done = 0
    while time.time() < deadline - 10.0:
        # pick a start: from top candidates or random
        if random.random() < 0.7 and top:
            start_ind = top[random.randrange(len(top))]
        else:
            start_ind = random_individual()
        # time per restart: adaptive
        elapsed = time.time() - t0
        remaining = deadline - 10.0 - time.time()
        # target: each restart ~ 1-3 seconds, more if few restarts
        if restarts_done < 10:
            tl = 2.0
        elif restarts_done < 40:
            tl = 1.5
        else:
            tl = max(0.5, min(1.0, remaining / max(1, (200 - restarts_done))))
        if time.time() + tl > deadline - 5.0:
            tl = max(0.2, deadline - 5.0 - time.time())
        if tl <= 0:
            break
        best_ind, best_sc = local_search(list(start_ind), tl)
        if best_sc > best_overall_score:
            best_overall_score = best_sc
            best_overall = best_ind
        # update top list with new best
        top.append(best_ind)
        top.sort(key=lambda x: fitness(x), reverse=True)
        top = top[:100]
        restarts_done += 1

    # Phase 3: final polish with more iterations on the single best
    if best_overall is not None and time.time() < deadline - 5.0:
        tl = max(0.5, deadline - 5.0 - time.time())
        best_ind, best_sc = local_search(list(best_overall), tl)
        if best_sc > best_overall_score:
            best_overall_score = best_sc
            best_overall = best_ind

    # Ensure we always return a valid 32-set
    if best_overall is None:
        best_overall = random_individual()
    # Final validation
    if not valid(best_overall):
        best_overall = repair(best_overall)
    return list(best_overall)

if __name__ == "__main__":
    print(construct())
