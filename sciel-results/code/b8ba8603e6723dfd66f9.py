# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _build_smooth(limit):
    smooth = []
    for n in range(1, limit + 1):
        y = n
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(n)
    return smooth

def _is_antichain(terms, comp_idx):
    for a in range(len(terms)):
        for b in range(a + 1, len(terms)):
            if not comp_idx[terms[a]][terms[b]]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 52.0
    max_target = 128
    smooth = _build_smooth(max_target)
    S = len(smooth)
    idx_of = {v: i for i, v in enumerate(smooth)}
    comp = [[True] * S for _ in range(S)]
    for i in range(S):
        a = smooth[i]
        for j in range(i + 1, S):
            b = smooth[j]
            if b % a == 0 or a % b == 0:
                comp[i][j] = comp[j][i] = False

    # For each target, candidate indices sorted by value
    cand_by_target = {}
    for n in range(1, max_target + 1):
        cand_by_target[n] = [i for i in range(S) if smooth[i] <= n]

    # DFS with pruning, returns list of values or None
    def solve_target(n, start_count):
        cand = cand_by_target[n]
        m = len(cand)
        best = [None]
        for count in range(start_count, 31):
            if count > m:
                break
            # lower bound: sum of smallest `count` smooth <= n
            lb = sum(smooth[cand[t]] for t in range(count))
            if lb > n:
                break
            # upper bound: sum of largest `count` smooth <= n
            ub = sum(smooth[cand[m - 1 - t]] for t in range(count))
            if ub < n:
                continue
            chosen = []
            def dfs(pos, cur_sum):
                if time.time() > deadline:
                    return False
                remaining = count - len(chosen)
                if remaining == 0:
                    if cur_sum == n:
                        best[0] = [smooth[i] for i in chosen]
                        return True
                    return False
                if cur_sum > n:
                    return False
                if m - pos < remaining:
                    return False
                # min possible add
                rem_min = 0
                for t in range(remaining):
                    rem_min += smooth[cand[pos + t]]
                if cur_sum + rem_min > n:
                    return False
                # max possible add
                rem_max = 0
                for t in range(remaining):
                    rem_max += smooth[cand[m - 1 - t]]
                if cur_sum + rem_max < n:
                    return False
                last = chosen[-1] if chosen else -1
                for idx in range(pos, m):
                    i = cand[idx]
                    if last != -1 and not comp[last][i]:
                        continue
                    chosen.append(i)
                    if dfs(idx + 1, cur_sum + smooth[i]):
                        return True
                    chosen.pop()
                return False
            if dfs(0, 0):
                return best[0]
        return None

    answers = [None] * max_target
    # Solve in order; prioritize getting a valid answer for every target quickly,
    # then refine counts if time remains.
    # Phase 1: get any valid answer for each target (count 1..30)
    for n in range(1, max_target + 1):
        if time.time() > deadline - 5.0:
            break
        ans = solve_target(n, 1)
        if ans is None:
            ans = _fallback(n, smooth, comp, deadline)
        answers[n - 1] = ans

    # Ensure all filled
    for n in range(1, max_target + 1):
        if answers[n - 1] is None:
            answers[n - 1] = _fallback(n, smooth, comp, deadline)
            if answers[n - 1] is None:
                answers[n - 1] = [n] if n <= max_target else [1]

    # Phase 2: refine to smaller counts if time remains
    if time.time() < deadline - 3.0:
        for n in range(1, max_target + 1):
            if time.time() > deadline - 3.0:
                break
            cur = answers[n - 1]
            cur_count = len(cur) if cur else 30
            if cur_count > 1:
                better = solve_target(n, 1)
                if better is not None and len(better) < cur_count:
                    answers[n - 1] = better

    return answers

def _fallback(n, smooth, comp, deadline):
    from itertools import combinations
    S = len(smooth)
    cand = [i for i in range(S) if smooth[i] <= n]
    for count in range(1, 31):
        if count > len(cand):
            break
        for combo in combinations(cand, count):
            if time.time() > deadline:
                return None
            if sum(smooth[i] for i in combo) != n:
                continue
            ok = True
            for a in range(count):
                for b in range(a + 1, count):
                    if not comp[combo[a]][combo[b]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return [smooth[i] for i in combo]
    return None
# EVOLVE-BLOCK-END
