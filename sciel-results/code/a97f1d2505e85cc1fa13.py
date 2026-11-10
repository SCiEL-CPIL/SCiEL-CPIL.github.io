"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_DEADLINE = 80.0
_NODE_BUDGET = 2_000_000  # cap on DFS nodes per _is_valid call

def _can_sum_to(fracs, target, budget):
    """Pruned DFS: can a subset of fracs sum to target? fracs sorted desc."""
    fracs = sorted(fracs, reverse=True)
    n = len(fracs)
    # suffix sums for pruning
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + fracs[i]
    total = suffix[0]
    if total < target:
        return False
    count = [0]
    def dfs(idx, cur):
        if count[0] >= budget:
            return None  # signal abort
        count[0] += 1
        if cur == target:
            return True
        if idx >= n:
            return False
        if cur + suffix[idx] < target:
            return False
        if cur > target:
            return False
        # take
        r = dfs(idx + 1, cur + fracs[idx])
        if r is not None:
            return r
        # skip
        r = dfs(idx + 1, cur)
        if r is not None:
            return r
        return False
    res = dfs(0, Fraction(0))
    return res is True

def _is_valid(S, budget=_NODE_BUDGET):
    S_list = list(S)
    for x in S_list:
        others = [Fraction(1, y) for y in S_list if y != x]
        target = Fraction(1, x)
        if _can_sum_to(others, target, budget):
            return False
    return True

def _try_greedy(order, deadline):
    S = []
    for x in order:
        if time.time() > deadline:
            break
        trial = S + [x]
        if _is_valid(trial):
            S = trial
    return S

def construct():
    start = time.time()
    deadline = start + _DEADLINE

    best = list(range(13, 25))  # fallback size 12

    for order in [list(range(24, 0, -1)), list(range(1, 25))]:
        if time.time() > deadline:
            break
        cand = _try_greedy(order, deadline)
        if len(cand) > len(best):
            best = cand

    if time.time() > deadline:
        return best

    # Extend best by trying to add missing smaller elements
    missing = [x for x in range(1, 25) if x not in best]
    for x in missing:
        if time.time() > deadline:
            break
        trial = best + [x]
        if _is_valid(trial):
            best = trial

    if time.time() > deadline:
        return best

    # Try full blocks from various starts
    for sv in [8, 7, 6, 5, 4]:
        if time.time() > deadline:
            break
        block = list(range(sv, 25))
        if _is_valid(block) and len(block) > len(best):
            best = block

    return best
# EVOLVE-BLOCK-END

