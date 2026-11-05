"""Factorial decompositions into distinct factors above the index.

Returns 1..12 distinct integers all > 12 whose product is exactly 12!,
minimizing the largest factor.

Strategy
--------
12! = 2^10 * 3^5 * 5^2 * 7 * 11 = 479001600.

We run a branch-and-bound depth-first search over the divisors of 12!.
The state is the remaining product; a "factor" is any divisor d > 12 of the
remaining product that has not been used yet.  We order the candidate
factors by closeness to the balanced (geometric-mean) size so that good
solutions are found early, which tightens the branch-and-bound bound on the
maximum factor.  A geometric-mean lower bound prunes any branch whose most
balanced continuation would already force a factor at or above the current
best maximum.  A hard wall-clock deadline returns the best solution found.

A fast greedy heuristic seeds the search with a decent initial upper bound
so that the DFS prunes aggressively from the very first node.
"""
import time
from math import factorial

_TARGET = factorial(12)  # 479001600
_MIN_FACTOR = 13          # all factors must be > 12
_MAX_COUNT = 12


def _divisors(n):
    """All divisors of n, sorted ascending."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    divs.sort()
    return divs


# All divisors of 12! that are > 12, sorted.  Membership test via set.
_ALL_DIVS = [d for d in _divisors(_TARGET) if d > _MIN_FACTOR]
_DIV_SET = set(_ALL_DIVS)


def _is_valid(factors):
    if not factors or len(factors) > _MAX_COUNT:
        return False
    if any(f <= _MIN_FACTOR - 1 for f in factors):
        return False
    if len(set(factors)) != len(factors):
        return False
    prod = 1
    for f in factors:
        prod *= f
    return prod == _TARGET


def _greedy_seed():
    """Quick balanced greedy: repeatedly split off a divisor near the
    geometric mean of the remaining product for the remaining slots.
    Returns a valid factor list (or None)."""
    # Try several target factor counts; keep the one with the smallest max.
    best = None
    for target in range(7, 13):
        remaining = _TARGET
        used = set()
        cur = []
        ok = True
        for slot in range(target):
            left = target - slot
            if remaining == 1:
                # Not enough product for distinct factors > 12.
                ok = False
                break
            if left == 1:
                d = remaining
            else:
                gm = remaining ** (1.0 / left)
                # Pick the divisor of remaining, > 12, not used, closest to gm.
                best_d = None
                best_key = None
                for d in _ALL_DIVS:
                    if d > remaining:
                        break
                    if remaining % d != 0:
                        continue
                    if d in used:
                        continue
                    key = abs(d - gm)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_d = d
                if best_d is None:
                    ok = False
                    break
                d = best_d
            if d in used:
                ok = False
                break
            used.add(d)
            cur.append(d)
            remaining //= d
        if ok and remaining == 1 and _is_valid(cur):
            mx = max(cur)
            if best is None or mx < max(best):
                best = cur
    return best


class _Searcher:
    def __init__(self, deadline, initial_best):
        self.deadline = deadline
        self.best_max = initial_best
        self.best_factors = None
        self.nodes = 0

    def run(self):
        seed = _greedy_seed()
        if seed is not None:
            mx = max(seed)
            if mx < self.best_max:
                self.best_max = mx
                self.best_factors = list(seed)
        self._dfs(_TARGET, frozenset(), [], 0)
        return self.best_factors

    def _dfs(self, remaining, used, cur, max_so_far):
        self.nodes += 1
        if time.time() > self.deadline:
            return
        if max_so_far >= self.best_max:
            return
        if remaining == 1:
            if cur and len(cur) <= _MAX_COUNT:
                mx = max_so_far if max_so_far > 0 else (max(cur) if cur else 0)
                if mx < self.best_max:
                    self.best_max = mx
                    self.best_factors = list(cur)
            return
        if len(cur) >= _MAX_COUNT:
            return
        # Lower bound: with r slots left (including this one), the largest
        # factor must be at least remaining**(1/r).  If that already meets
        # best_max, no continuation can improve.
        slots_left = _MAX_COUNT - len(cur)
        # We may use fewer than slots_left factors; the tightest lower bound
        # on the eventual max is the geometric mean over the fewest slots
        # that can still hold distinct factors > 12.  Use the full remaining
        # product over 1..slots_left slots; the minimal possible max is at
        # least remaining**(1/slots_left).  Also each factor > 12.
        lb = remaining ** (1.0 / slots_left)
        if lb >= self.best_max:
            return
        # Candidate next factors: divisors of remaining, > 12, not used,
        # and strictly less than best_max (and >= max_so_far is not required;
        # the factor can be smaller than current max).
        # Order by closeness to the balanced size to find good solutions first.
        rem_slots = _MAX_COUNT - len(cur)
        gm = remaining ** (1.0 / rem_slots)
        cands = []
        for d in _ALL_DIVS:
            if d > remaining:
                break
            if d >= self.best_max:
                continue
            if remaining % d != 0:
                continue
            if d in used:
                continue
            cands.append(d)
        if not cands:
            return
        # Sort by closeness to gm (balanced first); ties by value.
        cands.sort(key=lambda d: (abs(d - gm), d))
        # Bound the number of candidates explored per node to control blowup.
        for d in cands:
            if time.time() > self.deadline:
                return
            new_max = max(max_so_far, d)
            if new_max >= self.best_max:
                continue
            self._dfs(remaining // d, used | {d}, cur + [d], new_max)
            if time.time() > self.deadline:
                return


def construct():
    start = time.time()
    # Generous but bounded budget; evaluator kills at 90s, keep a margin.
    deadline = start + 50.0

    # Seed with a good initial upper bound (greedy) so the DFS prunes hard.
    seed = _greedy_seed()
    initial_best = max(seed) if seed is not None else float('inf')

    searcher = _Searcher(deadline, initial_best)
    result = searcher.run()

    # Verify before returning.
    if result is not None and _is_valid(result):
        return result

    # Fallbacks (should not be reached).
    if seed is not None and _is_valid(seed):
        return list(seed)
    return [_TARGET]
