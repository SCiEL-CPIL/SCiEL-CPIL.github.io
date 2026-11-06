"""Rewritten: greedy + 1-swap local search for max independent set."""
import time
import random
import math

_DEADLINE = 85.0  # safety margin under 90s wall-clock
_START = time.time()

# Precompute conflict structure
_N = 256
# squares up to 512
_SQ = set()
for _s in range(1, 23):  # 1^2..22^2=484, 23^2=529>512
    _SQ.add(_s * _s)

# conflict[a][b] for a<=b: is a+b a square?
# Diagonal: 2a is square => a in {2,8,18,32,50,72,98,128,162,200,242}
# We represent conflicts as: for each a, the set of b (b>=a) that conflict with a
# Use list of lists for adjacency (only b>=a stored for a, and symmetric for b<a)
_adj = [set() for _ in range(_N + 1)]
for _a in range(1, _N + 1):
    for _b in range(_a, _N + 1):
        if (_a + _b) in _SQ:
            _adj[_a].add(_b)
            if _b != _a:
                _adj[_b].add(_a)


def _is_valid(S):
    """Check S is independent (no a<=b in S with a+b square)."""
    L = sorted(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i, n):
            b = L[j]
            if (a + b) in _SQ:
                return False
    return True


def _greedy_order(order, S, conflicts):
    """Try to add each element in order; return new S and updated conflicts.
    conflicts: dict a -> set of elements in S that conflict with a (for a not in S).
    We maintain S as a set, and for each candidate a (not in S) we track count of conflicts in S.
    """
    # conflicts: dict from a (not in S) to number of neighbors in S
    new_S = set()
    # rebuild conflicts from scratch each call is O(N^2) but N=256 fine
    # Actually let's do it efficiently: for each a in order, check if all neighbors not in S
    for a in order:
        if a in new_S:
            continue
        # check conflicts with new_S
        ok = True
        for nb in _adj[a]:
            if nb in new_S:
                ok = False
                break
        if ok:
            new_S.add(a)
    return new_S


def _local_search(S, rng, deadline):
    """1-swap local search. S is a set. Try replacing one element with one not in S."""
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # For each a in S, try to remove it and add some b not in S
        # We'll do a randomized pass
        S_list = list(S)
        rng.shuffle(S_list)
        for a in S_list:
            if time.time() > deadline:
                break
            # Try removing a
            S_minus = S - {a}
            # Find candidates b not in S that can be added to S_minus
            # and prefer ones that allow further growth
            best_b = None
            best_gain = 0
            # Iterate candidates not in S
            for b in range(1, _N + 1):
                if b in S_minus:
                    continue
                # check b against S_minus
                ok = True
                for nb in _adj[b]:
                    if nb in S_minus:
                        ok = False
                        break
                if ok:
                    # net change: -1 (remove a) +1 (add b) = 0, but maybe enables more
                    # For 1-swap, size stays same. We need 2-swap or pure add to grow.
                    # Instead, let's try: remove a, then greedily add as many as possible
                    # This is expensive. Let's do a simpler hill-climb: 
                    # Actually for growth, we should try "remove a, then add b and c" 
                    # Let's do: after removing a, try to add any single b (size same) 
                    # then try to add another c (size +1). 
                    # Simpler: just try to add b to S (without removing) - that's pure add.
                    pass
            # Let's do pure add first: try to add any b not in S
            # (handled by the add loop below)
            # For swap: remove a, add b, size same, but may enable more adds later.
            # We'll do: for each a, for each b not in S, if S - {a} + {b} is valid, 
            # accept if it leads to more adds. Too expensive.
            # Simplified: do "add if possible" and "swap if it enables add".
            # Let's just do: try to add b to S. If can't, try swap a->b then add c.
            # For now, pure add:
            pass
        # Pure add pass
        for b in range(1, _N + 1):
            if b in S:
                continue
            ok = True
            for nb in _adj[b]:
                if nb in S:
                    ok = False
                    break
            if ok:
                S.add(b)
                improved = True
        if not improved:
            # Try swaps: for each a in S, for each b not in S, if S-{a}+{b} valid,
            # do it (size same), then re-run add pass.
            # To make progress, only do swap if it enables a subsequent add.
            # Expensive but N=256, S~80, candidates~176. 
            # Let's do a limited number of swap attempts.
            swap_done = False
            S_list2 = list(S)
            rng.shuffle(S_list2)
            for a in S_list2:
                if time.time() > deadline:
                    break
                S_minus = S - {a}
                for b in range(1, _N + 1):
                    if b in S_minus:
                        continue
                    # check b against S_minus
                    ok = True
                    for nb in _adj[b]:
                        if nb in S_minus:
                            ok = False
                            break
                    if ok:
                        # S_minus + {b} is valid. Now check if we can add c to it.
                        S_try = S_minus | {b}
                        # try to add any c
                        for c in range(1, _N + 1):
                            if c in S_try:
                                continue
                            okc = True
                            for nb in _adj[c]:
                                if nb in S_try:
                                    okc = False
                                    break
                            if okc:
                                # accept swap+add
                                S = S_try | {c}
                                swap_done = True
                                improved = True
                                break
                        if swap_done:
                            break
                if swap_done:
                    break
    return S


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    rng = random.Random(42)
    best = set()
    best_size = 0

    # Multiple randomized restarts
    deadline = _DEADLINE
    while time.time() < deadline - 2.0:
        # Random order
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        S = _greedy_order(order, set(), None)
        if len(S) > best_size:
            best = set(S)
            best_size = len(S)
        # Local search
        S2 = _local_search(S, rng, deadline - 1.0)
        if len(S2) > best_size:
            best = set(S2)
            best_size = len(S2)
        # Occasionally do local search on best
        if time.time() < deadline - 5.0 and best_size > 0:
            S3 = _local_search(set(best), rng, deadline - 1.0)
            if len(S3) > best_size:
                best = set(S3)
                best_size = len(S3)

    # Final: try to add to best
    for b in range(1, _N + 1):
        if b in best:
            continue
        ok = True
        for nb in _adj[b]:
            if nb in best:
                ok = False
                break
        if ok:
            best.add(b)

    result = sorted(best)
    if not result:
        result = [1]
    return result


if __name__ == "__main__":
    res = construct()
    print("Size:", len(res))
    # Verify
    L = res
    n = len(L)
    valid = True
    for i in range(n):
        for j in range(i, n):
            if (L[i] + L[j]) in _SQ:
                valid = False
                print("Conflict:", L[i], L[j])
    print("Valid:", valid)
