"""Pairwise coprime selection below 512 via exact branch-and-bound on the conflict graph."""
import time
from math import gcd
from fractions import Fraction

_N = 512
_DEADLINE = time.time() + 135.0  # safety margin under the 120s evaluator limit


def _build_graph():
    n = _N - 1  # numbers 1..511
    # weights w[i] = 1/(512 - (i+1))
    w = [Fraction(1, _N - (i + 1)) for i in range(n)]
    # conflict[i] = bitmask of j (j<i) with gcd(i+1, j+1) > 1
    conflict = [0] * n
    for i in range(n):
        ai = i + 1
        mask = 0
        for j in range(i):
            if gcd(ai, j + 1) > 1:
                mask |= (1 << j)
        conflict[i] = mask
    return n, w, conflict


def _greedy_clique(nodes, w, conflict):
    # nodes: list of indices; return (weight, list)
    order = sorted(nodes, key=lambda x: (-w[x], x))
    chosen = []
    for i in order:
        if all(gcd(i + 1, j + 1) == 1 for j in chosen):
            chosen.append(i)
    return sum((w[i] for i in chosen), Fraction(0)), chosen


def _bound(cand, w, conflict):
    # greedy coloring upper bound (weights sorted desc)
    if not cand:
        return Fraction(0)
    order = sorted(cand, key=lambda x: (-w[x], x))
    color_class = []
    for i in order:
        placed = False
        for c in range(len(color_class)):
            if all(gcd(i + 1, j + 1) > 1 for j in color_class[c]):
                color_class[c].append(i)
                placed = True
                break
        if not placed:
            color_class.append([i])
    return sum((w[order[c]] for c in range(len(color_class))), Fraction(0))


def _bbs(cand, w, conflict, cur_w, best_w, best_set, state):
    if time.time() > _DEADLINE:
        state["killed"] = True
        return
    if not cand:
        if cur_w > best_w[0]:
            best_w[0] = cur_w
            best_set[0] = list(state["chosen"])
        return
    ub = _bound(cand, w, conflict)
    if cur_w + ub <= best_w[0]:
        return
    order = sorted(cand, key=lambda x: (-w[x], x))
    for i in order:
        if time.time() > _DEADLINE:
            state["killed"] = True
            return
        if cur_w + w[i] <= best_w[0]:
            continue
        newc = [j for j in order if j != i and gcd(i + 1, j + 1) == 1]
        if not newc:
            if cur_w + w[i] > best_w[0]:
                best_w[0] = cur_w + w[i]
                best_set[0] = list(state["chosen"]) + [i]
            continue
        state["chosen"].append(i)
        _bbs(newc, w, conflict, cur_w + w[i], best_w, best_set, state)
        state["chosen"].pop()
        if state["killed"]:
            return


def construct():
    n, w, conflict = _build_graph()
    state = {"chosen": [], "killed": False}
    best_w = [Fraction(0)]
    best_set = [[]]
    # seed with greedy
    gw, gset = _greedy_clique(list(range(n)), w, conflict)
    best_w[0] = gw
    best_set[0] = list(gset)
    # root: exclude nothing; recurse over all nodes
    _bbs(list(range(n)), w, conflict, Fraction(0), best_w, best_set, state)
    return [i + 1 for i in best_set[0]]


if __name__ == "__main__":
    res = construct()
    print(len(res), float(sum(Fraction(1, _N - a) for a in res)))
