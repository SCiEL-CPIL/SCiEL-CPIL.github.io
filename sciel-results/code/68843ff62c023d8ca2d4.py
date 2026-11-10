"""Rewritten program for large integer sets avoiding equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

_N = 128
_gcd_table = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_table[_i][_j] = _g
        _gcd_table[_j][_i] = _g

# Precompute bad triples and build conflict structure
# For each element x, store which pairs (a,b) with a<b cannot coexist with x
# i.e., {a,b,x} forms a bad triple
_bad_pairs = [set() for _ in range(_N + 1)]
for a in range(1, _N + 1):
    for b in range(a + 1, _N + 1):
        for c in range(b + 1, _N + 1):
            g_ab = _gcd_table[a][b]
            g_ac = _gcd_table[a][c]
            g_bc = _gcd_table[b][c]
            if g_ab == g_ac == g_bc:
                _bad_pairs[a].add(b * (_N + 1) + c)
                _bad_pairs[b].add(a * (_N + 1) + c)
                _bad_pairs[c].add(a * (_N + 1) + b)

def _encode_pair(a, b):
    if a > b:
        a, b = b, a
    return a * (_N + 1) + b

def _is_valid_addition(x, selected_pairs):
    """Check if x can be added to the set represented by selected_pairs."""
    for p in _bad_pairs[x]:
        if p in selected_pairs:
            return False
    return True

def _greedy_construct():
    """Greedy construction with multiple random restarts."""
    best_set = []
    best_size = 0
    deadline = time.time() + 55.0  # Leave 5s for final processing

    for trial in range(200):
        if time.time() > deadline:
            break

        order = list(range(1, _N + 1))
        random.shuffle(order)

        selected = []
        selected_pairs = set()

        for x in order:
            if _is_valid_addition(x, selected_pairs):
                selected.append(x)
                for y in selected[:-1]:
                    selected_pairs.add(_encode_pair(x, y))

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

        # Local improvement: try to add any remaining element
        remaining = [x for x in range(1, _N + 1) if x not in selected]
        random.shuffle(remaining)
        for x in remaining:
            if _is_valid_addition(x, selected_pairs):
                selected.append(x)
                for y in selected[:-1]:
                    selected_pairs.add(_encode_pair(x, y))

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

    return best_set

def construct():
    deadline = time.time() + 58.0
    best_set = _greedy_construct()
    return sorted(best_set)
# EVOLVE-BLOCK-END
