import time
import random
from itertools import combinations

N = 192
MAXK = 48
DEADLINE = 55.0  # seconds, safety margin under 60s/90s

# Precompute: for each a in 1..192, the set of triples (b,c,d) distinct, !=a,
# such that a divides b+c+d.  We store as a list of frozensets or sorted tuples.
# Actually for speed we store for each a a set of "conflict triples" but that's huge.
# Better: for each a, store a set of sums S such that a|S and S can be formed by 3
# distinct elements from [1,192]\{a}.  But we need to check membership of a specific
# triple sum quickly.

# Precompute sum->(b,c,d) not needed. Instead precompute for each a a boolean check:
# given a set of selected elements (as a bitset or list), check if any 3 distinct
# others sum to a multiple of a.

# For speed, precompute for each a the list of all 3-element-combination sums that
# are multiples of a, but that's C(191,3) ~ 1.1M per a * 192 = too big.

# Alternative: precompute for each a a set of "forbidden sum values" that are multiples
# of a in the range [3, 3*191] = [3, 573].  Then for a candidate set, check if any
# 3 distinct others have a sum in that forbidden set.

# For each a, forbidden_sums[a] = {s in [3, 573] : a | s}
# Then checking a triple (b,c,d) just needs (b+c+d) in forbidden_sums[a].

forbidden_sums = [set() for _ in range(N + 1)]
for a in range(1, N + 1):
    # multiples of a in [3, 573]
    s = a
    while s < 3:
        s += a
    while s <= 573:
        forbidden_sums[a].add(s)
        s += a

# Convert to a fast lookup: for each a, a set of forbidden sums.
# For checking, we'll use: sum3 in forbidden_sums[a]

def valid_set(selected):
    """Check if selected set is valid. selected is a list/set of distinct ints."""
    sel = list(selected)
    n = len(sel)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        # check all triples of others
        for b_idx in range(len(others)):
            b = others[b_idx]
            for c_idx in range(b_idx + 1, len(others)):
                c = others[c_idx]
                for d_idx in range(c_idx + 1, len(others)):
                    d = others[d_idx]
                    if (b + c + d) in forbidden_sums[a]:
                        return False
    return True

def count_conflicts(a, selected_set):
    """Count number of triples in selected_set (excluding a) whose sum is divisible by a."""
    others = [x for x in selected_set if x != a]
    cnt = 0
    n = len(others)
    for i in range(n):
        b = others[i]
        for j in range(i + 1, n):
            c = others[j]
            for k in range(j + 1, n):
                d = others[k]
                if (b + c + d) in forbidden_sums[a]:
                    cnt += 1
    return cnt

def construct():
    start_time = time.time()
    
    # Greedy: try to build a large set.
    # Strategy: iterate candidates in a smart order, add if valid.
    # Use multiple random orders and keep best.
    
    best_set = []
    best_size = 0
    
    # First, a deterministic greedy: largest first (as seed) but with a twist.
    # Try largest-first greedy
    selected = []
    sel_set = set()
    for x in range(N, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        # Check if adding x keeps validity
        candidate_set = sel_set | {x}
        # Quick check: only need to check x against existing, and existing against x
        # Actually adding x could break existing elements' constraints too.
        # Full check is expensive. Let's do a targeted check.
        # Check: does x divide sum of any 3 existing?
        ok = True
        others = list(sel_set)
        n_o = len(others)
        for i in range(n_o):
            b = others[i]
            for j in range(i + 1, n_o):
                c = others[j]
                for k in range(j + 1, n_o):
                    d = others[k]
                    if (b + c + d) % x == 0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            # Also check: does any existing a divide sum of 3 from (sel_set ∪ {x}) \ {a}?
            # This is the expensive part. Only check if sel_set is small enough.
            if len(sel_set) <= 25:
                for a in sel_set:
                    others_a = [v for v in sel_set if v != a] + [x]
                    n_a = len(others_a)
                    for i in range(n_a):
                        b = others_a[i]
                        for j in range(i + 1, n_a):
                            c = others_a[j]
                            for k in range(j + 1, n_a):
                                d = others_a[k]
                                if (b + c + d) in forbidden_sums[a]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
        if ok:
            selected.append(x)
            sel_set.add(x)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = list(selected)
    
    # Local search: try to swap out one element for another to increase size or maintain.
    # Actually we want to maximize cardinality, so try to add more elements.
    # Try adding each remaining element, and if it causes conflicts, try removing one
    # conflicting element to make room.
    
    remaining = [x for x in range(1, N + 1) if x not in sel_set]
    
    for _round in range(3):
        if time.time() - start_time > DEADLINE:
            break
        for x in remaining:
            if time.time() - start_time > DEADLINE:
                break
            if x in sel_set:
                continue
            # Try adding x
            candidate_set = sel_set | {x}
            # Check validity of x against existing
            ok = True
            others = list(sel_set)
            n_o = len(others)
            for i in range(n_o):
                b = others[i]
                for j in range(i + 1, n_o):
                    c = others[j]
                    for k in range(j + 1, n_o):
                        d = others[k]
                        if (b + c + d) % x == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check existing elements against new triple including x
            for a in sel_set:
                others_a = [v for v in sel_set if v != a] + [x]
                n_a = len(others_a)
                for i in range(n_a):
                    b = others_a[i]
                    for j in range(i + 1, n_a):
                        c = others_a[j]
                        for k in range(j + 1, n_a):
                            d = others_a[k]
                            if (b + c + d) in forbidden_sums[a]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                sel_set.add(x)
                remaining.remove(x)
                if len(sel_set) > best_size:
                    best_size = len(sel_set)
                    best_set = list(sel_set)
    
    # Randomized greedy with different orderings
    rng = random.Random(42)
    for trial in range(50):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel = set()
        for x in order:
            if time.time() - start_time > DEADLINE:
                break
            # Quick check
            ok = True
            others = list(sel)
            n_o = len(others)
            for i in range(n_o):
                b = others[i]
                for j in range(i + 1, n_o):
                    c = others[j]
                    for k in range(j + 1, n_o):
                        d = others[k]
                        if (b + c + d) % x == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                for a in sel:
                    others_a = [v for v in sel if v != a] + [x]
                    n_a = len(others_a)
                    for i in range(n_a):
                        b = others_a[i]
                        for j in range(i + 1, n_a):
                            c = others_a[j]
                            for k in range(j + 1, n_a):
                                d = others_a[k]
                                if (b + c + d) in forbidden_sums[a]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
            if ok:
                sel.add(x)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    return best_set

# Ensure the function is at module level with correct name
# (already defined above)
