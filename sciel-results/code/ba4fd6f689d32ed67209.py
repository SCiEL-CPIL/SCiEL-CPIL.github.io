"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

# Precompute conflict structure:
# For each element a, and each pair (b,c) with b,c != a, b != c,
# the sum s = b + c + d for some d. Actually, let's think differently.
#
# A conflict exists when a | (b + c + d) where a,b,c,d are distinct elements.
# For a fixed set S, element a in S is "bad" if there exist b,c,d in S\{a} with a | (b+c+d).
#
# Key insight: For a given a, and given we've already selected some set T,
# adding x creates a conflict if:
#   1. x | (sum of 3 from T) for some triplet in T
#   2. For some a in T, a | (x + b + c) for some b,c in T\{a}
#
# Precompute for each a: for each pair (b,c), the "needed" values of x such that a | (x+b+c)
# i.e., x ≡ -(b+c) (mod a). So x mod a must equal (a - (b+c) % a) % a.
#
# Better approach: precompute for each a, a set of forbidden residues or specific values.
# Since x is in [1,192], for each a and pair (b,c) in [1,192]\{a}, compute what x values
# would make a | (x+b+c). That's x ≡ target (mod a).
#
# Let's precompute: conflict[a] = set of x values in [1,192] such that there exist b,c in [1,192]\{a,x}
# with a | (x+b+c). But this depends on which b,c are available...
#
# Actually, for the greedy check: given selected set S and candidate x:
# - Check 1: Does x divide sum of any 3 elements in S? (x not in S, so all 3 are "other")
# - Check 2: For each a in S, does a divide (x + b + c) for some b,c in S\{a}?
#
# Let's precompute more efficiently:
# For each a in [1,192], precompute a dictionary: for each pair-sum p = b+c where b,c in [1,192]\{a},
# the set of x values in [1,192]\{a} such that a | (x+p). That's x ≡ -p (mod a).
#
# Actually, let me just precompute the full conflict hypergraph edges.
# An edge is (a, b, c, d) where a | (b+c+d), all distinct.
# There are C(192,4) ≈ 24 million possible 4-tuples, too many to enumerate all.
#
# Better: For each a, for each pair (b,c) with b,c != a, b != c:
#   target = (-b - c) % a
#   x values: all x in [1,192]\{a,b,c} such that x % a == target
#
# Let's precompute: for each a, a list of (pair_key, x_values) where pair_key = (b,c) sorted,
# and x_values are the x's that would conflict.
#
# Actually, simplest fast approach:
# Precompute for each a: bad_pairs[a] = set of (b, c) such that for SOME x in [1,192]\{a,b,c}, a | (x+b+c).
# No wait, that's not quite right either.
#
# Let me think about this differently. For the incremental check:
# Given S (selected) and x (candidate not in S):
#
# Violation type 1: x | (s1+s2+s3) for some s1,s2,s3 in S
#   Precompute: for each x, the set of triplets from [1,192] whose sum is divisible by x.
#   Then check if any triplet is a subset of S.
#   This is hard to precompute efficiently for all x.
#
# Violation type 2: a | (x + b + c) for some a in S, b,c in S\{a}
#   For each a in S, we need: is there a pair (b,c) in S\{a} such that (x+b+c) % a == 0?
#   i.e., (b+c) % a == (-x) % a
#
# Precompute for each a: pair_sum_mods[a] = dict mapping residue r to set of pairs (b,c)
# where (b+c) % a == r, and b,c in [1,192]\{a}.
#
# Then for check 2: for each a in S, look up pair_sum_mods[a][(-x)%a] and check if any pair is subset of S\{a}.
#
# This is still potentially expensive. Let me just make the existing checks faster with numpy or
# a smarter data structure.
#
# Actually, let's use a different approach: precompute for each a, for each possible pair-sum s (from 3 to 384),
# whether a divides (x + s) for any x. No, that doesn't help.
#
# Let me just optimize the existing approach with better data structures.
#
# For check 2 (the expensive one): Given S, x, and a in S:
#   We need pairs (b,c) in S\{a} such that (b+c) % a == (-x) % a
#   Precompute: for each a, a set of all pair-sums from [1,192]\{a} grouped by mod a.
#   pair_sums_by_mod[a][r] = set of (b,c) pairs where b,c in [1,192]\{a}, b<c, and (b+c)%a == r
#
# Then checking if any pair from S\{a} has the right mod: iterate over S\{a} pairs and check.
#
# Let me just do a direct optimization with caching.

_N = 192
_MAX = 48

def _precompute():
    """Precompute data structures for fast conflict checking."""
    # For each a in 1..192, precompute: for each residue r mod a,
    # the set of pairs (b,c) from [1,192]\{a} with b<c and (b+c) % a == r
    # This is used for check 2.
    
    # Also for check 1: for each x, precompute the set of triplets from [1,192]
    # whose sum is divisible by x. But that's C(192,3) ≈ 1.17M triplets per x, too many.
    # Instead, for check 1, we'll just iterate over triplets of S (which is at most 48 elements,
    # so C(48,3) = 17296, manageable).
    
    # Precompute pair sums by mod for each a
    pair_sums_by_mod = {}
    for a in range(1, _N + 1):
        pair_sums_by_mod[a] = {}
        for b in range(1, _N + 1):
            if b == a:
                continue
            for c in range(b + 1, _N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                if r not in pair_sums_by_mod[a]:
                    pair_sums_by_mod[a][r] = set()
                pair_sums_by_mod[a][r].add((b, c))
    return pair_sums_by_mod

_pair_sums_by_mod = None

def _get_pair_sums_by_mod():
    global _pair_sums_by_mod
    if _pair_sums_by_mod is None:
        _pair_sums_by_mod = _precompute()
    return _pair_sums_by_mod

def _can_add(x, selected, selected_set):
    """Check if x can be added to selected without creating any violation.
    selected: list of selected elements
    selected_set: set of selected elements
    Returns True if x can be added."""
    psbm = _get_pair_sums_by_mod()
    
    # Check 1: x divides sum of any 3 distinct elements in selected
    n = len(selected)
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, a divides (x + b + c) for some b,c in selected\{a}
    for a in selected:
        if a == x:
            continue
        target_res = (-x) % a
        # Need pairs (b,c) in selected\{a} with (b+c) % a == target_res
        # Iterate over pairs in selected\{a}
        others = [v for v in selected if v != a]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                if (others[i] + others[j]) % a == target_res:
                    return False
    
    return True

def _is_valid(selected):
    """Check if the set is valid."""
    selected_set = set(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_degree():
    """Greedy using degree-based selection: pick element with fewest conflicts with remaining candidates."""
    selected = []
    selected_set = set()
    remaining = set(range(1, _N + 1))
    
    while remaining and len(selected) < _MAX:
        best_x = None
        best_conflicts = float('inf')
        
        for x in sorted(remaining, reverse=True):
            # Count how many elements in remaining would be blocked by adding x
            # (i.e., how many y in remaining would have a conflict with x given current selected)
            # Simpler: just check if x can be added, and prefer larger x
            if _can_add(x, selected, selected_set):
                best_x = x
                best_conflicts = 0
                break
        
        if best_x is None:
            # Try all remaining to find one with 0 conflicts
            for x in sorted(remaining, reverse=True):
                if _can_add(x, selected, selected_set):
                    best_x = x
                    break
        
        if best_x is None:
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        remaining.remove(best_x)
    
    return selected

def _greedy_random(order_seed):
    """Greedy with a specific random ordering."""
    rng = random.Random(order_seed)
    selected = []
    selected_set = set()
    candidates = list(range(1, _N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected, selected_set):
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_search(selected, time_limit=5.0):
    """Try to improve the set by adding elements or swapping."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_set = set(best)
    best_size = len(best)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in best_set]
        if not candidates:
            break
        
        random.shuffle(candidates)
        
        # Try adding
        for x in candidates:
            if _can_add(x, best, best_set):
                best.append(x)
                best_set.add(x)
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add one
            if len(best) >= 1:
                swap_found = False
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    removed = best[i]
                    temp_set = best_set - {removed}
                    temp_list = best[:i] + best[i+1:]
                    
                    for x in candidates:
                        if x in temp_set:
                            continue
                        if _can_add(x, temp_list, temp_set):
                            best = temp_list + [x]
                            best_set = temp_set | {x}
                            swap_found = True
                            break
                    
                    if swap_found:
                        break
        
        if not improved and not swap_found:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin
    
    # Precompute
    _get_pair_sums_by_mod()
    
    # Start with greedy (largest first)
    result = _greedy_degree()
    
    # Try multiple random greedy strategies
    for attempt in range(50):
        if time.time() > deadline - 15:
            break
        
        selected = _greedy_random(attempt)
        if len(selected) > len(result):
            result = selected
    
    # Local search to improve
    if time.time() < deadline - 5:
        remaining_time = min(10.0, deadline - time.time() - 2.0)
        if remaining_time > 1.0:
            result = _local_search(result, time_limit=remaining_time)
    
    return result
# EVOLVE-BLOCK-END
