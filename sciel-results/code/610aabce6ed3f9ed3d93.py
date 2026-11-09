"""Max independent set in reciprocal-Schur hypergraph via structure-exploiting greedy + local search."""
import time
import random
import math
from collections import defaultdict

_DEADLINE = 85.0
N = 180


def _precompute_triples():
    """
    Find all triples (a, b, c) of distinct integers in [1, 180] such that
    1/a = 1/b + 1/c, with a < b < c (WLOG).
    
    Key identity: (b-a)(c-a) = a^2
    So for each a, factor pairs (d, e) of a^2 give b = a+d, c = a+e.
    """
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue
            b = a + d
            c = a + e
            if b > N or c > N:
                continue
            if a == b or a == c or b == c:
                continue
            # Ensure a < b < c
            if a < b and b < c:
                triples.append((a, b, c))
            elif a < c and c < b:
                triples.append((a, c, b))
    return triples


def _build_pair_conflicts(triples):
    """
    For each pair (u, v), store the list of triples containing both.
    Two elements conflict if they appear together in at least one triple.
    """
    pair_conflicts = defaultdict(list)  # (min(u,v), max(u,v)) -> [triple_indices]
    for idx, (a, b, c) in enumerate(triples):
        pair_conflicts[(a, b)].append(idx)
        pair_conflicts[(a, c)].append(idx)
        pair_conflicts[(b, c)].append(idx)
    return pair_conflicts


def _can_add(x, selected_bits, pair_conflicts, triples):
    """
    Check if x can be added to the current set without creating a violation.
    A violation occurs if there exist y, z in selected such that {x, y, z} is a triple.
    """
    for (u, v), tlist in pair_conflicts.items():
        if u == x or v == x:
            other = v if u == x else u
            if selected_bits[other]:
                for ti in tlist:
                    a, b, c = triples[ti]
                    if a == x and selected_bits[b] and selected_bits[c]:
                        return False
                    if b == x and selected_bits[a] and selected_bits[c]:
                        return False
                    if c == x and selected_bits[a] and selected_bits[b]:
                        return False
    return True


def _greedy_structural(ordering, triples, pair_conflicts):
    """
    Greedy: process elements in given order, add if no conflict.
    """
    selected = set()
    selected_bits = [False] * (N + 1)
    
    for x in ordering:
        if _can_add(x, selected_bits, pair_conflicts, triples):
            selected.add(x)
            selected_bits[x] = True
    
    return selected


def _local_search_improve(initial_set, triples, pair_conflicts, time_limit):
    """
    Local search: try single removals and double swaps to increase set size.
    """
    selected = set(initial_set)
    selected_bits = [False] * (N + 1)
    for x in selected:
        selected_bits[x] = True
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try adding any element that fits
        for x in range(1, N + 1):
            if not selected_bits[x]:
                if _can_add(x, selected_bits, pair_conflicts, triples):
                    selected.add(x)
                    selected_bits[x] = True
                    improved = True
        
        # Try single removals: remove x, see if we can add 2+ new elements
        for x in list(selected):
            if (time.time() - start_time) > time_limit:
                break
            
            # Temporarily remove x
            selected_bits[x] = False
            selected.discard(x)
            
            # Try to add as many as possible
            added = []
            for y in range(1, N + 1):
                if not selected_bits[y]:
                    if _can_add(y, selected_bits, pair_conflicts, triples):
                        selected.add(y)
                        selected_bits[y] = True
                        added.append(y)
            
            if len(added) <= 1:
                # Revert: remove what we added, re-add x
                for y in added:
                    selected.discard(y)
                    selected_bits[y] = False
                selected.add(x)
                selected_bits[x] = True
            else:
                # Keep the improvement
                improved = True
    
    return selected


def _multi_start_greedy(triples, pair_conflicts, n_starts, time_limit):
    """
    Run multiple greedy passes with different orderings.
    """
    best = set()
    start_time = time.time()
    
    for i in range(n_starts):
        if (time.time() - start_time) > time_limit:
            break
        
        if i == 0:
            ordering = list(range(1, N + 1))
        elif i == 1:
            ordering = list(range(N, 0, -1))
        elif i == 2:
            # Evens first, then odds
            ordering = [x for x in range(2, N + 1, 2)] + [x for x in range(1, N + 1, 2)]
        elif i == 3:
            # Odds first, then evens
            ordering = [x for x in range(1, N + 1, 2)] + [x for x in range(2, N + 1, 2)]
        elif i == 4:
            # Random
            ordering = list(range(1, N + 1))
            random.shuffle(ordering)
        elif i == 5:
            # Large to small
            ordering = sorted(range(1, N + 1), key=lambda x: -x)
        elif i == 6:
            # Small to large, but skip multiples of 6
            ordering = [x for x in range(1, N + 1) if x % 6 != 0] + [x for x in range(1, N + 1) if x % 6 == 0]
        elif i == 7:
            # Random
            ordering = list(range(1, N + 1))
            random.shuffle(ordering)
        else:
            # Weighted random: prefer smaller numbers
            ordering = list(range(1, N + 1))
            random.shuffle(ordering)
            ordering.sort(key=lambda x: random.random() + 0.1 * x / N)
        
        result = _greedy_structural(ordering, triples, pair_conflicts)
        
        if len(result) > len(best):
            best = result
    
    return best


def _validate_and_fix(s, triples):
    """Ensure the set is valid; remove offending elements if needed."""
    bits = [False] * (N + 1)
    for x in s:
        bits[x] = True
    
    # Check for violations
    violated = set()
    for (a, b, c) in triples:
        if bits[a] and bits[b] and bits[c]:
            violated.add(a)
            violated.add(b)
            violated.add(c)
    
    if not violated:
        return s
    
    # Remove elements from violated set, re-check
    result = s - violated
    bits = [False] * (N + 1)
    for x in result:
        bits[x] = True
    
    # May need more removals
    changed = True
    while changed:
        changed = False
        for (a, b, c) in triples:
            if bits[a] and bits[b] and bits[c]:
                # Remove the largest
                m = max(a, b, c)
                result.discard(m)
                bits[m] = False
                changed = True
    
    return result


def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    pair_conflicts = _build_pair_conflicts(triples)
    
    # Multi-start greedy
    best = _multi_start_greedy(triples, pair_conflicts, n_starts=8, time_limit=30)
    
    # Local search improvement
    if time.time() - start_time < _DEADLINE - 10:
        remaining_time = min(_DEADLINE - (time.time() - start_time) - 5, 30)
        improved = _local_search_improve(best, triples, pair_conflicts, remaining_time)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    best = _validate_and_fix(best, triples)
    
    if not best:
        # Fallback: just take {1, 2} or something safe
        best = {1, 2}
        best = _validate_and_fix(best, triples)
        if not best:
            best = {1}
    
    return sorted(best)
