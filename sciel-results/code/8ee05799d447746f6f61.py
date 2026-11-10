"""Large integer sets avoiding equal-gcd triples.

Strategy:
1. Precompute all bad triples (where all three pairwise gcds are equal).
2. Start from a large structural seed (odd numbers + 1) and greedily add
   elements that don't create bad triples.
3. Use a maximum-removal heuristic to remove the minimum number of elements
   to eliminate all bad triples.
4. Use local search (swap/add) to try to grow the set further.
"""

# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # safety margin under the 90s wall-clock kill


def _bad_triple(a, b, c):
    """Return True if the triple (a,b,c) has all three pairwise gcds equal."""
    g_ab = math.gcd(a, b)
    g_ac = math.gcd(a, c)
    g_bc = math.gcd(b, c)
    return g_ab == g_ac == g_bc


def _is_valid(S):
    """Check if set S has no bad triples."""
    S_list = list(S)
    for a, b, c in combinations(S_list, 3):
        if _bad_triple(a, b, c):
            return False
    return True


def construct():
    """Construct a large subset of [1,128] with no three distinct elements
    whose three pairwise gcds are all equal."""
    random.seed(42)
    full = list(range(1, 129))
    
    def within_deadline():
        return (time.time() - _START_TIME) < _DEADLINE - 2.0

    # Precompute all bad triples
    bad_triples = []
    for a, b, c in combinations(full, 3):
        if _bad_triple(a, b, c):
            bad_triples.append((a, b, c))
    
    # For each element, track which bad triples it's in
    element_triples = {x: [] for x in full}
    for idx, (a, b, c) in enumerate(bad_triples):
        element_triples[a].append(idx)
        element_triples[b].append(idx)
        element_triples[c].append(idx)

    best_set = None
    best_size = 0

    # ---- Strategy 1: Start with odd numbers, greedily add evens ----
    def strategy_odd_start():
        S = set(range(1, 129, 2))  # all odd numbers
        # Verify no bad triples among odds
        S_list = list(S)
        valid = True
        for a, b, c in combinations(S_list, 3):
            if _bad_triple(a, b, c):
                valid = False
                break
        if not valid:
            # Remove problematic elements
            S = set(range(1, 129, 2))
            changed = True
            while changed and within_deadline():
                changed = False
                for a, b, c in combinations(list(S), 3):
                    if _bad_triple(a, b, c):
                        # Remove the one with most conflicts
                        conf = {}
                        for x in (a, b, c):
                            conf[x] = len([t for t in bad_triples if x in t and all(y in S for y in t)])
                        v = max((a, b, c), key=lambda x: conf.get(x, 0))
                        S.discard(v)
                        changed = True
                        break
        
        # Try to add even numbers
        evens = [x for x in range(2, 129, 2) if x not in S]
        random.shuffle(evens)
        for x in evens:
            if not within_deadline():
                break
            # Check if adding x creates any bad triple
            ok = True
            for a, b in combinations(S, 2):
                if _bad_triple(a, b, x):
                    ok = False
                    break
            if ok:
                S.add(x)
        return list(S)

    # ---- Strategy 2: Greedy max-conflict removal from full set ----
    def strategy_greedy_removal():
        S = set(full)
        # Count conflicts per element
        conflict_count = {x: len(element_triples[x]) for x in full}
        
        # Iteratively remove the element with highest conflict count
        while within_deadline():
            if not S:
                break
            # Find max conflict element in S
            max_conf = max(conflict_count[x] for x in S)
            if max_conf == 0:
                break
            candidates = [x for x in S if conflict_count[x] == max_conf]
            v = random.choice(candidates)
            S.discard(v)
            # Update conflict counts: for each bad triple containing v, decrement others
            for idx in element_triples[v]:
                a, b, c = bad_triples[idx]
                for u in (a, b, c):
                    if u != v and u in S:
                        # Only decrement if this triple is still "active" (all elements in S)
                        if a in S and b in S and c in S:
                            pass  # We'll recompute periodically
            # Periodic recompute
            if len(S) % 10 == 0:
                conflict_count = {}
                for x in S:
                    count = 0
                    for idx in element_triples[x]:
                        a, b, c = bad_triples[idx]
                        if a in S and b in S and c in S:
                            count += 1
                    conflict_count[x] = count
        return list(S)

    # ---- Strategy 3: Randomized greedy add from empty ----
    def strategy_random_greedy_add():
        S = []
        remaining = list(full)
        random.shuffle(remaining)
        for x in remaining:
            if not within_deadline():
                break
            if all(not _bad_triple(a, b, x) for a, b in combinations(S, 2)):
                S.append(x)
        return S

    # ---- Strategy 4: Local search - try swaps to grow set ----
    def strategy_local_search(start_set):
        S = set(start_set)
        improved = True
        while improved and within_deadline():
            improved = False
            # Try to add any element not in S
            for x in full:
                if x in S:
                    continue
                ok = True
                for a, b in combinations(S, 2):
                    if _bad_triple(a, b, x):
                        ok = False
                        break
                if ok:
                    S.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one, add one
            for remove_x in list(S):
                if not within_deadline():
                    break
                S.discard(remove_x)
                for add_x in full:
                    if add_x in S or add_x == remove_x:
                        continue
                    ok = True
                    for a, b in combinations(S, 2):
                        if _bad_triple(a, b, add_x):
                            ok = False
                            break
                    if ok:
                        S.add(add_x)
                        improved = True
                        break
                if not improved:
                    S.add(remove_x)  # restore
        return list(S)

    # Run strategies
    # Strategy 1: Odd start
    S1 = strategy_odd_start()
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Strategy 2: Greedy removal
    if within_deadline():
        S2 = strategy_greedy_removal()
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2

    # Strategy 3: Random greedy add (multiple runs)
    if within_deadline():
        for _ in range(10):
            S3 = strategy_random_greedy_add()
            if len(S3) > best_size:
                best_size = len(S3)
                best_set = S3

    # Strategy 4: Local search on best found so far
    if within_deadline() and best_set is not None:
        S4 = strategy_local_search(best_set)
        if len(S4) > best_size:
            best_size = len(S4)
            best_set = S4

    # Final: ensure best_set is nonempty and valid
    if best_set is None or len(best_set) == 0:
        return [1]
    
    # Verify the best set is valid
    if not _is_valid(set(best_set)):
        # Fallback: use odd numbers
        S_odd = list(range(1, 129, 2))
        if _is_valid(set(S_odd)):
            return S_odd
        return [1]
    
    return best_set

# EVOLVE-BLOCK-END
