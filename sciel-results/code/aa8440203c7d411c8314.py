"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

N = 192
MAX_SIZE = 48

# Precomputation: for each element a, which triples of OTHER elements sum to a multiple of a?
# We'll build a conflict structure for backtracking.

def _precompute():
    """Precompute all necessary data for constraint-based search."""
    # For each element a in 1..192, find all pairs (b,c) with b<c, b!=a, c!=a
    # such that a divides (a' + b + c) where a' is the third element... 
    # Actually, the constraint is: for selected set S, for each a in S,
    # no triple (b,c,d) from S\{a} has (b+c+d) % a == 0.
    
    # Strategy: During backtracking, when we tentatively add x to current set C,
    # we need to check:
    # 1. For each a in C: does any triple from C\{a} plus x... no wait.
    #    Actually: for each a in C ∪ {x}, check if any 3 distinct OTHER elements in C ∪ {x}\{a}
    #    sum to a multiple of a.
    
    # Let's think differently. When building set S incrementally:
    # When we add x to current set C to form S = C ∪ {x}:
    # - For x: check if any triple from C sums to multiple of x
    # - For each a in C: check if any triple from (C\{a}) ∪ {x} sums to multiple of a
    #   This means: triples entirely in C\{a} (already checked if C was valid),
    #   OR triples with exactly one element from {x} and two from C\{a}
    #   OR triples with... wait, x is one element, so triples with x and 2 from C\{a}
    
    # So when adding x to valid set C:
    # Check A: Does x divide sum of any 3 elements in C? (need |C| >= 3)
    # Check B: For each a in C, does a divide (x + b + c) for any b,c in C\{a}? (need |C\{a}| >= 2)
    
    # Precompute for efficiency:
    # divisors_of_sum[a][s] = True if a divides s, for relevant sums
    # But sums range from 3 to 3*192=576, so we can precompute divisibility.
    
    # Better: precompute for each a, the set of values v such that a | v.
    # Then when checking if a | (x+b+c), check if (x+b+c) % a == 0.
    
    # For speed, precompute:
    # triple_sums[a] = set of all possible sums of 3 distinct elements from [1..192]\{a}
    # That's too large. Instead, let's just do the check on the fly with early termination.
    
    # Precompute: for each a, list of (b,c) pairs where b<c, b!=a, c!=a
    # and store (b+c) so we can quickly check if (b+c) % a gives a value that x could complete.
    
    # Actually, let's precompute:
    # For each a, and for each possible remainder r = (x % a), 
    # find if there exist b,c in C\{a} such that (b+c) % a == (a - r) % a
    # i.e., (x + b + c) % a == 0
    
    # This is getting complex. Let's use a simpler but effective approach:
    # Precompute all pairs (b,c) with b<c from 1..192, and for each pair, 
    # precompute (b+c) % a for each a.
    
    # Even simpler: just do the checks directly but with numpy vectorization where possible.
    
    # For the backtracking solver, let's precompute:
    # 1. For each x, which triples (i,j,k) from 1..192 have (i+j+k) % x == 0
    #    Store as a bitmask or list of frozensets
    
    # Given time constraints, let's use a hybrid approach:
    # - Precompute pair sums modulo each element
    # - Use bitmasks for fast intersection checks
    
    # Precompute pair_sums_mod[a] = dict mapping (b,c) -> (b+c) % a
    # Too much memory. Let's just precompute for each a, the set of (b+c) % a values
    # that appear, but we need to know WHICH pairs.
    
    # Alternative: Precompute for each a, a 2D boolean array compat[a][b][c] 
    # meaning a | (b+c+d) for some... no, we need triples.
    
    # Let's take a different structural approach:
    # Model as: we want to select a subset S of {1,...,192} such that
    # for all a in S, and all distinct b,c,d in S\{a}, (b+c+d) % a != 0.
    
    # This is equivalent to: S is an independent set in a 4-uniform hypergraph H
    # where hyperedge {a,b,c,d} exists if a | (b+c+d) or b | (a+c+d) or c | (a+b+d) or d | (a+b+c).
    
    # Finding max independent set in 4-uniform hypergraph is NP-hard, but with n=192 and
    # target size ~48, we can use branch-and-bound with good pruning.
    
    # Precomputation:
    # 1. For each quadruple (a,b,c,d) with a<b<c<d, check if any of the 4 divisibility conditions hold.
    #    If so, mark that these 4 cannot all be in S together.
    #    But that's C(192,4) ~ 57 million, too many.
    
    # Better: Precompute for each pair (a,b) whether they can coexist given some context.
    # Actually, the constraint is 4-ary, so pairwise compatibility isn't sufficient.
    
    # Let's use incremental checking with memoization during backtracking.
    
    # Precompute: for each x in 1..192, the set of all triples (i,j,k) with i<j<k, i!=x, j!=x, k!=x
    # such that (i+j+k) % x == 0. Store as a list of frozensets for fast lookup.
    
    # Memory: for each x, number of such triples could be large. Let's estimate:
    # For x=1, all triples work: C(191,3) ~ 1.1M - too many to store explicitly.
    
    # For x=1, the constraint is trivially violated if |S\{1}| >= 3, so we can handle x=1 specially.
    # For small x, many triples work. For large x, fewer triples work.
    
    # Let's precompute only for x >= some threshold, and handle small x with direct checking.
    
    # Actually, let's just implement efficient direct checking and rely on good search ordering.
    
    pass  # We'll do the precomputation inline in the solver

def _can_add_to_set(x, current_set, current_list):
    """Check if x can be added to current_list (which is a valid set) without violating constraints."""
    n = len(current_list)
    current_set_local = set(current_list)
    
    # Check 1: Does x divide sum of any 3 elements in current_list?
    if n >= 3:
        # Use itertools for clarity, but this is O(n^3)
        for i in range(n - 2):
            a = current_list[i]
            for j in range(i + 1, n - 1):
                b = current_list[j]
                for k in range(j + 1, n):
                    c = current_list[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # Check 2: For each a in current_list, does a divide (x + b + c) for any b,c in current_list\{a}?
    if n >= 2:
        for idx, a in enumerate(current_list):
            # Get all elements except a
            others = [current_list[j] for j in range(n) if j != idx]
            m = len(others)
            if m >= 2:
                for p in range(m - 1):
                    b = others[p]
                    for q in range(p + 1, m):
                        c = others[q]
                        if (x + b + c) % a == 0:
                            return False
    
    return True

def _backtrack_search(available, current, current_set, deadline, best_size):
    """
    Backtracking search with branch-and-bound.
    available: list of candidate elements not yet decided
    current: list of selected elements so far
    current_set: set of selected elements for O(1) lookup
    deadline: time limit
    best_size: current best size found
    Returns: (best_list, best_size)
    """
    start = time.time()
    if time.time() > deadline:
        return current, len(current)
    
    # Branch and bound: if current size + available count <= best_size, prune
    if len(current) + len(available) <= best_size:
        return current, len(current)
    
    if not available:
        return current, len(current)
    
    best_local = list(current)
    best_local_size = len(current)
    
    # Choose next element to branch on
    # Heuristic: try elements in descending order (larger elements are "safer" as they
    # are less likely to divide sums of others)
    x = available[0]
    remaining = available[1:]
    
    # Option 1: Include x
    if _can_add_to_set(x, current_set, current):
        new_current = current + [x]
        new_set = current_set | {x}
        result, size = _backtrack_search(remaining, new_current, new_set, deadline, best_local_size)
        if size > best_local_size:
            best_local = result
            best_local_size = size
    
    # Option 2: Exclude x
    result, size = _backtrack_search(remaining, current, current_set, deadline, best_local_size)
    if size > best_local_size:
        best_local = result
        best_local_size = size
    
    return best_local, best_local_size

def _greedy_with_conflict_check(order, deadline):
    """Greedy construction with fast conflict checking."""
    selected = []
    selected_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _can_add_to_set(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_improve(selected, deadline):
    """Try to improve a valid set by adding elements or doing swaps."""
    best = list(selected)
    best_set = set(best)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            if _can_add_to_set(x, best_set, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try swaps: remove one, try to add two or more
        if len(best) > 0:
            for rem_idx in range(len(best)):
                if time.time() > deadline:
                    break
                rem_elem = best[rem_idx]
                temp = [e for i, e in enumerate(best) if i != rem_idx]
                temp_set = set(temp)
                
                added_count = 0
                for x in range(N, 0, -1):
                    if time.time() > deadline:
                        break
                    if x in temp_set or x == rem_elem:
                        continue
                    if _can_add_to_set(x, temp_set, temp):
                        temp.append(x)
                        temp_set.add(x)
                        added_count += 1
                
                if len(temp) > len(best):
                    best = temp
                    best_set = temp_set
                    improved = True
                    break
    
    return best

def construct():
    start_time = time.time()
    # Safety margin: return before 90s wall-clock limit
    deadline = start_time + 80.0
    
    # Phase 1: Multiple greedy starts
    best = []
    
    # Start 1: Descending order
    order1 = list(range(N, 0, -1))
    cand1 = _greedy_with_conflict_check(order1, deadline)
    if len(cand1) > len(best):
        best = cand1
    
    # Start 2: Primes first
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(N, 0, -1) if is_prime(x)]
    non_primes = [x for x in range(N, 0, -1) if not is_prime(x)]
    order2 = primes + non_primes
    cand2 = _greedy_with_conflict_check(order2, deadline)
    if len(cand2) > len(best):
        best = cand2
    
    # Start 3: Randomized
    random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 5.0:
            break
        order = list(range(N, 0, -1))
        random.shuffle(order)
        cand = _greedy_with_conflict_check(order, deadline)
        if len(cand) > len(best):
            best = cand
    
    # Phase 2: Local improvement
    if time.time() < deadline - 5.0:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: Backtracking from current best to try to exceed it
    if time.time() < deadline - 10.0 and len(best) < MAX_SIZE:
        # Use backtracking to try to find a larger set
        # Start from the current best and try to add more
        current_best_set = set(best)
        available = [x for x in range(N, 0, -1) if x not in current_best_set]
        
        # Try backtracking with a limited budget
        bt_deadline = min(deadline - 2.0, start_time + 75.0)
        bt_result, bt_size = _backtrack_search(available, list(best), current_best_set, bt_deadline, len(best))
        if bt_size > len(best):
            best = bt_result
    
    # Final validation and cleanup
    if len(best) == 0:
        best = [N]
    
    # Ensure distinct positive integers <= N
    best = list(set(best))
    best = [x for x in best if 1 <= x <= N]
    best.sort(reverse=True)
    
    # Enforce max size
    if len(best) > MAX_SIZE:
        best = best[:MAX_SIZE]
    
    return best
# EVOLVE-BLOCK-END
