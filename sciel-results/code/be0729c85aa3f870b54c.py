"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Exact integer arithmetic via LCM(1..22).
  - Fast greedy construction with randomized restarts.
  - Simulated annealing with add/remove/swap moves.
  - Global deadline guard.
"""
import time
import random
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 80.0  # Wall-clock guard (evaluator kills at 90s; keep 10s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    if not elements:
        return True
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _compute_sums(elements):
    """Compute all subset sums for a given element list."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        sums |= {s + v for s in sums}
    return sums


def _greedy_construct(order):
    """Greedy construction: pick elements in 'order' if they fit."""
    current_set = []
    sums = {0}
    for idx in order:
        v = _REC[idx]
        # Check if adding v causes collision
        # New sums would be sums U (sums + v)
        # Collision if (sums + v) intersects sums
        if not (any((s + v) in sums for s in sums)):
            current_set.append(idx)
            sums |= {s + v for s in sums}
    return current_set, sums


def _try_add(current_set, sums, candidate):
    """Try to add candidate to current_set. Returns (new_set, new_sums) or (None, None)."""
    v = _REC[candidate]
    if any((s + v) in sums for s in sums):
        return None, None
    new_sums = sums | {s + v for s in sums}
    new_set = current_set + [candidate]
    return new_set, new_sums


def _remove_element(current_set, sums, element):
    """Remove element from current_set. Returns (new_set, new_sums) or (None, None)."""
    if element not in current_set:
        return None, None
    new_set = [x for x in current_set if x != element]
    if not new_set:
        return new_set, {0}
    new_sums = _compute_sums(new_set)
    return new_set, new_sums


def _simulated_annealing(initial_set, deadline, max_iter=5000):
    """Simulated annealing to maximize dissociated set size."""
    current_set = list(initial_set)
    current_sums = _compute_sums(current_set)
    best_set = list(current_set)
    best_sums = _compute_sums(best_set)
    best_size = len(best_set)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    unused = set(range(1, _N + 1)) - set(current_set)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and unused:
            # Add a random unused element
            cand = random.choice(list(unused))
            new_set, new_sums = _try_add(current_set, current_sums, cand)
            if new_set is not None:
                delta = 1  # Size increased
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_set = new_set
                    current_sums = new_sums
                    unused.remove(cand)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_sums = _compute_sums(best_set)
                        best_size = len(best_set)
        
        elif move_type < 0.7 and current_set:
            # Remove a random element
            rem = random.choice(current_set)
            new_set, new_sums = _remove_element(current_set, current_sums, rem)
            if new_set is not None:
                delta = -1  # Size decreased
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_set = new_set
                    current_sums = new_sums
                    unused.add(rem)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_sums = _compute_sums(best_set)
                        best_size = len(best_set)
        
        elif move_type < 1.0 and current_set and unused:
            # Swap: remove one, add one
            rem = random.choice(current_set)
            add = random.choice(list(unused))
            if rem == add:
                continue
            
            # Try the swap
            temp_set = [x for x in current_set if x != rem] + [add]
            if _is_dissociated(temp_set):
                temp_sums = _compute_sums(temp_set)
                delta = 0  # Size unchanged, but maybe enables future adds
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_set = temp_set
                    current_sums = temp_sums
                    unused.remove(add)
                    unused.add(rem)
                    if len(current_set) > best_size:
                        best_set = list(current_set)
                        best_sums = _compute_sums(best_set)
                        best_size = len(best_set)
        
        # Cool down
        T = max(T * cooling_rate, T_end)
    
    return best_set, best_sums


def _local_search(current_set, sums, deadline):
    """Try to improve current_set by adding elements or swapping."""
    # First, try to just add any unused element
    unused = set(range(1, _N + 1)) - set(current_set)
    for cand in list(unused):
        if time.time() > deadline:
            break
        new_set, new_sums = _try_add(current_set, sums, cand)
        if new_set is not None:
            return new_set, new_sums, True  # Improved

    # Try swaps: remove one existing, add one unused
    for _ in range(20):
        if time.time() > deadline:
            break
        unused_list = list(unused)
        current_list = list(current_set)
        random.shuffle(unused_list)
        random.shuffle(current_list)
        
        improved = False
        for rem in current_list:
            if time.time() > deadline:
                break
            for add in unused_list:
                if add == rem:
                    continue
                new_set = [x for x in current_set if x != rem] + [add]
                if _is_dissociated(new_set):
                    new_sums = _compute_sums(new_set)
                    return new_set, new_sums, True
        if not improved:
            break
            
    return current_set, sums, False


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    deadline = _T0 + _BUDGET
    
    # Initial baseline: Greedy descending
    best_set, best_sums = _greedy_construct(list(range(_N, 0, -1)))
    best_size = len(best_set)
    
    # Ensure we have a valid nonempty set
    if not best_set:
        best_set = [1]
        best_sums = {0, _REC[1]}
        best_size = 1

    # Randomized restarts with simulated annealing
    while time.time() < deadline - 5.0:
        # Generate a random order
        order = list(range(1, _N + 1))
        random.shuffle(order)
        
        # Greedy construct
        curr_set, curr_sums = _greedy_construct(order)
        
        # If this greedy result is better, update best
        if len(curr_set) > best_size:
            best_set = list(curr_set)
            best_sums = _compute_sums(best_set)
            best_size = len(best_set)
        
        # Apply simulated annealing starting from current best
        if time.time() < deadline - 10.0:
            sa_set, sa_sums = _simulated_annealing(best_set, deadline, max_iter=2000)
            if len(sa_set) > best_size:
                best_set = sa_set
                best_sums = sa_sums
                best_size = len(best_set)
        
        # Try local search on the best
        if time.time() < deadline - 5.0:
            new_set, new_sums, improved = _local_search(best_set, best_sums, deadline)
            if improved:
                best_set = new_set
                best_sums = new_sums
                best_size = len(best_set)

    # Final verification
    if not _is_dissociated(best_set):
        # Fallback to a known valid small set
        return [1]
        
    return best_set
