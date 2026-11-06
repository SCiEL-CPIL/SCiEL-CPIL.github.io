"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Exact integer arithmetic via LCM(1..22).
  - Fast randomized greedy construction with local search (add/swap).
  - Global deadline guard.
"""
import time
import random
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 50.0  # Wall-clock guard (evaluator kills at 60s; keep 10s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _greedy_construct(order):
    """Greedy construction: pick elements in 'order' if they fit."""
    current_set = []
    sums = {0}
    for idx in order:
        v = _REC[idx]
        # Check if adding v causes collision
        # Collision if any s in sums equals s' + v for some s' in sums
        # i.e., v - s' == s for some s, s' in sums
        # Equivalent to: (sums + v) intersects sums? No, that's not right.
        # The new sums are sums U (sums+v).
        # We need sums and (sums+v) to be disjoint.
        # This is equivalent to: for all s in sums, s+v not in sums.
        # Or: (sums + v) & sums == empty set.
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


def _local_search(current_set, sums, deadline):
    """Try to improve current_set by adding elements or swapping."""
    # First, try to just add any unused element
    unused = set(range(1, _N + 1)) - set(current_set)
    for cand in list(unused):
        if time.time() > deadline:
            break
        new_set, new_sums = _try_add(current_set, sums, cand)
        if new_set is not None:
            return new_set, new_sums, True # Improved

    # Try swaps: remove one existing, add one unused
    # This is O(N^2) per iteration, so be careful
    # We only do a few iterations
    for _ in range(10):
        if time.time() > deadline:
            break
        # Shuffle unused and current to randomize
        unused_list = list(unused)
        current_list = list(current_set)
        random.shuffle(unused_list)
        random.shuffle(current_list)
        
        improved = False
        for rem in current_list:
            if time.time() > deadline:
                break
            # Calculate sums without rem
            # This is expensive if done naively. 
            # Better: precompute? Or just check if (current_set - {rem}) + {cand} is valid.
            # Since N is small (<=22), and set size is small (<=10ish), 
            # we can reconstruct sums for the candidate swap if needed.
            
            # Optimization: 
            # If we remove rem, the new sums are not easily derived from old sums.
            # However, we can just check the validity of the new set directly using _is_dissociated?
            # That is 2^k operations. k is small.
            
            for add in unused_list:
                if add == rem: continue
                new_set = [x for x in current_set if x != rem] + [add]
                # Check if valid
                if _is_dissociated(new_set):
                    # Recalculate sums for the next iteration
                    new_sums = {0}
                    for e in new_set:
                        v = _REC[e]
                        new_sums |= {s + v for s in new_sums}
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

    # Randomized restarts
    # We run many greedy restarts with different random orders
    # And then apply local search to the best found so far
    
    while time.time() < deadline - 2.0:
        # Generate a random order
        order = list(range(1, _N + 1))
        random.shuffle(order)
        
        # Greedy construct
        curr_set, curr_sums = _greedy_construct(order)
        
        # If this greedy result is better, update best
        if len(curr_set) > best_size:
            best_set = curr_set
            best_sums = curr_sums
            best_size = len(curr_set)
        elif len(curr_set) == best_size:
            # Tie-break: maybe keep the one with larger sum? Or just random?
            # For now, just keep the first one found or randomize?
            # Let's stick to the first one unless we find a strictly larger one.
            pass

        # Try to improve the current best (or the current candidate if it's close)
        # We apply local search to the global best
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
