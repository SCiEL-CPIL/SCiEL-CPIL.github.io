import time
import random
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated."""
        if not elements:
            return True
        # Use integer arithmetic for speed if possible, but Fraction is safer for correctness
        # and fast enough for k <= 20.
        n = len(elements)
        if n > 20: 
            # Fallback or early exit if too large, though 2^20 is 1M which is okay
            pass
        
        sums = [Fraction(0, 1)]
        seen = {Fraction(0, 1)}
        for e in elements:
            r = Fraction(1, e)
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True

    all_elements = list(range(1, 23))
    best_set = []
    best_len = 0
    
    # Helper to try to extend a set greedily with a given order of remaining elements
    def extend_greedily(current_set, order):
        current = current_set[:]
        for e in order:
            if e in current:
                continue
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        return current

    # 1. Try a few deterministic orders first
    deterministic_orders = [
        list(range(22, 0, -1)),           # Descending
        list(range(1, 23)),               # Ascending
        list(range(22, 0, -2)) + list(range(1, 23, 2)), # Evens desc, Odds asc
        list(range(1, 23, 2)) + list(range(22, 0, -2)), # Odds asc, Evens desc
    ]
    
    for order in deterministic_orders:
        if time.time() > deadline:
            break
        s = extend_greedily([], order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s[:]

    # 2. Randomized Restart
    # Try many random permutations
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(all_elements)
        # We can also try to start with a random subset of the best_set found so far
        # But pure random greedy from scratch is a good baseline.
        s = extend_greedily([], all_elements)
        
        # Local search improvement on the greedy result
        # Try to add any remaining element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            remaining = [e for e in all_elements if e not in s]
            for e in remaining:
                if time.time() > deadline:
                    break
                candidate = s + [e]
                if is_dissociated(candidate):
                    s = candidate
                    improved = True
                    break # Restart loop with new remaining
            
        # Try swaps: remove one, add another
        if time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(s)):
                    if time.time() > deadline:
                        break
                    for e in all_elements:
                        if e in s:
                            continue
                        candidate = [x for j, x in enumerate(s) if j != i] + [e]
                        if is_dissociated(candidate):
                            # Try to fill again
                            temp_s = candidate
                            fill_improved = True
                            while fill_improved and time.time() < deadline:
                                fill_improved = False
                                remaining_after = [x for x in all_elements if x not in temp_s]
                                for x in remaining_after:
                                    if time.time() > deadline:
                                        break
                                    test = temp_s + [x]
                                    if is_dissociated(test):
                                        temp_s = test
                                        fill_improved = True
                                        break
                            if len(temp_s) > len(s):
                                s = temp_s
                                improved = True
                                break
                    if improved:
                        break
            if len(s) > best_len:
                best_len = len(s)
                best_set = s[:]
        
        # If we found a set of size 17 or more, we can be more aggressive or stop
        # But let's keep searching for 18+ if possible.
        # Optimization: If best_len is high, we might want to bias the random order
        # to include elements from the best_set early?
        # For now, pure random is a good diversification.

    # Final verification
    if not is_dissociated(best_set):
        # Fallback to a known valid set
        # Size 9 is safe. Size 16 is the current best.
        # Let's try to construct a safe fallback if something went wrong.
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
