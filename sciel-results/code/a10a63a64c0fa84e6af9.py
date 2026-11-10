"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s limit

    def check_cardinality_separated(elements):
        """Check if the set has cardinality-separated subset sums."""
        # sums[k] = set of all subset sums with exactly k elements
        sums = [set() for _ in range(len(elements) + 1)]
        sums[0].add(0)
        
        for val in elements:
            # Update in reverse to avoid using val multiple times
            for k in range(len(sums) - 2, -1, -1):
                for s in sums[k]:
                    sums[k + 1].add(s + val)
        
        # Check that sums at different cardinalities are disjoint
        for i in range(len(sums)):
            for j in range(i + 1, len(sums)):
                if sums[i] & sums[j]:
                    return False
        return True

    def greedy_construct(order, deadline):
        """Greedy: add each element if it doesn't violate the constraint."""
        elements = []
        # sums[k] = set of all subset sums with exactly k elements
        sums = [set() for _ in range(len(order) + 1)]
        sums[0].add(0)
        current_max_k = 0
        
        for val in order:
            if time.time() > deadline:
                break
            
            # Try adding val
            new_sums = [set(s) for s in sums]  # deep copy
            new_max_k = current_max_k + 1
            
            valid = True
            for k in range(current_max_k, -1, -1):
                for s in sums[k]:
                    ns = s + val
                    nk = k + 1
                    if ns in new_sums[nk]:
                        # Check if this sum already exists at a different cardinality
                        for j in range(len(new_sums)):
                            if j != nk and ns in new_sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    else:
                        new_sums[nk].add(ns)
                if not valid:
                    break
            
            if valid:
                # Also need to check against all existing sums
                for k in range(1, new_max_k + 1):
                    for s in new_sums[k]:
                        for j in range(len(new_sums)):
                            if j != k and s in new_sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                sums = new_sums
                current_max_k = new_max_k
                elements.append(val)
        
        return elements

    def repair_set(elements, deadline):
        """Try to add more elements to an existing valid set."""
        elements = list(elements)
        used = set(elements)
        sums = [set() for _ in range(len(elements) + 1)]
        sums[0].add(0)
        
        for val in elements:
            for k in range(len(sums) - 2, -1, -1):
                for s in sums[k]:
                    sums[k + 1].add(s + val)
        
        current_max_k = len(elements)
        
        # Try adding each unused element
        for val in range(1, 257):
            if time.time() > deadline:
                break
            if val in used:
                continue
            
            new_sums = [set(s) for s in sums]
            new_max_k = current_max_k + 1
            
            valid = True
            for k in range(current_max_k, -1, -1):
                for s in sums[k]:
                    ns = s + val
                    nk = k + 1
                    if ns in new_sums[nk]:
                        for j in range(len(new_sums)):
                            if j != nk and ns in new_sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    else:
                        new_sums[nk].add(ns)
                if not valid:
                    break
            
            if valid:
                for k in range(1, new_max_k + 1):
                    for s in new_sums[k]:
                        for j in range(len(new_sums)):
                            if j != k and s in new_sums[j]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                sums = new_sums
                current_max_k = new_max_k
                elements.append(val)
                used.add(val)
        
        return elements

    def swap_local_search(elements, deadline):
        """Try swapping elements to find a larger valid set."""
        elements = list(elements)
        used = set(elements)
        
        # Try removing one element and adding two
        for i in range(len(elements)):
            if time.time() > deadline:
                break
            
            trial = [x for j, x in enumerate(elements) if j != i]
            removed = elements[i]
            used_trial = set(trial)
            
            sums = [set() for _ in range(len(trial) + 1)]
            sums[0].add(0)
            for val in trial:
                for k in range(len(sums) - 2, -1, -1):
                    for s in sums[k]:
                        sums[k + 1].add(s + val)
            current_max_k = len(trial)
            
            # Try adding two new elements
            added = []
            for val1 in range(1, 257):
                if time.time() > deadline:
                    break
                if val1 in used_trial or val1 == removed:
                    continue
                
                new_sums1 = [set(s) for s in sums]
                new_max_k1 = current_max_k + 1
                valid1 = True
                for k in range(current_max_k, -1, -1):
                    for s in sums[k]:
                        ns = s + val1
                        nk = k + 1
                        if ns in new_sums1[nk]:
                            for j in range(len(new_sums1)):
                                if j != nk and ns in new_sums1[j]:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        else:
                            new_sums1[nk].add(ns)
                    if not valid1:
                        break
                
                if valid1:
                    for k in range(1, new_max_k1 + 1):
                        for s in new_sums1[k]:
                            for j in range(len(new_sums1)):
                                if j != k and s in new_sums1[j]:
                                    valid1 = False
                                    break
                            if not valid1:
                                break
                        if not valid1:
                            break
                
                if valid1:
                    # Try adding a second element
                    used_trial2 = set(used_trial) | {val1}
                    for val2 in range(val1 + 1, 257):
                        if time.time() > deadline:
                            break
                        if val2 in used_trial2:
                            continue
                        
                        new_sums2 = [set(s) for s in new_sums1]
                        new_max_k2 = new_max_k1 + 1
                        valid2 = True
                        for k in range(new_max_k1, -1, -1):
                            for s in new_sums1[k]:
                                ns = s + val2
                                nk = k + 1
                                if ns in new_sums2[nk]:
                                    for j in range(len(new_sums2)):
                                        if j != nk and ns in new_sums2[j]:
                                            valid2 = False
                                            break
                                    if not valid2:
                                        break
                                else:
                                    new_sums2[nk].add(ns)
                            if not valid2:
                                break
                        
                        if valid2:
                            for k in range(1, new_max_k2 + 1):
                                for s in new_sums2[k]:
                                    for j in range(len(new_sums2)):
                                        if j != k and s in new_sums2[j]:
                                            valid2 = False
                                            break
                                        if not valid2:
                                            break
                                if not valid2:
                                    break
                        
                        if valid2:
                            # Found a valid swap
                            new_elements = trial + [val1, val2]
                            if len(new_elements) > len(elements):
                                return new_elements
                            break
        
        return elements

    # Baseline: superincreasing sequence (powers of 2)
    baseline = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = baseline
    best_size = len(best)

    # Greedy from smallest to largest
    result = greedy_construct(range(1, 257), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)

    # Greedy from largest to smallest
    result2 = greedy_construct(range(256, 0, -1), deadline)
    if len(result2) > best_size:
        best = result2
        best_size = len(best)

    # Greedy from middle outward
    mid = list(range(129, 257)) + list(range(1, 129))
    result3 = greedy_construct(mid, deadline)
    if len(result3) > best_size:
        best = result3
        best_size = len(best)

    # Try repair on current best
    if time.time() < deadline:
        repaired = repair_set(best, deadline)
        if len(repaired) > best_size:
            best = repaired
            best_size = len(best)

    # Try swap local search
    if time.time() < deadline:
        swapped = swap_local_search(best, deadline)
        if len(swapped) > best_size:
            best = swapped
            best_size = len(best)

    # Try random orders
    rng = random.Random(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        result_r = greedy_construct(order, deadline)
        if len(result_r) > best_size:
            best = result_r
            best_size = len(best)
        if time.time() < deadline - 5:
            repaired_r = repair_set(best, deadline)
            if len(repaired_r) > best_size:
                best = repaired_r
                best_size = len(best)

    # Ensure all elements are distinct ints in 1..256
    best = [int(x) for x in best]
    return best
# EVOLVE-BLOCK-END
