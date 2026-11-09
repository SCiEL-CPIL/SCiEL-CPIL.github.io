"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset has unique pair sums (a+b for a<=b)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        a = subset[i]
        for j in range(i, n):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 0

def count_sidon_4_subsets(arr):
    """Count how many 4-element subsets are Sidon."""
    count = 0
    for sub4 in combinations(arr, 4):
        if is_sidon(sub4):
            count += 1
    return count

def has_any_sidon_4(arr):
    """Check if any 4-element subset is Sidon. Returns True if found."""
    for sub4 in combinations(arr, 4):
        if is_sidon(sub4):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = 70.0  # Safety margin before 90s limit

    # Best known: range(12) gives max Sidon = 4
    best_set = list(range(12))
    best_score = 4  # max_sidon_size(range(12)) = 4

    random.seed(12345)

    # Phase 1: Try structured candidates with many 3-term APs
    # A 3-term AP {a, a+d, a+2d} means a + (a+2d) = 2*(a+d), causing a sum collision
    # If we pack many such structures, we reduce the chance of any 4-subset being Sidon

    structured_candidates = []

    # Pure 3-term APs: {0, d, 2d, 3d, ...} but that's just an AP
    # Better: multiple overlapping 3-term APs with different steps
    # e.g., step 1: 0,1,2,3,4,5 and step 2: 0,2,4,6,8,10 interleaved

    # Generate sets with many small-step 3-term APs
    for base in range(0, 40):
        # Create a set with elements clustered in pairs/triples
        elems = set()
        for i in range(6):
            # Each "cluster" is a 3-term AP with small step
            step = 1
            center = base + i * 4
            elems.add(center)
            elems.add(center + step)
            elems.add(center + 2 * step)
        elems = list(elems)
        if len(elems) >= 12:
            # Take 12 smallest
            elems = sorted(elems)[:12]
            if len(set(elems)) == 12 and all(0 <= x <= 128 for x in elems):
                structured_candidates.append(elems)

    # Also try: pairs of close numbers (distance 1) to create a+b = c+d collisions
    for base in range(0, 50):
        elems = []
        for i in range(6):
            c = base + i * 3
            elems.append(c)
            elems.append(c + 1)
        elems = sorted(set(elems))
        if len(elems) >= 12:
            elems = elems[:12]
            if len(set(elems)) == 12 and all(0 <= x <= 128 for x in elems):
                structured_candidates.append(elems)

    # Try: arithmetic progression with small step (already known to give 4)
    # But also try: two interleaved APs with different steps
    for step1 in [1, 2, 3]:
        for step2 in [2, 3, 4, 5]:
            if step1 == step2:
                continue
            elems = set()
            for i in range(6):
                elems.add(i * step1)
                elems.add(i * step2)
            elems = list(elems)
            if len(elems) >= 12:
                elems = sorted(elems)[:12]
                if len(set(elems)) == 12 and all(0 <= x <= 128 for x in elems):
                    structured_candidates.append(elems)

    for cand in structured_candidates:
        if len(set(cand)) != 12:
            continue
        if not all(0 <= x <= 128 for x in cand):
            continue
        # Quick check: does it have any Sidon 4-subset?
        if not has_any_sidon_4(cand):
            # Verify exact max Sidon size
            actual_max = max_sidon_size(cand)
            if actual_max < best_score:
                best_score = actual_max
                best_set = cand[:]
                if best_score <= 3:
                    return best_set
        else:
            # Count Sidon 4-subsets as a proxy
            s4 = count_sidon_4_subsets(cand)
            if s4 < 10:  # Fewer Sidon 4-subsets is better
                actual_max = max_sidon_size(cand)
                if actual_max < best_score:
                    best_score = actual_max
                    best_set = cand[:]
                elif actual_max == best_score and s4 < 10:
                    # Keep track but don't change best_score
                    pass

    if best_score <= 3:
        return best_set

    # Phase 2: Local search starting from best candidate
    current_set = best_set[:]
    current_s4 = count_sidon_4_subsets(current_set)

    # If we have a set with no Sidon 4-subsets, we're done
    if current_s4 == 0:
        actual_max = max_sidon_size(current_set)
        if actual_max < best_score:
            best_score = actual_max
            best_set = current_set[:]
        return best_set

    # Local search: try to reduce the number of Sidon 4-subsets
    for iteration in range(200000):
        if time.time() - start_time > deadline:
            break

        # Pick a random element to perturb
        idx = random.randint(0, 11)
        old_val = current_set[idx]

        # Try a few new values
        improved = False
        for _ in range(5):
            if time.time() - start_time > deadline:
                break

            # Bias toward values close to existing ones to create collisions
            if random.random() < 0.5:
                # Pick a value near an existing element
                near_idx = random.randint(0, 11)
                base_val = current_set[near_idx]
                offset = random.randint(-3, 3)
                new_val = base_val + offset
            else:
                new_val = random.randint(0, 128)

            if new_val in current_set or new_val < 0 or new_val > 128:
                continue

            neighbor = current_set[:]
            neighbor[idx] = new_val
            neighbor = sorted(neighbor)

            if len(set(neighbor)) != 12:
                continue

            # Quick check: count Sidon 4-subsets
            new_s4 = count_sidon_4_subsets(neighbor)

            if new_s4 < current_s4:
                current_set = neighbor
                current_s4 = new_s4
                improved = True

                # Check if we achieved max Sidon <= 3
                if new_s4 == 0:
                    actual_max = max_sidon_size(current_set)
                    if actual_max < best_score:
                        best_score = actual_max
                        best_set = current_set[:]
                    if best_score <= 3:
                        return best_set
                else:
                    # Even if not 0, check if we improved the overall score
                    # by seeing if max Sidon is lower
                    if new_s4 < 5:  # Very few Sidon 4-subsets
                        actual_max = max_sidon_size(current_set)
                        if actual_max < best_score:
                            best_score = actual_max
                            best_set = current_set[:]
                            if best_score <= 3:
                                return best_set

                break
            elif new_s4 == current_s4 and random.random() < 0.1:
                # Accept with low probability to escape local minima
                current_set = neighbor
                improved = True
                break

    # Final verification
    final_max = max_sidon_size(current_set)
    if final_max < best_score:
        best_score = final_max
        best_set = current_set[:]

    return best_set
# EVOLVE-BLOCK-END
