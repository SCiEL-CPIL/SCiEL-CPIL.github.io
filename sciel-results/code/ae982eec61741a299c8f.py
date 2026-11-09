"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from heapq import nlargest

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    denominators = list(range(2, 49))
    n = len(denominators)

    # Compute LCM of 2..48 for exact integer arithmetic
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator: 1/d = (lcm_val/d) / lcm_val
    nums = [lcm_val // d for d in denominators]

    # Sort indices by weight descending (largest unit fractions first)
    sorted_indices = sorted(range(n), key=lambda i: -nums[i])

    def beam_search(beam_width, order):
        """
        Beam search: at each step, maintain top-K partial sums.
        Each state is (total_sum, frozenset_of_indices).
        We process elements in 'order', and at each step decide include/exclude.
        Keep only the top beam_width states by total sum.
        """
        # State: (total, frozenset)
        # Start with empty set
        beam = [(0, frozenset())]
        
        for idx in order:
            vi = nums[idx]
            new_beam = []
            for total, s in beam:
                # Option 1: exclude idx
                new_beam.append((total, s))
                # Option 2: include idx if sum stays < lcm_val
                new_total = total + vi
                if new_total < lcm_val:
                    new_beam.append((new_total, s | frozenset([idx])))
            
            # Keep top beam_width by total sum
            if len(new_beam) > beam_width:
                # Sort by total descending, take top beam_width
                new_beam.sort(key=lambda x: x[0], reverse=True)
                new_beam = new_beam[:beam_width]
            beam = new_beam
        
        # Return the best state
        if not beam:
            return 0, frozenset()
        best_total, best_set = max(beam, key=lambda x: x[0])
        return best_total, best_set

    def local_search(selected, total):
        """Improve via 1-add and 1-swap moves."""
        in_set = [False] * n
        for i in selected:
            in_set[i] = True

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False

            # 1-add: try adding an unselected element
            best_add = None
            for j in range(n):
                if not in_set[j]:
                    if total + nums[j] < lcm_val:
                        if best_add is None or nums[j] > nums[best_add]:
                            best_add = j
            if best_add is not None:
                selected.append(best_add)
                in_set[best_add] = True
                total += nums[best_add]
                improved = True
                continue

            # 1-swap: remove i, add j where gain > 0 and new total < lcm_val
            best_gain = 0
            best_move = None
            for i in selected:
                vi = nums[i]
                for j in range(n):
                    if not in_set[j]:
                        vj = nums[j]
                        gain = vj - vi
                        if gain > 0:
                            new_total = total - vi + vj
                            if new_total < lcm_val:
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (i, j)
            if best_move:
                i, j = best_move
                in_set[i] = False
                in_set[j] = True
                selected.remove(i)
                selected.append(j)
                total += nums[j] - nums[i]
                improved = True

        return selected, total

    # Best result tracking
    best_indices = None
    best_sum = 0

    # Try beam search with increasing beam widths
    beam_widths = [10, 50, 100, 200, 500]
    
    for bw in beam_widths:
        if time.time() > deadline - 5.0:
            break
        total, s = beam_search(bw, sorted_indices)
        if total > best_sum:
            best_sum = total
            best_indices = list(s)
        
        # Apply local search to the beam result
        if best_indices:
            sel, tot = local_search(list(best_indices), best_sum)
            if tot > best_sum:
                best_sum = tot
                best_indices = sel

    # Also try a few random orderings with moderate beam width
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        perm = list(range(n))
        random.shuffle(perm)
        total, s = beam_search(50, perm)
        if total > best_sum:
            best_sum = total
            best_indices = list(s)
            if best_indices:
                sel, tot = local_search(list(best_indices), best_sum)
                if tot > best_sum:
                    best_sum = tot
                    best_indices = sel

    # Fallback: if best_indices is empty, use a single element
    if not best_indices:
        best_indices = [0]  # index 0 corresponds to denominator 2

    return best_indices
# EVOLVE-BLOCK-END
