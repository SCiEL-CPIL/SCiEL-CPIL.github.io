"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECS = [Fraction(1, d) for d in DENOMS]

def _subset_sum_fractions(indices):
    total = Fraction(0, 1)
    for i in indices:
        total += RECS[i]
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    best_indices = None
    best_deficit = None
    
    def consider(indices):
        nonlocal best_indices, best_deficit
        if not indices:
            return
        s = _subset_sum_fractions(indices)
        if s < 1:
            deficit = Fraction(1, 1) - s
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = list(indices)
    
    # Phase 1: Greedy from smallest denominators (largest reciprocals)
    greedy_indices = []
    total = Fraction(0, 1)
    for i in range(N):
        if total + RECS[i] < 1:
            greedy_indices.append(i)
            total += RECS[i]
    consider(greedy_indices)
    
    # Phase 2: Exhaustive search for small subsets (sizes 2-6)
    from itertools import combinations
    for size in range(2, 7):
        if time.time() > deadline - 5:
            break
        for combo in combinations(range(N), size):
            if time.time() > deadline - 2:
                break
            s = sum(RECS[i] for i in combo)
            if s < 1:
                deficit = Fraction(1, 1) - s
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(combo)
    
    # Phase 3: Local search / hill climbing from current best
    if best_indices is not None:
        selected_set = set(best_indices)
        unselected_set = set(range(N)) - selected_set
        current_sum_val = _subset_sum_fractions(best_indices)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding each unselected index
            for i in sorted(unselected_set):
                if time.time() > deadline - 3:
                    break
                new_sum = current_sum_val + RECS[i]
                if new_sum < 1:
                    new_deficit = Fraction(1, 1) - new_sum
                    if new_deficit < best_deficit:
                        best_indices = best_indices + [i]
                        best_deficit = new_deficit
                        selected_set.add(i)
                        unselected_set.remove(i)
                        current_sum_val = new_sum
                        improved = True
            
            if not improved:
                # Try swaps
                for out_i in sorted(selected_set):
                    if time.time() > deadline - 3:
                        break
                    for in_i in sorted(unselected_set):
                        if time.time() > deadline - 3:
                            break
                        new_sum = current_sum_val - RECS[out_i] + RECS[in_i]
                        if new_sum < 1:
                            new_deficit = Fraction(1, 1) - new_sum
                            if new_deficit < best_deficit:
                                best_indices = [x for x in best_indices if x != out_i] + [in_i]
                                best_deficit = new_deficit
                                selected_set.remove(out_i)
                                selected_set.add(in_i)
                                unselected_set.add(out_i)
                                unselected_set.remove(in_i)
                                current_sum_val = new_sum
                                improved = True
                                break
                    if improved:
                        break
        
        # Try adding pairs of unselected
        if time.time() < deadline - 3:
            unselected_list = list(unselected_set)
            for a_idx, i in enumerate(unselected_list):
                if time.time() > deadline - 2:
                    break
                for j in unselected_list[a_idx+1:]:
                    if time.time() > deadline - 2:
                        break
                    new_sum = current_sum_val + RECS[i] + RECS[j]
                    if new_sum < 1:
                        new_deficit = Fraction(1, 1) - new_sum
                        if new_deficit < best_deficit:
                            best_indices = best_indices + [i, j]
                            best_deficit = new_deficit
                            selected_set.add(i)
                            selected_set.add(j)
                            unselected_set.remove(i)
                            unselected_set.remove(j)
                            current_sum_val = new_sum
    
    # Phase 4: Randomized restarts
    rng = random.Random(12345)
    
    def random_greedy():
        order = list(range(N))
        rng.shuffle(order)
        indices = []
        total = Fraction(0, 1)
        for i in order:
            if total + RECS[i] < 1:
                indices.append(i)
                total += RECS[i]
        consider(indices)
    
    def random_subset_trial():
        k = rng.randint(1, N)
        indices = rng.sample(range(N), k)
        consider(indices)
    
    # Phase 5: Targeted search - try to improve by looking at near-optimal candidates
    if time.time() < deadline - 10:
        # Try random restarts with greedy
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            random_greedy()
        
        # Try random subsets
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            random_subset_trial()
        
        # Local search from random solutions
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Generate random starting point
            k = rng.randint(5, 20)
            start_indices = rng.sample(range(N), k)
            s = sum(RECS[i] for i in start_indices)
            
            if s < 1:
                # Local search from this point
                sel_set = set(start_indices)
                unsel_set = set(range(N)) - sel_set
                cur_sum = s
                cur_deficit = Fraction(1, 1) - s
                
                # Try additions
                for i in sorted(unsel_set):
                    if time.time() > deadline - 3:
                        break
                    new_sum = cur_sum + RECS[i]
                    if new_sum < 1:
                        new_deficit = Fraction(1, 1) - new_sum
                        if new_deficit < cur_deficit:
                            start_indices.append(i)
                            cur_deficit = new_deficit
                            sel_set.add(i)
                            unsel_set.remove(i)
                            cur_sum = new_sum
                
                # Try swaps
                for out_i in list(sel_set):
                    if time.time() > deadline - 3:
                        break
                    for in_i in sorted(unsel_set):
                        if time.time() > deadline - 3:
                            break
                        new_sum = cur_sum - RECS[out_i] + RECS[in_i]
                        if new_sum < 1:
                            new_deficit = Fraction(1, 1) - new_sum
                            if new_deficit < cur_deficit:
                                start_indices = [x for x in start_indices if x != out_i] + [in_i]
                                cur_deficit = new_deficit
                                sel_set.remove(out_i)
                                sel_set.add(in_i)
                                unsel_set.add(out_i)
                                unsel_set.remove(in_i)
                                cur_sum = new_sum
                                break
                    if cur_deficit < best_deficit:
                        best_deficit = cur_deficit
                        best_indices = list(start_indices)
    
    if best_indices is None:
        return greedy_indices
    return best_indices
# EVOLVE-BLOCK-END
