"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute unit fractions
    unit_fracs = [Fraction(1, d) for d in denominators]
    one = Fraction(1, 1)
    
    # Greedy seed: pick denominators in increasing order while sum < 1
    # This gives a baseline
    greedy_indices = []
    greedy_total = Fraction(0)
    for i in range(n):
        if greedy_total + unit_fracs[i] < one:
            greedy_indices.append(i)
            greedy_total += unit_fracs[i]
    
    best_deficit = one - greedy_total
    best_indices = list(greedy_indices)
    
    # Local improvement: try to swap or add/remove to reduce deficit
    # Key insight: we want sum as close to 1 from below as possible.
    # Strategy: start from greedy, then try local moves.
    
    def get_deficit(indices):
        s = Fraction(0)
        for i in indices:
            s += unit_fracs[i]
        return one - s
    
    # Check if adding a denominator that's not in the set keeps us < 1
    # and reduces deficit (i.e., brings sum closer to 1)
    
    current_set = set(best_indices)
    current_sum = greedy_total
    current_deficit = best_deficit
    
    # Iterative local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding each non-selected index
        best_add_gain = 0
        best_add_idx = -1
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum + unit_fracs[i]
                if new_sum < one:
                    gain = unit_fracs[i]  # deficit decreases by this amount
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add_idx = i
        
        if best_add_idx >= 0:
            current_set.add(best_add_idx)
            current_sum += unit_fracs[best_add_idx]
            current_deficit -= best_add_gain
            improved = True
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_indices = list(current_set)
        
        # Try swapping: remove one selected, add one non-selected
        if time.time() < deadline:
            best_swap_gain = 0
            best_swap_remove = -1
            best_swap_add = -1
            selected_list = list(current_set)
            for ri in selected_list:
                for ai in range(n):
                    if ai in current_set:
                        continue
                    # Remove ri, add ai
                    new_sum = current_sum - unit_fracs[ri] + unit_fracs[ai]
                    if new_sum < one:
                        gain = unit_fracs[ai] - unit_fracs[ri]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_remove = ri
                            best_swap_add = ai
            
            if best_swap_remove >= 0 and best_swap_add >= 0:
                current_set.remove(best_swap_remove)
                current_set.add(best_swap_add)
                current_sum = current_sum - unit_fracs[best_swap_remove] + unit_fracs[best_swap_add]
                current_deficit = one - current_sum
                improved = True
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_indices = list(current_set)
        
        # Try removing one and adding two
        if time.time() < deadline:
            best_pair_gain = 0
            best_pair_remove = -1
            best_pair_add1 = -1
            best_pair_add2 = -1
            selected_list = list(current_set)
            for ri in selected_list:
                for ai in range(n):
                    if ai in current_set:
                        continue
                    for bi in range(ai+1, n):
                        if bi in current_set:
                            continue
                        new_sum = current_sum - unit_fracs[ri] + unit_fracs[ai] + unit_fracs[bi]
                        if new_sum < one:
                            gain = unit_fracs[ai] + unit_fracs[bi] - unit_fracs[ri]
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair_remove = ri
                                best_pair_add1 = ai
                                best_pair_add2 = bi
            
            if best_pair_remove >= 0:
                current_set.remove(best_pair_remove)
                current_set.add(best_pair_add1)
                current_set.add(best_pair_add2)
                current_sum = current_sum - unit_fracs[best_pair_remove] + unit_fracs[best_pair_add1] + unit_fracs[best_pair_add2]
                current_deficit = one - current_sum
                improved = True
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_indices = list(current_set)
    
    # Final verification
    final_sum = Fraction(0)
    for i in best_indices:
        final_sum += unit_fracs[i]
    
    assert final_sum < one, f"Sum {final_sum} is not strictly below 1"
    assert len(best_indices) > 0, "Empty selection"
    assert len(set(best_indices)) == len(best_indices), "Duplicate indices"
    
    return best_indices
# EVOLVE-BLOCK-END
