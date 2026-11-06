"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (evaluator kills at 90s, but we want to be safe)
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute conflict masks
    # conflict_mask[i] contains bits for all j != i such that smooth[i] | smooth[j] or smooth[j] | smooth[i]
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                conflict_mask[i] |= (1 << j)
    
    # Strategy:
    # We want to minimize the total number of terms across all 128 targets.
    # This is equivalent to finding, for each target n, the minimum k such that there exists an antichain of size k summing to n.
    
    # Approach:
    # 1. Enumerate all antichains of the smooth numbers.
    #    The poset is small. We can generate them via backtracking.
    #    State: (next_index_to_consider, current_mask_of_chosen, blocked_mask)
    #    When we pick an element i, we add it to current_mask and add all its conflicts to blocked_mask.
    #    We record (sum, count, mask) for every non-empty antichain generated.
    # 2. For each sum 1..128, keep the antichain with the minimum count.
    # 3. If a sum is not covered (should not happen for 1..128 with this density), fall back to a local search.
    
    # Optimization:
    # We only care about sums <= 128.
    # We can prune the search if the current sum already exceeds 128?
    # No, because we are building the set incrementally. If current_sum > 128, we can prune.
    # However, we don't track current_sum in the state easily without passing it or calculating it.
    # Calculating sum from mask is O(k). Since k is small, it's fine.
    # But for pruning, we need to know the sum before expanding.
    # Let's pass current_sum and current_count in the recursion.
    
    best_for_sum = [None] * 129  # best_for_sum[s] = (min_count, mask)
    
    def generate_antichains(index, current_mask, blocked_mask, current_sum, current_count):
        # If current_set is not empty, record it if sum is valid
        if current_count > 0:
            if current_sum <= 128:
                if best_for_sum[current_sum] is None or current_count < best_for_sum[current_sum][0]:
                    best_for_sum[current_sum] = (current_count, current_mask)
        
        # Pruning: if current_sum > 128, no point in adding more positive numbers
        if current_sum > 128:
            return
        
        # Check time
        if time.time() > deadline:
            return

        for i in range(index, n_smooth):
            if time.time() > deadline:
                return
            
            # If i is blocked, skip
            if blocked_mask & (1 << i):
                continue
            
            # Optional pruning: if current_sum + smooth[i] > 128, we might still want to check if adding MORE
            # numbers would help? No, all numbers are positive. So if current_sum + smooth[i] > 128,
            # any superset including i will also exceed 128 (since all other added numbers are >= 1).
            # Wait, we are adding i to the set. The new sum is current_sum + smooth[i].
            # If that exceeds 128, we can skip this branch entirely?
            # Yes, because all subsequent additions will only increase the sum.
            if current_sum + smooth[i] > 128:
                continue
            
            new_mask = current_mask | (1 << i)
            new_blocked = blocked_mask | (1 << i) | conflict_mask[i]
            new_sum = current_sum + smooth[i]
            new_count = current_count + 1
            
            generate_antichains(i + 1, new_mask, new_blocked, new_sum, new_count)
            
            # Note: We do NOT have a "skip i" branch in the standard way because `index` is passed as i+1
            # in the recursive call, but the loop continues to i+1 naturally in the for loop.
            # The loop `for i in range(index, n_smooth)` handles the "skip" by just moving to the next i.
            # The recursion handles the "take" branch.
            # This is a standard subset generation pattern.
            
    generate_antichains(0, 0, 0, 0, 0)
    
    # Check if all targets 1..128 are covered
    answers = []
    missing = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            cnt, mask = best_for_sum[n]
            terms = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                terms.append(smooth[idx])
                temp_mask ^= lsb
            answers.append(terms)
        else:
            answers.append(None)
            missing.append(n)
    
    # Fallback for any missing targets (should be empty, but just in case)
    # Use a simple backtracking for specific missing targets
    if missing:
        for n in missing:
            found = None
            # Try small k first
            for k in range(1, 8):
                candidates = [x for x in smooth if x <= n]
                # If k > len(candidates), break
                if k > len(candidates):
                    break
                for combo in combinations(candidates, k):
                    if sum(combo) != n:
                        continue
                    # Check antichain
                    valid = True
                    for a in range(len(combo)):
                        for b in range(a+1, len(combo)):
                            if combo[a] % combo[b] == 0 or combo[b] % combo[a] == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        found = list(combo)
                        break
                if found:
                    break
            if found:
                answers[n-1] = found
            else:
                # Last resort: if n is smooth, [n] is valid (size 1 antichain)
                if n in smooth:
                    answers[n-1] = [n]
                else:
                    # This should never happen for 1..128
                    answers[n-1] = []
    
    return answers
# EVOLVE-BLOCK-END
