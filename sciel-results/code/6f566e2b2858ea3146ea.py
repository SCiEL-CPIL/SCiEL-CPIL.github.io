"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    N = 1000
    # sum_counts[s] = number of pairs (a,b) in chosen with a<=b and a+b=s
    sum_counts = np.zeros(2 * N + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    
    # Phase 1: Greedy construction with O(1) validity check via array lookup
    for x in range(1, N + 1):
        if time.time() > deadline - 30:
            break
        # Check if adding x would violate the constraint
        # New sums: x+x=2x, and x+y for y in chosen
        # We need to check that no sum exceeds count 2
        valid = True
        # Check 2x
        if sum_counts[2 * x] >= 2:
            valid = False
        else:
            # Check x+y for all y in chosen
            for y in chosen:
                if sum_counts[x + y] >= 2:
                    valid = False
                    break
        if valid:
            # Update sum_counts
            sum_counts[2 * x] += 1
            for y in chosen:
                sum_counts[x + y] += 1
            chosen.append(x)
            chosen_set.add(x)
    
    # Phase 2: Local search - try to replace elements to increase size
    # Try adding elements that were skipped, possibly by removing others
    best_chosen = chosen[:]
    best_count = len(chosen)
    
    # Build a list of candidates that weren't chosen
    skipped = [x for x in range(1, N + 1) if x not in chosen_set]
    
    # Try to add each skipped element by removing one or more existing elements
    # to make room. We'll use a greedy local search.
    remaining_time_budget = deadline - 5  # keep 5s for cleanup
    
    # Try single swaps: remove one element, add one
    for add_candidate in skipped:
        if time.time() > remaining_time_budget:
            break
        if len(chosen) >= best_count:
            continue
        
        # Try adding add_candidate: check which existing elements conflict
        # We need to find a subset of chosen to remove so that add_candidate fits
        # Simple approach: try removing each single element
        for i, rem in enumerate(chosen):
            if time.time() > remaining_time_budget:
                break
            # Temporarily remove rem, check if add_candidate fits
            # Compute the new sum counts after removing rem
            # Sums involving rem: rem+rem, rem+y for y in chosen, y!=rem
            # This is O(|chosen|) per trial, but we can be smarter
            
            # Quick check: does add_candidate conflict with remaining elements?
            # After removing rem, the elements are chosen - {rem}
            # We need: for all pairs (a,b) in (chosen-{rem}) ∪ {add_candidate} with a<=b,
            # the count of a+b across all pairs in the new set is <= 2
            
            # Efficient check: 
            # 1. Remove rem's contributions from sum_counts
            # 2. Check if add_candidate can be added
            # 3. If yes, commit; if no, restore
            
            # Remove rem's contributions
            old_counts_backup = None
            # Sums from rem: 2*rem, and rem+y for y in chosen, y != rem
            # Actually rem pairs with all other elements in chosen
            rem_sums = [2 * rem] + [rem + y for y in chosen if y != rem]
            for s in rem_sums:
                sum_counts[s] -= 1
            
            # Now check if add_candidate fits
            valid = True
            if sum_counts[2 * add_candidate] >= 2:
                valid = False
            else:
                for y in chosen:
                    if y != rem and sum_counts[add_candidate + y] >= 2:
                        valid = False
                        break
            
            if valid:
                # Commit the swap
                for s in rem_sums:
                    sum_counts[s] += 0  # already removed
                sum_counts[2 * add_candidate] += 1
                for y in chosen:
                    if y != rem:
                        sum_counts[add_candidate + y] += 1
                chosen[i] = add_candidate
                chosen_set.discard(rem)
                chosen_set.add(add_candidate)
                if len(chosen) > best_count:
                    best_count = len(chosen)
                    best_chosen = chosen[:]
            else:
                # Restore rem's contributions
                for s in rem_sums:
                    sum_counts[s] += 1
            
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_chosen = chosen[:]
    
    # Phase 3: Try to add more elements after swaps
    # Re-check skipped elements
    current_set = set(best_chosen)
    new_skipped = [x for x in range(1, N + 1) if x not in current_set]
    
    # Rebuild sum_counts from best_chosen
    sum_counts2 = np.zeros(2 * N + 1, dtype=np.int32)
    for i, a in enumerate(best_chosen):
        for j in range(i, len(best_chosen)):
            sum_counts2[a + best_chosen[j]] += 1
    
    for x in new_skipped:
        if time.time() > deadline - 2:
            break
        valid = True
        if sum_counts2[2 * x] >= 2:
            valid = False
        else:
            for y in best_chosen:
                if sum_counts2[x + y] >= 2:
                    valid = False
                    break
        if valid:
            sum_counts2[2 * x] += 1
            for y in best_chosen:
                sum_counts2[x + y] += 1
            best_chosen.append(x)
            current_set.add(x)
    
    return best_chosen
# EVOLVE-BLOCK-END
