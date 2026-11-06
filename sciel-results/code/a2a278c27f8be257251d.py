import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin
    
    # 1. Generate initial set using the seed's greedy logic
    def greedy_construct():
        chosen, counts = [], {}
        for x in range(1, 1001):
            new_sums = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts = trial
        return chosen, counts

    best_set, best_counts = greedy_construct()
    best_size = len(best_set)
    
    # Helper to calculate slack
    def calc_slack(counts):
        slack = 0
        for s, c in counts.items():
            if c < 2:
                slack += (2 - c)
        return slack

    best_slack = calc_slack(best_counts)
    
    # Local Search to increase slack and size
    while time.time() < deadline:
        # Phase 1: Try to add elements
        added = False
        # Sort candidates by how "safe" they are? Just iterate.
        # To make it faster, we can precompute which sums are tight.
        # But for N=1000, checking all 1000 candidates is cheap (1000 * 1000 = 1M ops max).
        current_set = set(best_set)
        
        # We need to check if adding x is valid.
        # Optimization: Only consider x that don't immediately violate.
        for x in range(1, 1001):
            if x in current_set:
                continue
            
            # Check validity of adding x
            valid = True
            # We need to check sums x+y for y in best_set, and 2x.
            # We can do this quickly by checking if any count[x+y] is 2.
            # If count[x+y] is 2, adding x makes it 3 -> invalid.
            # If count[2x] is 2, adding x makes it 3 -> invalid.
            
            # Note: best_counts only contains sums from best_set.
            # When we add x, new sums are x+y.
            # We just need to ensure no existing sum s=x+y has count 2, and 2x doesn't have count 2.
            
            # Check 2x
            if best_counts.get(2*x, 0) >= 2:
                continue
                
            # Check x+y
            # If any y in best_set has best_counts[x+y] == 2, invalid.
            # We can iterate y in best_set.
            # To speed up, we can break early.
            for y in best_set:
                if best_counts.get(x+y, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                # Accept the addition
                best_set.append(x)
                best_set.sort()
                current_set.add(x)
                
                # Update counts
                for y in best_set:
                    if y < x: # Avoid double counting? 
                         # Actually, the seed logic adds x+y for all y in chosen.
                         # Since we are adding x to the end, y < x is not guaranteed in the list if we sorted.
                         # Let's just update all sums involving x.
                         pass
                 
                # Simpler update: recompute counts? No, that's O(N^2).
                # Update incrementally.
                # Sums to update: x+y for all y in best_set (excluding x itself to avoid double count of 2x? 
                # The seed logic: new = [x + y for y in chosen] + [2 * x]
                # So we add 1 to count[x+y] for all y in OLD best_set, and 1 to count[2x].
                
                # Let's revert the append to do this cleanly
                best_set.pop()
                
                old_set = list(best_set)
                for y in old_set:
                    s = x + y
                    best_counts[s] = best_counts.get(s, 0) + 1
                s = 2*x
                best_counts[s] = best_counts.get(s, 0) + 1
                
                best_set.append(x)
                best_set.sort()
                added = True
                break
        
        if added:
            best_size = len(best_set)
            best_slack = calc_slack(best_counts)
            continue
            
        # Phase 2: If no addition, try to improve slack via swaps
        # Find a swap (i in best_set, j not in best_set) that maintains validity and increases slack
        improved = False
        current_set = set(best_set)
        
        # To speed up, we can limit the number of swaps attempted per epoch
        # or just iterate through all i.
        for i in best_set:
            # Try to remove i
            # This is expensive to check all j.
            # Heuristic: Only try to swap with j that are "close" to i or random j.
            # For now, let's try a few random j for each i.
            # If we do this too much, we might hit the deadline.
            
            # Let's try a limited number of random swaps
            # If we find a swap that increases slack, do it.
            
            # This part is tricky to implement efficiently without O(N^3) checks.
            # Given the time limit, and that Phase 1 often yields many additions,
            # maybe we can just rely on Phase 1 with a slightly different greedy order?
            
            # Alternative: Restart greedy with a different randomization?
            # The seed is deterministic.
            
            # Let's try a simple heuristic: 
            # If we can't add, we are stuck. 
            # We can try to "remove" an element that has many tight sums, and "add" one that has few.
            
            # Calculate "cost" of each element i: number of sums s=x+y involving i that have count 2.
            # If we remove i, those sums drop to 1 (slack +1 each).
            # If we add j, sums t=j+y might go to 2 (slack -1 each) or 3 (invalid).
            
            # This is getting complex. Let's stick to a simpler improvement:
            # Randomized Greedy.
            # Instead of 1, 2, 3..., shuffle the order?
            # No, the seed is already a valid lower bound.
            
            # Let's try one more thing: 
            # If we can't add, try to swap i for j where j is a "better" candidate.
            # We'll just break out of the local search if it's taking too long per step.
            break

    return best_set

# The above code structure is a bit messy in the thought process. 
# Let's refine the local search to be cleaner and safer.
