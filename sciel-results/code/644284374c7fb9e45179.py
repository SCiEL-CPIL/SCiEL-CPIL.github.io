import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # The optimal 20-mark Golomb ruler with length 283 is known.
    # We need to check if it contains 1, 2, 5, 11.
    # If not, we need to find a Sidon set (distinct pairwise sums a+b for a<=b) 
    # which is equivalent to a Golomb ruler if we consider differences, but here it's sums.
    # Wait, the task says "pairwise sums a + b (a <= b) are all distinct".
    # This is a Sidon set (B_2 set) in the additive sense, but specifically for sums.
    # A Golomb ruler requires distinct differences.
    # A set with distinct pairwise sums is a Sidon set.
    # Let's verify: if a+b = c+d with {a,b} != {c,d}, it's not a Sidon set.
    # The problem statement says "without the prescribed elements the maximum in [1, 284] is 20".
    # This suggests we are looking for a large Sidon set.
    
    # Known optimal Sidon sets for small n are often related to Golomb rulers but not identical.
    # However, the prompt mentions "optimal 20-mark Golomb ruler has length 283".
    # This implies the benchmark is likely a Golomb ruler, but the constraint is on sums.
    # Let's re-read carefully: "pairwise sums a + b (a <= b) are all distinct".
    # This is a B_2 set.
    
    # Let's try to construct a large B_2 set containing 1, 2, 5, 11 in [1, 284].
    # We can use a greedy approach with backtracking or local search.
    
    # First, let's see if the known optimal Golomb ruler for 20 marks fits.
    # A Golomb ruler ensures distinct differences. Does it ensure distinct sums?
    # Not necessarily.
    # Example: {0, 1, 4}. Sums: 0, 1, 4, 5, 8. Distinct.
    # Example: {0, 2, 3}. Sums: 0, 2, 3, 4, 5, 6. Distinct.
    # Actually, any Sidon set (distinct sums) is a Golomb ruler? No.
    # A Golomb ruler has distinct differences. A Sidon set has distinct sums.
    # They are dual concepts.
    
    # Let's use a local search algorithm to maximize the size of the set.
    # Start with the prescribed elements and add more.
    
    prescribed = [1, 2, 5, 11]
    current_set = set(prescribed)
    current_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            current_sums.add(x + y)
    
    def is_valid_addition(candidate, current_set, current_sums):
        new_sums = set()
        for x in current_set:
            s = candidate + x
            if s in current_sums or s in new_sums:
                return False
            new_sums.add(s)
        s = 2 * candidate
        if s in current_sums or s in new_sums:
            return False
        return True
    
    # Greedy initialization
    for x in range(12, 285):
        if is_valid_addition(x, current_set, current_sums):
            current_set.add(x)
            for y in list(current_set)[:-1]:
                current_sums.add(x + y)
            current_sums.add(2 * x)
    
    best_set = list(current_set)
    best_size = len(best_set)
    
    # Local search: try to swap elements to increase size or maintain size while allowing better placement
    # Since we want to maximize size, we can try to add more elements by removing some.
    # Or we can use simulated annealing.
    
    # Let's try a simple local search: try to add a random element, if it fails, try to remove one to make space.
    # Better: use a heuristic to find a maximal Sidon set.
    
    # Given the time constraint, a robust greedy + local search is best.
    
    # Let's try to improve the greedy result by checking if we can swap out an element for a better one.
    # But since we want max size, we should try to add more.
    
    # The greedy algorithm above adds elements in increasing order.
    # We can try randomizing the order or starting points.
    
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Randomize the order of candidate additions
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        trial_set = set(prescribed)
        trial_sums = set()
        for i, x in enumerate(prescribed):
            for y in prescribed[i:]:
                trial_sums.add(x + y)
        
        for x in candidates:
            if x in trial_set:
                continue
            if is_valid_addition(x, trial_set, trial_sums):
                trial_set.add(x)
                for y in list(trial_set)[:-1]:
                    trial_sums.add(x + y)
                trial_sums.add(2 * x)
        
        if len(trial_set) > best_size:
            best_size = len(trial_set)
            best_set = list(trial_set)
            
    # Final check and return
    # Ensure the returned list is sorted and valid
    best_set.sort()
    return best_set

