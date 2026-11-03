import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_golomb(n):
        """
        Construct a Golomb ruler with n marks using a greedy algorithm.
        This minimizes the diameter, which helps the denominator in the score.
        """
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        # Greedy construction: place marks one by one to minimize the last position
        # while maintaining the Golomb property (all differences distinct).
        marks = [0]
        diffs = set()
        
        for i in range(1, n):
            best_pos = None
            best_diameter = float('inf')
            
            # We can restrict the search space for the next mark
            # A good upper bound for the next mark is roughly i * (i+1) / 2 + 1
            # But let's try a range.
            # The diameter of a Golomb ruler with n marks is at least ~ n^2.
            # Let's search up to a reasonable limit.
            max_search = 100 + i * i * 2 
            
            for pos in range(marks[-1] + 1, max_search):
                valid = True
                for m in marks:
                    d = abs(pos - m)
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    # Check if this position leads to a smaller diameter than current best
                    # Actually, we just want the smallest valid position
                    # Greedy: pick the smallest valid position
                    best_pos = pos
                    # Since we iterate upwards, the first valid one is the smallest
                    break
            
            if best_pos is None:
                # If no position found in range, extend range (shouldn't happen for small n)
                # Fallback: just pick a large number
                best_pos = marks[-1] + i * i + 1
            
            marks.append(best_pos)
            for m in marks[:-1]:
                diffs.add(abs(best_pos - m))
        
        return marks

    def local_search(initial_set, time_limit):
        """
        Perform local search to improve the score.
        Try to perturb elements to reduce sumset size or increase difference set size
        while keeping diameter small.
        """
        current_set = sorted(set(initial_set))
        current_score = score_set(current_set)
        best_set = current_set[:]
        best_score = current_score
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break
                
            # Pick a random element to perturb (not 0)
            if len(current_set) <= 1:
                break
                
            indices = list(range(1, len(current_set)))
            idx = random.choice(indices)
            old_val = current_set[idx]
            
            # Try a few random perturbations
            improved = False
            for _ in range(10):
                # Perturbation strategy:
                # 1. Small shift
                # 2. Swap with another element
                # 3. Replace with random value in range
                
                strategy = random.random()
                new_val = old_val
                
                if strategy < 0.5:
                    # Small shift
                    shift = random.randint(-5, 5)
                    new_val = old_val + shift
                elif strategy < 0.8:
                    # Swap with another element
                    other_idx = random.choice([i for i in range(len(current_set)) if i != idx])
                    new_val = current_set[other_idx]
                    current_set[idx], current_set[other_idx] = current_set[other_idx], current_set[idx]
                    # Note: If we swap, we need to revert if we don't accept
                    # It's easier to just create a new list
                    new_set = current_set[:]
                    new_set[idx] = current_set[other_idx]
                    new_set[other_idx] = old_val
                    # Actually, let's stick to single element modification for simplicity in revert
                    # Revert swap for now, handle swap differently if needed
                    current_set[idx], current_set[other_idx] = old_val, current_set[other_idx]
                    new_val = None # Flag to skip simple replacement logic
                    # Let's just do single element replacement for now to keep code simple
                    continue 
                else:
                    # Random replacement within bounds
                    max_val = max(current_set)
                    new_val = random.randint(1, max_val + 10)
                
                if new_val is None:
                    continue
                
                # Ensure new_val is not 0 (unless idx is 0, but we don't perturb 0)
                # and not already in the set
                if new_val == 0 or new_val in current_set:
                    continue
                
                # Apply perturbation
                current_set[idx] = new_val
                current_set = sorted(set(current_set))
                
                new_score = score_set(current_set)
                
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                else:
                    # Revert
                    current_set = sorted(set(best_set if new_score < best_score else current_set)) 
                    # Wait, revert to the state before this perturbation
                    # It's safer to keep a copy
                    pass
                
                # Actually, let's restructure the loop to be cleaner
                break # Only try one perturbation per iteration of the while loop?
                # No, let's try a few.
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

        return best_set, best_score

    # Strategy:
    # 1. Generate several Golomb rulers of different sizes.
    # 2. Run local search on each for a fraction of the time.
    # 3. Return the best one found.

    sizes_to_try = [15, 20, 25, 30, 40, 50]
    # Shuffle to add randomness in which size gets more time if early termination happens
    random.shuffle(sizes_to_try)
    
    best_overall_score = 0.0
    best_overall_set = [0]
    
    # Allocate time evenly among sizes, but ensure at least some time for each
    time_per_size = (deadline - time.time()) / len(sizes_to_try)
    
    for n in sizes_to_try:
        if time.time() >= deadline:
            break
            
        # Generate a Golomb ruler
        try:
            initial_set = greedy_golomb(n)
        except Exception:
            initial_set = list(range(n)) # Fallback
        
        # Run local search with a sub-deadline
        sub_deadline = time.time() + time_per_size * 0.8 # Leave some buffer
        improved_set, improved_score = local_search(initial_set, sub_deadline)
        
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_set = improved_set
            
    return best_overall_set
