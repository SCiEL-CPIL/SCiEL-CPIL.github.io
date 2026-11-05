"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    best_deficit = Fraction(1)  # start with worst possible
    best_indices = None
    
    def time_up():
        return time.time() >= deadline
    
    def evaluate(indices):
        """Return deficit for a set of indices, or None if sum >= 1"""
        total = sum((fracs[i] for i in indices), Fraction(0))
        if total >= 1:
            return None
        return 1 - total
    
    # Strategy 1: Greedy - pick largest fractions that fit
    def greedy_largest():
        nonlocal best_deficit, best_indices
        selected = []
        total = Fraction(0)
        # Sort by fraction value descending (i.e., denominator ascending)
        for i in range(n):  # i=0 is 1/2, i=1 is 1/3, etc.
            if total + fracs[i] < 1:
                selected.append(i)
                total += fracs[i]
        deficit = 1 - total
        if deficit < best_deficit:
            best_deficit = deficit
            best_indices = selected[:]
        return selected
    
    # Strategy 2: Greedy - pick smallest fractions that fit (to leave room)
    def greedy_smallest():
        nonlocal best_deficit, best_indices
        selected = []
        total = Fraction(0)
        for i in range(n - 1, -1, -1):  # i=46 is 1/48, etc.
            if total + fracs[i] < 1:
                selected.append(i)
                total += fracs[i]
        deficit = 1 - total
        if deficit < best_deficit:
            best_deficit = deficit
            best_indices = selected[:]
        return selected
    
    # Local search
    def local_search(initial_indices, max_iters=10000):
        nonlocal best_deficit, best_indices
        current = set(initial_indices)
        current_sum = sum((fracs[i] for i in current), Fraction(0))
        current_deficit = 1 - current_sum
        
        for _ in range(max_iters):
            if time_up():
                break
            improved = False
            
            # Try adding
            for i in range(n):
                if i not in current:
                    new_sum = current_sum + fracs[i]
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < current_deficit:
                            current.add(i)
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                            break
            if improved:
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_indices = sorted(current)
                continue
            
            # Try removing
            for i in list(current):
                new_sum = current_sum - fracs[i]
                new_deficit = 1 - new_sum
                if new_deficit < current_deficit:
                    current.remove(i)
                    current_sum = new_sum
                    current_deficit = new_deficit
                    improved = True
                    break
            if improved:
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_indices = sorted(current)
                continue
            
            # Try swapping
            found_swap = False
            for i in list(current):
                if found_swap or time_up():
                    break
                for j in range(n):
                    if j not in current:
                        new_sum = current_sum - fracs[i] + fracs[j]
                        if new_sum < 1:
                            new_deficit = 1 - new_sum
                            if new_deficit < current_deficit:
                                current.remove(i)
                                current.add(j)
                                current_sum = new_sum
                                current_deficit = new_deficit
                                improved = True
                                found_swap = True
                                break
            if improved:
                if current_deficit < best_deficit:
                    best_deficit = current_deficit
                    best_indices = sorted(current)
                continue
            
            # No improvement, break
            break
        
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_indices = sorted(current)
        
        return sorted(current)
    
    # Run initial strategies
    greedy_largest()
    greedy_smallest()
    
    # Local search from best
    if best_indices:
        local_search(best_indices)
    
    # Random restarts
    random.seed(12345)
    restart_count = 0
    while not time_up() and restart_count < 50:
        restart_count += 1
        # Generate random subset
        size = random.randint(3, 25)
        random_indices = random.sample(range(n), size)
        # Filter to valid
        valid = []
        vsum = Fraction(0)
        for i in random_indices:
            if vsum + fracs[i] < 1:
                valid.append(i)
                vsum += fracs[i]
        if len(valid) >= 2:
            local_search(valid, max_iters=5000)
    
    if best_indices is None or len(best_indices) == 0:
        best_indices = [0]  # fallback: just 1/2
    
    return best_indices
# EVOLVE-BLOCK-END
