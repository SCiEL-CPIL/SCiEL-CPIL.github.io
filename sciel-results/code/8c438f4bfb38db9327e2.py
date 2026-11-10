import time
import random
from itertools import combinations

def _max_sidon_size(A):
    """
    Exhaustively computes the largest subset of A with all unordered pair sums unique.
    A is a list of distinct integers.
    """
    n = len(A)
    if n == 0:
        return 0
    
    # Sort for deterministic behavior and potential pruning
    A = sorted(A)
    
    best_size = 0
    # Check subsets in decreasing size order to find max quickly?
    # Or just check all and track max.
    # For n=12, 2^12 = 4096 subsets. Very fast.
    
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(A, size):
            # Check if subset is Sidon
            sums = set()
            is_sidon = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                best_size = size
                found = True
                break
        if found:
            break
            
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial candidate: Arithmetic Progression
    current_best = list(range(12))
    current_best_val = _max_sidon_size(current_best)
    
    # Local Search
    # We want to minimize the max Sidon subset size.
    # Start with dense sets.
    
    # Define a set of "dense" starting points or just mutate the AP.
    # Mutate one element at a time.
    
    # To ensure diversity, we can try a few random restarts if time permits.
    # But hill climbing from AP is a good start.
    
    # Helper to generate a random neighbor
    def get_neighbor(A):
        A = list(A)
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        # Choose a new value, preferably close to keep density
        # But allow some spread
        # Range [0, 128]
        # Try values in [0, 128]
        # To keep it dense, try [max(0, old_val-10), min(128, old_val+10)]
        # But sometimes we need to jump out of local minima.
        
        # Strategy: 80% local move, 20% global move
        if random.random() < 0.8:
            new_val = random.randint(max(0, old_val - 15), min(128, old_val + 15))
        else:
            new_val = random.randint(0, 128)
            
        if new_val in A and new_val != old_val:
            # Ensure distinctness
            # If collision, retry or pick another
            # Simple retry
            for _ in range(10):
                new_val = random.randint(0, 128)
                if new_val not in A:
                    break
            else:
                return A # Failed to generate valid neighbor
        
        A[idx] = new_val
        return A

    # Hill Climb
    while time.time() < deadline:
        neighbor = get_neighbor(current_best)
        if neighbor == current_best:
            continue
            
        val = _max_sidon_size(neighbor)
        
        if val < current_best_val:
            current_best = neighbor
            current_best_val = val
            if current_best_val == 3: # Theoretical min?
                break
            if current_best_val == 2:
                break
        # Optional: Accept worse moves with small probability (Simulated Annealing)
        # But for now, strict hill climbing is safer for "minimize"
        
        # If we get stuck, maybe try a random restart?
        # Given the small search space of "dense" sets, hill climbing should work.
        
    return current_best
