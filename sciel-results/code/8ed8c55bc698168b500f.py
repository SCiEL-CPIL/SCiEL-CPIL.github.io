import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    def check_sidon(subset):
        """Check if subset is a Sidon set including diagonal sums."""
        n = len(subset)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def max_sidon_size(ambient):
        """Find the maximum size of a Sidon subset in ambient."""
        n = len(ambient)
        # Check from 6 down to 3
        # If we find a Sidon subset of size k, return k
        for k in range(6, 2, -1):
            for subset in itertools.combinations(ambient, k):
                if check_sidon(subset):
                    return k
        # If no subset of size 3 is Sidon, return 2
        # But we established 2 is likely impossible for 12 distinct ints
        for subset in itertools.combinations(ambient, 3):
            if check_sidon(subset):
                return 3
        return 2

    # Initial candidate: Dense set [0, 11]
    # This is likely to have small max Sidon size
    current = list(range(12))
    current_size = max_sidon_size(current)
    
    best_ambient = list(current)
    best_size = current_size
    
    rng = random.Random(42)
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        i = rng.randrange(12)
        old_val = current[i]
        
        # Random new value
        # Bias towards small values to maintain density?
        # Or random in [0, 128]?
        # Dense sets are "closer" to non-Sidon.
        # Let's try random in [0, 50]
        new_val = rng.randrange(51)
        
        if new_val == old_val:
            continue
            
        if new_val in current:
            continue
            
        current[i] = new_val
        
        new_size = max_sidon_size(current)
        
        if new_size < best_size:
            best_size = new_size
            best_ambient = list(current)
            # Reset temperature?
            T = 100.0
        elif new_size == best_size:
            # Tie-breaker: keep current if it's not worse
            # We can also try to minimize the number of Sidon subsets of size best_size
            # But for now, just accept if it's the same size?
            # Actually, if size is same, we might want to explore more.
            # Accept with probability 1 if same size?
            pass
        else:
            # new_size > best_size
            # Accept with probability exp(-(new_size - current_size) / T)
            # Since sizes are integers, delta is at least 1.
            delta = new_size - current_size
            if delta > 0:
                prob = pow(2.71828, -delta / T)
                if rng.random() > prob:
                    # Revert
                    current[i] = old_val
            else:
                # Should not happen if we check correctly
                pass
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_ambient
