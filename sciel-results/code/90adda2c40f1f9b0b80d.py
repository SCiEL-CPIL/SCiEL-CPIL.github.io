"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _peak_multiplicity(A):
    """Compute the peak representation multiplicity for sums in [128, 256]."""
    A = sorted(set(A))
    n = len(A)
    # Count representations for each sum in [128, 256]
    # For unordered pairs (i,j) with i <= j, sum = A[i] + A[j]
    # We need to count how many such pairs give each sum
    # Efficient approach: use convolution or direct counting
    # Since n can be up to 129, direct O(n^2) is fine
    rep_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            if 128 <= s <= 256:
                rep_counts[s] = rep_counts.get(s, 0) + 1
    
    # Check coverage
    for s in range(128, 257):
        if s not in rep_counts or rep_counts[s] == 0:
            return float('inf')
    
    return max(rep_counts.values())

def _random_initial_solution():
    """Generate a random initial solution."""
    # Start with a reasonable base: we need sums covering [128, 256]
    # A subset of [0, 128]
    # Strategy: include some structure to ensure coverage
    # Minimum size needed: if A has k elements, max sum is 2*128=256, min sum from A is 0
    # To cover [128,256], we need pairs summing to each value in that range
    
    # A good heuristic: use a "half" set
    # If A = [a, a+1, ..., b], then sums range from 2a to 2b
    # To cover [128, 256], need 2a <= 128 and 2b >= 256, so a <= 64 and b >= 128
    # But b <= 128, so b = 128, and a <= 64
    # For A = [a, ..., 128], sums go from 2a to 256
    # Need 2a <= 128, so a <= 64
    # Number of elements: 128 - a + 1
    # For a=64: A=[64,...,128], 65 elements, sums [128,256]
    # This is the seed solution
    
    # Let's try random subsets with some structure
    # Approach: start with a base and perturb
    base_start = random.randint(40, 70)
    A = set(range(base_start, 129))
    
    # Randomly remove some elements
    to_remove = random.sample(list(A), random.randint(0, 10))
    for x in to_remove:
        A.discard(x)
    
    # Ensure coverage by adding back if needed
    # Check if all sums in [128,256] are covered
    peak = _peak_multiplicity(list(A))
    if peak == float('inf'):
        # Add back removed elements
        for x in to_remove:
            A.add(x)
    
    # Also try adding some small elements
    small_adds = random.sample(range(0, base_start), random.randint(0, 5))
    for x in small_adds:
        A.add(x)
    
    return list(A)

def _local_search(A, max_iters=5000, deadline=None):
    """Perform local search to minimize peak multiplicity."""
    A = set(A)
    current_peak = _peak_multiplicity(list(A))
    best_A = set(A)
    best_peak = current_peak
    
    for iteration in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try a random move: add, remove, or swap an element
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add' and len(A) < 129:
            candidates = [x for x in range(129) if x not in A]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_A = A | {x}
        elif move_type == 'remove' and len(A) > 1:
            x = random.choice(list(A))
            new_A = A - {x}
        elif move_type == 'swap' and len(A) > 1:
            x = random.choice(list(A))
            candidates = [y for y in range(129) if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            new_A = (A - {x}) | {y}
        else:
            continue
        
        new_peak = _peak_multiplicity(list(new_A))
        
        if new_peak < current_peak:
            A = new_A
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(A)
        elif new_peak == current_peak and random.random() < 0.3:
            # Accept equal moves to escape local minima
            A = new_A
    
    return list(best_A), best_peak

def construct():
    """Construct a subset A of [0,128] minimizing peak representation multiplicity."""
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, leaving 10s margin
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Start with the seed solution
            A = list(range(64, 129))
        elif restart == 1:
            # Start with a slightly different base
            A = list(range(60, 129))
        elif restart == 2:
            # Random initial
            A = _random_initial_solution()
        elif restart == 3:
            # Another random
            A = _random_initial_solution()
        else:
            # Symmetric-ish design
            A = list(range(50, 129))
        
        A, peak = _local_search(A, max_iters=3000, deadline=deadline)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    return best_A
# EVOLVE-BLOCK-END
