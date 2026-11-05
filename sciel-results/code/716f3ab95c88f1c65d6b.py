import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a subset A of [0, 128] such that every integer in [128, 256] is a sum
    of two elements of A (unordered pairs, equal summands allowed), minimizing the
    largest representation multiplicity in that target interval.
    
    Strategy:
    1. Start with a heuristic initial solution (e.g., symmetric or structured set).
    2. Use local search (simulated annealing or greedy hill-climbing) to minimize
       the peak multiplicity.
    3. Ensure coverage: every sum in [128, 256] must be representable.
    
    Since the problem is small (n=129 elements, target sums 129 values), we can use
    efficient representation and fast evaluation.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    target_lo, target_hi = 128, 256
    
    # Represent A as a boolean array of size 129
    # A[i] = 1 if i is in A
    
    def count_representations(A):
        """
        For each target sum s in [128, 256], count the number of unordered pairs
        (a, b) with a <= b, a, b in A, such that a + b = s.
        Returns a dict or array of counts, and the max count.
        """
        # Use convolution-like approach or direct counting
        # Since n is small, direct counting is fine but let's be smart.
        # For each a in A, for each b in A with b >= a, if a+b in [128,256], increment count[a+b]
        
        counts = np.zeros(target_hi - target_lo + 1, dtype=np.int32)
        A_list = np.where(A)[0]
        for i in range(len(A_list)):
            a = A_list[i]
            # b must be >= a and in A
            # We can iterate over A_list from i onwards
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    counts[s - target_lo] += 1
        return counts
    
    def evaluate(A):
        """
        Evaluate the objective: max multiplicity among target sums, 
        but also need to ensure all are covered (count > 0).
        If any sum is not covered, return a large penalty.
        """
        counts = count_representations(A)
        if np.any(counts == 0):
            # Not all sums covered
            uncovered = np.sum(counts == 0)
            return 10000 + uncovered  # Penalty
        return int(np.max(counts))
    
    def initial_solution():
        """
        Create an initial solution.
        A good start: take a structured set that likely covers the range.
        For example, take every other number in a range, or a symmetric set.
        
        Let's try: A = [0, 1, ..., 63] ∪ [65, 66, ..., 128] ? 
        Or simpler: A = range(0, 129, 2) ∪ {128}? 
        Actually, the seed is range(64, 129). Let's try something denser but structured.
        
        A good heuristic: take a set that is "balanced" to spread out the sums.
        Let's try: A = [i for i in range(0, 129) if i % 3 != 0] ? 
        Or use a greedy approach to build A.
        
        Let's start with the seed and perturb it.
        Seed: A = range(64, 129)
        """
        A = np.zeros(n + 1, dtype=np.int8)
        # Start with seed: 64..128
        A[64:129] = 1
        # Add some small numbers to help cover lower sums? 
        # Sums in [128, 256]. If we have 64..128, min sum is 128, max is 256.
        # So the seed already covers the range in terms of min/max possible sums.
        # But are all sums covered? 
        # 128 = 64+64
        # 129 = 64+65
        # ...
        # 192 = 64+128
        # 193 = 65+128
        # ...
        # 256 = 128+128
        # So the seed covers all sums in [128, 256].
        # What is the max multiplicity?
        # For sum s, pairs (a,b) with a<=b, a,b in [64,128], a+b=s.
        # The number of pairs is maximized when s is in the middle.
        # For s=192, a can range from 64 to 128, b=s-a.
        # a in [64, 128], b in [64, 128] => s-a in [64, 128] => a in [s-128, s-64].
        # For s=192: a in [64, 128]. All 65 values? 
        # a=64, b=128; a=65, b=127; ... a=128, b=64.
        # Since a<=b, we only count a <= s/2.
        # For s=192, s/2=96. a in [64, 96]. That's 33 values.
        # So max multiplicity is around 33.
        
        # Can we do better? 
        # If we spread out A more, we might reduce the peak.
        # Let's return the seed as initial.
        return A
    
    # Local search: flip bits in A to improve the objective.
    def local_search(A, max_time=10):
        """
        Hill climbing with random restarts or simulated annealing.
        We flip one bit at a time (add or remove an element) and accept if it improves
        or doesn't worsen too much (simulated annealing).
        """
        current = A.copy()
        current_score = evaluate(current)
        best = current.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        max_iter = 100000
        
        for it in range(max_iter):
            if time.time() > max_time + start_time:
                break
            
            # Choose a random index to flip
            idx = random.randint(0, n)
            new_A = current.copy()
            new_A[idx] = 1 - new_A[idx]
            
            new_score = evaluate(new_A)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current = new_A
                current_score = new_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            
            T *= alpha
            if T < T_min:
                break
        
        return best, best_score
    
    # Run local search for a few iterations with different initializations
    best_A = None
    best_score = float('inf')
    
    # Initial solution
    A_init = initial_solution()
    A_best, score_best = local_search(A_init, max_time=20)
    best_A = A_best
    best_score = score_best
    
    # Try a few more initializations
    for i in range(5):
        if time.time() > deadline:
            break
        
        # Perturb the initial solution
        A_perturbed = A_init.copy()
        # Randomly flip 10-20 bits
        num_flips = random.randint(10, 20)
        for _ in range(num_flips):
            idx = random.randint(0, n)
            A_perturbed[idx] = 1 - A_perturbed[idx]
        
        A_candidate, score_candidate = local_search(A_perturbed, max_time=10)
        if score_candidate < best_score:
            best_A = A_candidate
            best_score = score_candidate
    
    # Convert best_A to a list of indices
    result = [i for i in range(n + 1) if best_A[i] == 1]
    
    # Ensure non-empty
    if not result:
        result = [64]  # Fallback
    
    return result

# EVOLVE-BLOCK-END
