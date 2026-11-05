"""Additive bases with few representations – optimized search."""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 128
    # Target interval: [128, 256]
    # A is a subset of [0, 128]
    # We need every s in [128, 256] to be representable as a+b with a,b in A, a<=b
    
    # Initial solution: A = {64, 65, ..., 128} gives coverage but high multiplicity
    # Let's use a smarter initial solution
    
    # Key insight: if A contains k elements, the number of pair sums is at most k(k+1)/2
    # We need to cover 129 values (128 to 256 inclusive)
    # So we need k(k+1)/2 >= 129, i.e., k >= 16
    
    # Let's try to find a small set with low peak multiplicity
    # Strategy: simulated annealing with vectorized convolution for fast evaluation
    
    def get_pair_sums(A):
        """Return array of all unordered pair sums (including equal) from sorted A."""
        A_arr = np.array(A)
        k = len(A_arr)
        # Use broadcasting for upper triangle including diagonal
        i_idx, j_idx = np.triu_indices(k, k=0)
        return A_arr[i_idx] + A_arr[j_idx]
    
    def compute_counts(A):
        """Compute multiplicity of each sum in [128, 256]."""
        A_arr = np.array(sorted(A))
        k = len(A_arr)
        i_idx, j_idx = np.triu_indices(k, k=0)
        sums = A_arr[i_idx] + A_arr[j_idx]
        # Count occurrences in [128, 256]
        mask = (sums >= 128) & (sums <= 256)
        relevant_sums = sums[mask]
        # Use np.bincount for efficiency
        counts = np.bincount(relevant_sums - 128, minlength=129)
        return counts
    
    def peak_multiplicity(A):
        """Return the maximum representation count in [128, 256]."""
        counts = compute_counts(A)
        return int(counts.max())
    
    def is_valid(A):
        """Check that every value in [128, 256] is covered."""
        counts = compute_counts(A)
        return bool(np.all(counts > 0))
    
    def objective(A):
        """Combined objective: peak multiplicity + penalty for uncovered values."""
        counts = compute_counts(A)
        uncovered = int(np.sum(counts == 0))
        peak = int(counts.max())
        # Penalize uncovered heavily
        return peak + 100 * uncovered
    
    # Random initialization
    def random_init(k):
        """Generate a random subset of [0,128] of size k."""
        return sorted(random.sample(range(0, 129), k))
    
    # Repair: ensure coverage by adding elements
    def repair(A):
        """Add elements to A to ensure full coverage of [128, 256]."""
        A = sorted(set(A))
        for _ in range(200):  # Max iterations
            counts = compute_counts(A)
            uncovered = np.where(counts == 0)[0] + 128
            if len(uncovered) == 0:
                break
            # Pick a random uncovered value and try to add an element that helps
            target = random.choice(list(uncovered))
            # Try adding a random element from [0, 128]
            for _ in range(50):
                candidate = random.randint(0, 128)
                if candidate not in A:
                    new_A = A + [candidate]
                    new_counts = compute_counts(new_A)
                    new_uncovered = int(np.sum(new_counts == 0))
                    old_uncovered = int(np.sum(counts == 0))
                    if new_uncovered < old_uncovered:
                        A = new_A
                        break
        return A
    
    # Local search: try swapping elements
    def local_search(A, max_iters=500):
        """Improve A by trying single-element swaps and additions/removals."""
        A = sorted(set(A))
        best_obj = objective(A)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding a random element
            for _ in range(10):
                candidate = random.randint(0, 128)
                if candidate not in A:
                    new_A = A + [candidate]
                    new_obj = objective(new_A)
                    if new_obj < best_obj:
                        A = new_A
                        best_obj = new_obj
                        improved = True
                        break
            if improved:
                continue
            # Try removing an element (only if we have more than minimum needed)
            if len(A) > 16:
                for idx in random.sample(range(len(A)), min(10, len(A))):
                    new_A = A[:idx] + A[idx+1:]
                    new_obj = objective(new_A)
                    if new_obj < best_obj:
                        A = new_A
                        best_obj = new_obj
                        improved = True
                        break
            if improved:
                continue
            # Try swapping
            if len(A) >= 2:
                for _ in range(10):
                    idx = random.randint(0, len(A) - 1)
                    candidate = random.randint(0, 128)
                    if candidate not in A:
                        new_A = A[:idx] + [candidate] + A[idx+1:]
                        new_obj = objective(new_A)
                        if new_obj < best_obj:
                            A = new_A
                            best_obj = new_obj
                            improved = True
                            break
        return A
    
    # Simulated annealing
    def simulated_annealing(A, time_limit):
        """SA with vectorized evaluation."""
        A = sorted(set(A))
        best_A = A[:]
        best_obj = objective(A)
        current_obj = best_obj
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Generate neighbor
            move_type = random.random()
            if move_type < 0.3 and len(A) > 16:
                # Remove an element
                idx = random.randint(0, len(A) - 1)
                new_A = A[:idx] + A[idx+1:]
            elif move_type < 0.6:
                # Add an element
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A + [candidate]
            else:
                # Swap
                if len(A) < 1:
                    continue
                idx = random.randint(0, len(A) - 1)
                candidate = random.randint(0, 128)
                if candidate in A:
                    continue
                new_A = A[:idx] + [candidate] + A[idx+1:]
            
            new_obj = objective(new_A)
            delta = new_obj - current_obj
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                A = new_A
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = A[:]
            
            T *= cooling
        
        return best_A, best_obj
    
    # Run multiple SA restarts
    best_A = None
    best_score = float('inf')
    
    # Try different sizes
    for k in range(16, 30):
        if time.time() > deadline - 15:
            break
        
        # Multiple restarts for this size
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            A = random_init(k)
            A = repair(A)
            if not is_valid(A):
                continue
            
            # Local search
            A = local_search(A, max_iters=200)
            obj = objective(A)
            
            # Simulated annealing
            sa_time = min(8, deadline - time.time() - 5)
            if sa_time > 2:
                A, obj = simulated_annealing(A, time.time() + sa_time)
            
            if obj < best_score:
                best_score = obj
                best_A = A[:]
    
    # Final local search on best
    if best_A is not None and time.time() < deadline - 5:
        best_A = local_search(best_A, max_iters=500)
        final_obj = objective(best_A)
        if final_obj < best_score:
            best_score = final_obj
    
    # If we still don't have a valid solution, use the seed
    if best_A is None or not is_valid(best_A):
        return list(range(64, 129))
    
    return sorted(best_A)
