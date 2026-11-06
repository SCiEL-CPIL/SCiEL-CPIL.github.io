import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute for each x in [0, 128] the list of targets s in [128, 256] 
    # such that there exists y in [0, 128] with x+y=s.
    # Actually, for a specific x, the contribution to counts depends on which y are in A.
    # However, for the "column generation" heuristic, we can precompute the "potential" 
    # of each x to cover specific sums.
    
    # Let's represent A as a boolean array of size 129.
    # counts[s] = sum_{x,y in A, x<=y} 1[x+y=s]
    
    # To speed up count updates, we can use convolution?
    # No, simple loops are fine for small A, but we might have up to 129 elements.
    # 129 * 129 / 2 ~ 8000 ops per full recalc. Very fast.
    # But incremental updates are better.
    
    def get_counts_from_A(A_bool):
        """
        Calculate counts for all sums in [128, 256] given a boolean array A_bool.
        Returns numpy array of size 129 (index 0 corresponds to sum 128).
        """
        counts = np.zeros(num_targets, dtype=np.int32)
        A_indices = np.where(A_bool)[0]
        k = len(A_indices)
        for i in range(k):
            x = A_indices[i]
            # y must be >= x to avoid double counting (unordered pairs)
            # s = x + y. We need s in [128, 256].
            # y in [x, 128]
            # s_min = x + x = 2x
            # s_max = x + 128
            
            # Determine valid y range for this x
            # y >= x
            # y >= 128 - x  (so that x+y >= 128)
            # y <= 256 - x  (so that x+y <= 256, always true since y<=128, x>=0 => s<=256)
            
            y_min = max(x, 128 - x)
            y_max = 128
            
            if y_min > y_max:
                continue
                
            # We need to count how many y in [y_min, y_max] are in A
            # and add 1 to counts[s] for each such y.
            # s = x + y => index = s - 128
            
            # Vectorized check for y in A
            y_range = np.arange(y_min, y_max + 1)
            valid_y = A_bool[y_range]
            
            if np.any(valid_y):
                # Get the indices of valid y
                valid_y_indices = y_range[valid_y]
                # Calculate sums
                sums = x + valid_y_indices
                # Convert to indices in counts array
                indices = sums - target_lo
                # Increment counts
                np.add.at(counts, indices, 1)
                
        return counts

    def get_objective(counts, A_size):
        """
        Objective: Minimize peak count, with secondary objective to minimize variance.
        The ideal count for each target is roughly (A_size * (A_size+1) / 2) / num_targets.
        We want to flatten the distribution.
        """
        if np.min(counts) == 0:
            return float('inf')
            
        peak = np.max(counts)
        # Variance-aware penalty to encourage balance
        # This helps escape local optima where peak is low but distribution is jagged,
        # which might prevent further improvement or lead to higher peaks in other configs.
        variance = np.var(counts)
        # Weighting: Peak is primary, variance is secondary.
        # If peak is same, lower variance is better.
        # We can combine them: peak * 100 + variance
        return peak * 100.0 + variance

    def local_search(A_bool, counts, time_limit):
        """
        Perform local search: try to add, remove, or swap elements to improve objective.
        """
        current_obj = get_objective(counts, np.sum(A_bool))
        A_indices = np.where(A_bool)[0]
        
        while time.time() < time_limit:
            improved = False
            # Shuffle candidates to avoid deterministic loops
            candidates = list(range(n + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > time_limit:
                    break
                    
                # Try adding x if not in A
                if not A_bool[x]:
                    # Calculate delta counts
                    # New pairs: (x, y) for y in A, y >= x, and (x, x)
                    # Wait, if we add x, we add pairs (x, y) for all y in A with y >= x?
                    # No, the set of pairs changes.
                    # Previously: Pairs among A.
                    # Now: Pairs among A U {x}.
                    # New pairs involving x: (x, y) for y in A, y >= x. Plus (x, x).
                    # Note: If y < x, the pair (y, x) was already counted as (y, x) in the old set?
                    # No, in get_counts_from_A, we iterate i (x) and y >= x.
                    # So if we add x, we need to add contributions where x is the smaller element.
                    # i.e., y in A such that y >= x. And x itself.
                    
                    # Let's just recalculate counts for simplicity in the check?
                    # A size is small (< 100 usually). Recalc is ~ O(N^2) ~ 10k ops.
                    # Very fast. Let's do that to avoid bugs.
                    
                    A_bool[x] = True
                    new_counts = get_counts_from_A(A_bool)
                    new_obj = get_objective(new_counts, np.sum(A_bool))
                    
                    if new_obj < current_obj:
                        current_obj = new_obj
                        counts = new_counts
                        improved = True
                    else:
                        A_bool[x] = False
                        
                # Try removing x if in A
                elif A_bool[x]:
                    # Only remove if A size is large enough (don't want to drop below min size for coverage)
                    # Min size? To cover 129 sums, we need at least sqrt(2*129) ~ 16 elements?
                    # Actually, with k elements, max pairs is k(k+1)/2.
                    # If k=20, pairs=210 > 129. So k>=20 is safe for coverage potential.
                    # Let's allow removal if size > 25.
                    if np.sum(A_bool) > 25:
                        A_bool[x] = False
                        new_counts = get_counts_from_A(A_bool)
                        new_obj = get_objective(new_counts, np.sum(A_bool))
                        
                        if new_obj < current_obj:
                            current_obj = new_obj
                            counts = new_counts
                            improved = True
                        else:
                            A_bool[x] = True

            if not improved:
                # Try swaps? (Add x, remove y)
                # This is more expensive. Let's stick to add/remove for now.
                # If stuck, break.
                break
                
        return A_bool, counts, current_obj

    # Main strategy: Greedy Column Generation + Local Search
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Greedy Construction
    # Start with empty set.
    # Iteratively pick x that minimizes the "slack" of the most saturated sum,
    # or maximizes coverage of empty sums, balancing the load.
    
    # We will run a few random restarts of the greedy process.
    
    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        random.seed(restart * 123 + 456)
        
        A_bool = np.zeros(n + 1, dtype=bool)
        
        # Initialize with a few random elements to break symmetry?
        # Or start empty and let greedy pick.
        # Starting empty might get stuck in bad local minima.
        # Let's seed with a random subset of size 10-20.
        initial_size = random.randint(10, 20)
        initial_elements = random.sample(range(n + 1), initial_size)
        for x in initial_elements:
            A_bool[x] = True
            
        counts = get_counts_from_A(A_bool)
        
        # Greedy Phase
        max_greedy_steps = 100
        for step in range(max_greedy_steps):
            if time.time() > deadline - 10:
                break
                
            # Check if all covered
            if np.min(counts) > 0:
                break
                
            # Find best x to add
            best_x = -1
            best_new_obj = float('inf')
            
            # Candidate x's: those not in A
            candidates = [x for x in range(n + 1) if not A_bool[x]]
            if not candidates:
                break
                
            # To speed up, we can sample candidates if A is large?
            # No, 129 is small. Evaluate all.
            
            for x in candidates:
                A_bool[x] = True
                new_counts = get_counts_from_A(A_bool)
                
                # Objective for greedy step:
                # 1. If not all covered, prioritize covering new sums.
                # 2. If all covered (or nearly), minimize peak/variance.
                
                if np.min(new_counts) == 0:
                    # Count how many new sums are covered
                    newly_covered = np.sum((new_counts > 0) & (counts == 0))
                    # Penalize peak
                    peak = np.max(new_counts)
                    # Score: Maximize newly covered, minimize peak
                    # We want to minimize a combined score.
                    # score = -newly_covered * 1000 + peak * 10
                    score = -newly_covered * 1000.0 + peak * 10.0
                else:
                    # All covered, use standard objective
                    score = get_objective(new_counts, np.sum(A_bool))
                    
                if score < best_new_obj:
                    best_new_obj = score
                    best_x = x
                    # Store best counts to avoid recalculation if we pick it?
                    # We'll recalc after loop.
                    
                A_bool[x] = False
                
            if best_x == -1:
                break
                
            # Commit best_x
            A_bool[best_x] = True
            counts = get_counts_from_A(A_bool)
            
        # Local Search Phase
        A_bool, counts, obj = local_search(A_bool, counts, deadline - 5)
        
        # Update best
        if obj < best_score:
            best_score = obj
            best_A = A_bool.copy()
            
            # Check if we hit a very good score
            # Theoretical lower bound for peak:
            # If |A|=k, pairs = k(k+1)/2. Avg = pairs/129.
            # If k=30, avg=3.6, peak>=4.
            # If k=25, avg=2.5, peak>=3.
            # If k=20, avg=1.6, peak>=2.
            # Can we get peak 2?
            # Need 129 sums covered by ~20 elements with max multiplicity 2.
            # Total pairs = 210. 210/129 = 1.62.
            # If peak is 2, sum of counts <= 129*2 = 258.
            # 210 <= 258. Possible.
            # If peak is 3, sum <= 387.
            
            if best_score <= 300: # Peak 3, low variance
                 # We might stop early if we find a very good solution
                 # But let's give it a bit more time to try for peak 2?
                 # Peak 2 is very hard.
                 pass

    # Fallback
    if best_A is None:
        # Use the current best from the last restart
        A_bool = np.zeros(n + 1, dtype=bool)
        # Just pick a large set to ensure coverage
        A_bool[:64] = True
        best_A = A_bool

    # Convert to list
    result_list = [int(x) for x in np.where(best_A)[0]]
    
    # Final validation
    # Ensure all targets are covered
    final_counts = get_counts_from_A(best_A)
    if np.min(final_counts) == 0:
        # Emergency fix: add elements until covered
        for x in range(n + 1):
            if not best_A[x]:
                best_A[x] = True
                final_counts = get_counts_from_A(best_A)
                if np.min(final_counts) > 0:
                    break
        result_list = [int(x) for x in np.where(best_A)[0]]
        
    return result_list

if __name__ == "__main__":
    A = construct()
    print(A)
