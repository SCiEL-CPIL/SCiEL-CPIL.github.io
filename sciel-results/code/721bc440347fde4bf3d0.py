import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 128
    targets = list(range(128, 257))
    target_set = set(targets)
    
    def calculate_multiplicities(A):
        """
        Returns a dictionary mapping sum to count of unordered pairs (a,b) with a<=b, a,b in A.
        """
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if 128 <= s <= 256:
                    counts[s] += 1
        return counts

    def get_peak_and_uncovered(A):
        counts = calculate_multiplicities(A)
        peak = 0
        uncovered = []
        for k in targets:
            c = counts[k]
            if c == 0:
                uncovered.append(k)
            if c > peak:
                peak = c
        return peak, uncovered, counts

    # 1. Greedy Initialization to ensure coverage
    A = set()
    uncovered = set(targets)
    
    # We need to cover [128, 256].
    # A simple greedy: pick x that maximizes new coverage.
    while uncovered:
        best_x = -1
        best_new_cov = 0
        # To speed up, we can sample or iterate all. 129 is small.
        for x in range(n + 1):
            if x in A:
                continue
            # How many uncovered targets does x help cover?
            # A target k is covered by x if there exists y in A such that x+y=k.
            # So k is covered if k-x in A.
            # New coverage = {k in uncovered | k-x in A}
            # Note: if x is added, pairs are (x, y) for y in A, and (x, x).
            # So if x is added, it creates sums x+y for y in A (including x).
            # Wait, the definition is sum of two elements of A.
            # If we add x to current A, the new pairs are {x, a} for a in A.
            # The sums are x+a.
            # We check if these sums are in `uncovered`.
            
            # Optimization: 
            # new_sums = {x + a for a in A} U {2x}
            # intersection with uncovered
            new_sums = set()
            for a in A:
                new_sums.add(x + a)
            new_sums.add(2*x)
            
            cov = len(new_sums & uncovered)
            if cov > best_new_cov:
                best_new_cov = cov
                best_x = x
        
        if best_x == -1 or best_new_cov == 0:
            # Should not happen if we start with empty set and range is valid, 
            # but if stuck, break to avoid infinite loop
            # Actually, if A is empty, adding any x covers 2x.
            # If 2x is not in targets, it covers 0.
            # We need to be careful. 
            # If best_new_cov is 0, we might need to add an element that doesn't cover immediately 
            # but enables future coverage? 
            # For this specific range [128, 256] and A subset [0, 128], 
            # greedy usually works. If stuck, we pick a random x.
            if A:
                 break
            else:
                 best_x = 128 # 256 is a target
                 
        A.add(best_x)
        # Update uncovered
        new_sums = set()
        for a in A:
            new_sums.add(best_x + a)
        new_sums.add(2*best_x)
        uncovered -= new_sums

    # Ensure A is not empty
    if not A:
        return [128]

    # 2. Local Search
    current_peak, _, current_counts = get_peak_and_uncovered(A)
    best_peak = current_peak
    best_A = set(A)
    
    # Simulated Annealing / Random Restart Local Search
    # We will try to reduce peak.
    
    def score(A_set):
        peak, unc, counts = get_peak_and_uncovered(A_set)
        if unc:
            return float('inf'), peak, counts
        # Secondary objective: sum of squares of counts (to flatten distribution)
        ss = np.sum(counts[128:257]**2)
        return peak, ss, counts

    curr_score, curr_peak, curr_counts = score(A)
    
    # Time loop
    while time.time() - start_time < time_limit:
        # Try to improve
        improved = False
        
        # 1. Try removing elements
        # If we can remove an element and still cover all targets, and peak doesn't increase, do it.
        # Removing an element usually reduces counts, so peak might decrease or stay same.
        # It might break coverage.
        
        # Randomize order of A for diversity
        A_list = list(A)
        random.shuffle(A_list)
        
        for x in A_list:
            A_trial = set(A)
            A_trial.remove(x)
            if not A_trial:
                continue
            trial_peak, trial_ss, trial_counts = score(A_trial)
            
            if trial_peak < float('inf'): # Coverage maintained
                if trial_peak < curr_peak:
                    A = A_trial
                    curr_peak = trial_peak
                    curr_score = trial_peak
                    curr_counts = trial_counts
                    improved = True
                    break
                elif trial_peak == curr_peak and trial_ss < curr_score[1] if isinstance(curr_score, tuple) else False:
                    # We need consistent scoring. Let's use tuple (peak, ss)
                    pass # Handled below
        
        # Let's structure the scoring properly in the loop
        # Re-evaluate current state variables
        curr_peak, curr_ss, curr_counts = score(A)
        
        # Try Swaps
        # Pick a random x in A and a random y not in A
        if len(A) > 1: # Keep at least one element
            x = random.choice(list(A))
            candidates_out = [i for i in range(n+1) if i not in A]
            if candidates_out:
                y = random.choice(candidates_out)
                A_trial = set(A)
                A_trial.remove(x)
                A_trial.add(y)
                
                trial_peak, trial_ss, trial_counts = score(A_trial)
                
                if trial_peak < float('inf'):
                    if trial_peak < curr_peak:
                        A = A_trial
                        curr_peak = trial_peak
                        curr_ss = trial_ss
                        curr_counts = trial_counts
                        improved = True
                    elif trial_peak == curr_peak and trial_ss < curr_ss:
                        A = A_trial
                        curr_ss = trial_ss
                        curr_counts = trial_counts
                        improved = True
                    # Accept worse peak with small probability? No, we want min peak.
                    # Accept same peak but better ss? Yes.
        
        # Try Adding? 
        # Adding increases counts, so peak likely increases. 
        # Only useful if it allows a subsequent removal that lowers peak? 
        # That's a 2-step move. Too complex for simple loop.
        # However, if we are stuck at a local minimum of peak, 
        # maybe we need to add an element to "bridge" a gap? 
        # But coverage is already full.
        # So adding is generally bad for peak minimization unless it enables a swap that is better.
        # Swaps cover the "replace" case.

        if not improved:
            # Maybe do a random perturbation?
            # Swap 2 elements?
            if len(A) > 2:
                x1, x2 = random.sample(list(A), 2)
                c1, c2 = random.sample([i for i in range(n+1) if i not in A], 2)
                A_trial = set(A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(c1)
                A_trial.add(c2)
                
                trial_peak, trial_ss, trial_counts = score(A_trial)
                if trial_peak < float('inf'):
                    if trial_peak < curr_peak or (trial_peak == curr_peak and trial_ss < curr_ss):
                        A = A_trial
                        curr_peak = trial_peak
                        curr_ss = trial_ss
                        curr_counts = trial_counts
                        improved = True

        # Update best
        if curr_peak < best_peak:
            best_peak = curr_peak
            best_A = set(A)

    # Return the best found
    return sorted(list(best_A))
