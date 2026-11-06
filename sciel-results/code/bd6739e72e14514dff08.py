import numpy as np
import time
import random

N = 128
LO, HI = 128, 256
RANGE = HI - LO + 1

def construct():
    start = time.time()
    deadline = start + 55
    
    best_A = None
    best_peak = float('inf')
    
    # Greedy Initialization
    def greedy_init():
        A = set()
        # counts[s - LO] for s in [LO, HI]
        counts = np.zeros(RANGE, dtype=np.int64)
        covered_mask = np.zeros(RANGE, dtype=bool)
        
        # Precompute potential sums for each x
        # sum(x, y) is in [LO, HI] if LO <= x+y <= HI
        # y must be in [max(0, LO-x), min(N, HI-x)]
        
        while True:
            if covered_mask.all():
                break
            best_x = -1
            best_gain = -1
            best_new_counts = None
            best_new_peak = float('inf')
            
            for x in range(N + 1):
                if x in A:
                    continue
                
                # Calculate new counts if x is added
                # New pairs: (x, y) for y in A, and (x, x)
                # We need to check if any count drops to 0? No, we are adding, so counts only increase.
                # We just need to check coverage and peak.
                
                # Efficiently calculate the delta
                # For each y in A:
                #   s = x + y
                #   if LO <= s <= HI:
                #       counts[s-LO] += 1
                # For (x, x):
                #   s = 2*x
                #   if LO <= s <= HI:
                #       counts[2*x-LO] += 1
                
                # Instead of modifying global counts, we can just simulate or track changes.
                # Since N is small, we can just compute the new counts array temporarily.
                temp_counts = counts.copy()
                
                for y in A:
                    s = x + y
                    if LO <= s <= HI:
                        temp_counts[s - LO] += 1
                        
                s = 2 * x
                if LO <= s <= HI:
                    temp_counts[s - LO] += 1
                
                new_covered = np.all(temp_counts > 0)
                new_peak = np.max(temp_counts)
                
                gain = 0
                # Prefer covering new sums
                if not covered_mask.all():
                    # Count how many new sums are covered
                    # Actually, just checking if it helps coverage is good.
                    # Let's use a score: -peak + 100*coverage_progress
                    # Or simply: maximize coverage, then minimize peak.
                    
                    # Number of newly covered sums
                    # We can just check if it reduces the number of uncovered sums
                    # Let's stick to: if it covers more uncovered sums, it's better.
                    # If tie, lower peak is better.
                    
                    # Calculate newly covered
                    # This is expensive to do for every x.
                    # Alternative: Just pick the x that minimizes the resulting peak,
                    # but only among those that don't break coverage (trivial for add)
                    # and prefer those that cover more.
                    
                    pass
                
                # Simple heuristic: Minimize the resulting peak.
                # If peaks are equal, prefer the one that covers more.
                # But calculating coverage gain for every x is O(|A|) anyway.
                
                # Let's just use: Score = -100 * (number of newly covered) - peak
                # To get newly covered:
                # newly_covered = 0
                # for idx in range(RANGE):
                #    if temp_counts[idx] > 0 and counts[idx] == 0:
                #        newly_covered += 1
                
                newly_covered = 0
                for idx in range(RANGE):
                    if temp_counts[idx] > 0 and counts[idx] == 0:
                        newly_covered += 1
                
                score = newly_covered * 1000 - new_peak
                
                if score > best_gain:
                    best_gain = score
                    best_x = x
                    best_new_counts = temp_counts
                    best_new_peak = new_peak
            
            if best_x == -1:
                break
            
            A.add(best_x)
            counts = best_new_counts
            covered_mask = (counts > 0)
            
            # Early stop if covered
            if covered_mask.all():
                break
        
        return A, int(np.max(counts))

    A, current_peak = greedy_init()
    
    if not np.all((np.convolve(np.array([1 if i in A else 0 for i in range(N+1)]), np.array([1 if i in A else 0 for i in range(N+1)]), 'full')[LO:HI+1]) > 0):
        # Fallback if greedy failed (shouldn't happen)
        A = set(range(64, N+1))
        current_peak = 1000

    best_A = set(A)
    best_peak = current_peak
    
    # Incremental Local Search
    # Maintain counts array
    def get_counts(A_set):
        counts = np.zeros(RANGE, dtype=np.int64)
        A_list = list(A_set)
        for i, a in enumerate(A_list):
            # (a, a)
            s = 2 * a
            if LO <= s <= HI:
                counts[s - LO] += 1
            # (a, b) for b > a
            for j in range(i + 1, len(A_list)):
                b = A_list[j]
                s = a + b
                if LO <= s <= HI:
                    counts[s - LO] += 1
        return counts

    counts = get_counts(A)
    
    # Helper to check validity and peak
    def is_valid(counts):
        return np.all(counts > 0)
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_peak = int(np.max(counts))
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_A = set(A)
            
        # 1. Try Removals
        A_list = list(A)
        best_removal = None
        best_removal_peak = current_peak
        
        for a in A_list:
            # Simulate removing a
            new_counts = counts.copy()
            
            # Subtract contributions of a
            # (a, a)
            s = 2 * a
            if LO <= s <= HI:
                new_counts[s - LO] -= 1
                
            # (a, b) for b in A, b != a
            for b in A:
                if b == a:
                    continue
                s = a + b
                if LO <= s <= HI:
                    new_counts[s - LO] -= 1
                    
            if np.all(new_counts > 0):
                p = int(np.max(new_counts))
                if p < best_removal_peak:
                    best_removal_peak = p
                    best_removal = a
                elif p == best_removal_peak and best_removal is None:
                    # Prefer removal if it keeps peak same (reduces set size, might help future)
                    # But we want to minimize peak. If peak is same, removing is neutral for peak.
                    # However, smaller set might allow better swaps later?
                    # Let's only remove if it strictly improves peak or if we are stuck.
                    # Actually, removing an element that doesn't increase peak is safe.
                    # But if peak stays same, we haven't improved the objective.
                    # Let's require strict improvement for removal to avoid cycling.
                    pass
        
        if best_removal is not None:
            A.remove(best_removal)
            # Update counts
            s = 2 * best_removal
            if LO <= s <= HI:
                counts[s - LO] -= 1
            for b in A:
                s = best_removal + b
                if LO <= s <= HI:
                    counts[s - LO] -= 1
            improved = True
            continue
            
        # 2. Try Swaps
        A_list = list(A)
        not_in_A = [i for i in range(N + 1) if i not in A]
        
        best_swap = None
        best_swap_peak = current_peak
        
        # To speed up, we can limit swaps or use random sampling if set is large
        # But N=128, |A|~50, so 50*78 ~ 4000 swaps.
        # Each swap check is O(|A|) ~ 50. Total 200k ops per iteration.
        # 60s / 200k = 300 iterations? Maybe more with numpy.
        
        for a in A_list:
            for b in not_in_A:
                # Simulate swap a -> b
                new_counts = counts.copy()
                
                # Remove a
                s = 2 * a
                if LO <= s <= HI:
                    new_counts[s - LO] -= 1
                for y in A:
                    if y == a: continue
                    s = a + y
                    if LO <= s <= HI:
                        new_counts[s - LO] -= 1
                        
                # Add b
                s = 2 * b
                if LO <= s <= HI:
                    new_counts[s - LO] += 1
                for y in A:
                    if y == a: continue
                    s = b + y
                    if LO <= s <= HI:
                        new_counts[s - LO] += 1
                        
                if np.all(new_counts > 0):
                    p = int(np.max(new_counts))
                    if p < best_swap_peak:
                        best_swap_peak = p
                        best_swap = (a, b)
                        
        if best_swap is not None:
            a, b = best_swap
            A.remove(a)
            A.add(b)
            
            # Update counts
            s = 2 * a
            if LO <= s <= HI:
                counts[s - LO] -= 1
            for y in A:
                s = a + y
                if LO <= s <= HI:
                    counts[s - LO] -= 1
                    
            s = 2 * b
            if LO <= s <= HI:
                counts[s - LO] += 1
            for y in A:
                s = b + y
                if LO <= s <= HI:
                    counts[s - LO] += 1
                    
            improved = True
            continue
            
        # 3. Random Perturbation (Simulated Annealing step)
        # If stuck, try a random swap that maintains coverage, even if peak increases slightly.
        # This helps escape local minima.
        if len(A) > 0 and len(not_in_A) > 0:
            a = random.choice(A_list)
            b = random.choice(not_in_A)
            # Check if swap is valid (coverage)
            new_counts = counts.copy()
            
            s = 2 * a
            if LO <= s <= HI:
                new_counts[s - LO] -= 1
            for y in A:
                if y == a: continue
                s = a + y
                if LO <= s <= HI:
                    new_counts[s - LO] -= 1
                    
            s = 2 * b
            if LO <= s <= HI:
                new_counts[s - LO] += 1
            for y in A:
                if y == a: continue
                s = b + y
                if LO <= s <= HI:
                    new_counts[s - LO] += 1
                    
            if np.all(new_counts > 0):
                # Accept if peak doesn't increase too much?
                # For now, just accept if valid to explore.
                # We should track temperature.
                # Simple: accept if peak <= current_peak + 1
                p = int(np.max(new_counts))
                if p <= current_peak + 1:
                    A.remove(a)
                    A.add(b)
                    
                    s = 2 * a
                    if LO <= s <= HI:
                        counts[s - LO] -= 1
                    for y in A:
                        s = a + y
                        if LO <= s <= HI:
                            counts[s - LO] -= 1
                            
                    s = 2 * b
                    if LO <= s <= HI:
                        counts[s - LO] += 1
                    for y in A:
                        s = b + y
                        if LO <= s <= HI:
                            counts[s - LO] += 1
                    improved = True # Force another iteration
                    continue

    return list(best_A)
