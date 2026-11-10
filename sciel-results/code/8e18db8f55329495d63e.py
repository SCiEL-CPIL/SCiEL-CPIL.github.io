import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 128
    targets = list(range(128, 257))
    
    # Precompute valid pairs for each target sum? 
    # Or just compute on the fly. 
    # Since |A| is small, on-the-fly is fine.
    
    def compute_counts(A_set):
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 256:
                    counts[s] += 1
        return counts

    def get_metrics(counts):
        c = counts[128:257]
        peak = np.max(c)
        sq = np.sum(c**2)
        covered = np.all(c > 0)
        return peak, sq, covered

    def greedy_init():
        A = set()
        # Start with some random elements to break symmetry
        for _ in range(10):
            A.add(random.randint(0, n))
        
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        while not covered:
            # Find uncovered targets
            uncovered = [t for t in targets if counts[t] == 0]
            if not uncovered:
                break
            
            best_x = -1
            best_gain = -1
            best_new_counts = None
            
            # Try adding each x not in A
            for x in range(n + 1):
                if x in A:
                    continue
                # Estimate gain
                # New sums: x + a for a in A, and 2x
                new_sums = set()
                for a in A:
                    s = x + a
                    if 128 <= s <= 256:
                        new_sums.add(s)
                s = 2 * x
                if 128 <= s <= 256:
                    new_sums.add(s)
                
                gain = len([s for s in new_sums if counts[s] == 0])
                # Prefer gains that also don't increase peak too much?
                # For now, just max coverage gain.
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_new_counts = None # Recalculate if needed, or just track x
                    
            if best_x == -1 or best_gain == 0:
                # Stuck, break
                break
            
            A.add(best_x)
            counts = compute_counts(A)
            peak, sq, covered = get_metrics(counts)
            
        return A

    def spectral_init():
        N = 129
        # Graph where edge i-j exists if i+j in [128, 256]
        # This graph is bipartite? No. 
        # i+j even/odd.
        # Let's just use the adjacency matrix of "compatibility".
        adj = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: continue
                s = i + j
                if 128 <= s <= 256:
                    adj[i][j] = 1.0
        
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            principal_vec = np.abs(eigvecs[:, -1])
        except:
            principal_vec = np.ones(N)
            
        # Select top k elements
        # k around 40-60
        k = 50
        indices = np.argsort(-principal_vec)
        A = set(indices[:k].tolist())
        
        # Fix coverage if needed
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        if not covered:
            # Greedy fill
            while not covered:
                uncovered = [t for t in targets if counts[t] == 0]
                if not uncovered: break
                best_x = -1
                best_gain = -1
                for x in range(n + 1):
                    if x in A: continue
                    new_sums = set()
                    for a in A:
                        s = x + a
                        if 128 <= s <= 256: new_sums.add(s)
                    s = 2*x
                    if 128 <= s <= 256: new_sums.add(s)
                    gain = len([s for s in new_sums if counts[s] == 0])
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain == 0: break
                A.add(best_x)
                counts = compute_counts(A)
                peak, sq, covered = get_metrics(counts)
                
        return A

    # 1. Initialization
    A = greedy_init()
    counts = compute_counts(A)
    curr_peak, curr_sq, curr_covered = get_metrics(counts)
    
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    # Try spectral init too, keep better one
    A2 = spectral_init()
    counts2 = compute_counts(A2)
    p2, s2, c2 = get_metrics(counts2)
    if c2 and (p2 < best_peak or (p2 == best_peak and s2 < best_sq)):
        A = A2
        counts = counts2
        curr_peak, curr_sq, curr_covered = p2, s2, c2
        best_peak, best_sq = p2, s2
        best_A = set(A)

    # 2. Local Search
    # We will use a simulated annealing-like approach or just hill climbing with restarts.
    # Given the small search space, hill climbing with 2-swaps is strong.
    
    no_improve_count = 0
    max_no_improve = 200 # Restart threshold
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try 1-Swaps (Add, Remove, Swap)
        
        # Try Removals
        A_list = list(A)
        random.shuffle(A_list)
        for x in A_list:
            A_trial = set(A)
            A_trial.remove(x)
            counts_trial = compute_counts(A_trial)
            p, s, cov = get_metrics(counts_trial)
            if cov:
                if p < curr_peak or (p == curr_peak and s < curr_sq):
                    A = A_trial
                    counts = counts_trial
                    curr_peak, curr_sq, curr_covered = p, s, cov
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                        best_peak = curr_peak
                        best_sq = curr_sq
                        best_A = set(A)
                    improved = True
                    no_improve_count = 0
                    break
        if improved: continue

        # Try Swaps (1 in, 1 out)
        A_list = list(A)
        C_list = [i for i in range(n + 1) if i not in A]
        random.shuffle(A_list)
        random.shuffle(C_list)
        
        # Limit checks if time is tight? 
        # For now, check all. 50 * 80 = 4000 checks. Each check O(50^2) = 2500 ops. Total 10M ops. ~0.1s.
        for x in A_list:
            for y in C_list:
                A_trial = set(A)
                A_trial.remove(x)
                A_trial.add(y)
                counts_trial = compute_counts(A_trial)
                p, s, cov = get_metrics(counts_trial)
                if cov:
                    if p < curr_peak or (p == curr_peak and s < curr_sq):
                        A = A_trial
                        counts = counts_trial
                        curr_peak, curr_sq, curr_covered = p, s, cov
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                            best_peak = curr_peak
                            best_sq = curr_sq
                            best_A = set(A)
                        improved = True
                        no_improve_count = 0
                        break
            if improved: break
        if improved: continue

        # 2. Try 2-Swaps (2 in, 2 out)
        # This is expensive. Only do if stuck.
        if no_improve_count > 50:
            A_list = list(A)
            C_list = [i for i in range(n + 1) if i not in A]
            
            # Random sample of pairs to avoid O(N^4)
            # Try 100 random 2-swaps
            for _ in range(100):
                if len(A) < 2 or len(C_list) < 2: break
                x1, x2 = random.sample(A_list, 2)
                y1, y2 = random.sample(C_list, 2)
                
                A_trial = set(A)
                A_trial.discard(x1)
                A_trial.discard(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                counts_trial = compute_counts(A_trial)
                p, s, cov = get_metrics(counts_trial)
                if cov:
                    # Accept if better, or with small probability if worse (Annealing)
                    # For now, strict hill climbing
                    if p < curr_peak or (p == curr_peak and s < curr_sq):
                        A = A_trial
                        counts = counts_trial
                        curr_peak, curr_sq, curr_covered = p, s, cov
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                            best_peak = curr_peak
                            best_sq = curr_sq
                            best_A = set(A)
                        improved = True
                        no_improve_count = 0
                        break
                        
        if not improved:
            no_improve_count += 1
            
            # Perturbation: Randomly swap 3 elements to escape local minimum
            if no_improve_count > max_no_improve:
                # Check if we can do a 3-swap
                if len(A) >= 3:
                    A_list = list(A)
                    C_list = [i for i in range(n + 1) if i not in A]
                    if len(C_list) >= 3:
                        x1, x2, x3 = random.sample(A_list, 3)
                        y1, y2, y3 = random.sample(C_list, 3)
                        A_trial = set(A)
                        A_trial.discard(x1)
                        A_trial.discard(x2)
                        A_trial.discard(x3)
                        A_trial.add(y1)
                        A_trial.add(y2)
                        A_trial.add(y3)
                        
                        # Only accept if it doesn't break coverage? 
                        # Or accept anyway to explore?
                        # If we accept invalid states, we need to handle it.
                        # Let's only accept valid states for the current state, 
                        # but if we are stuck, maybe accept a valid state even if worse?
                        # No, let's just reset to best_A if we are stuck for too long?
                        # Better: Perform a "kick" that is valid.
                        # Find a valid 3-swap.
                        found_valid = False
                        for _ in range(20):
                            if len(A) >= 3:
                                A_list = list(A)
                                C_list = [i for i in range(n + 1) if i not in A]
                                if len(C_list) >= 3:
                                    x1, x2, x3 = random.sample(A_list, 3)
                                    y1, y2, y3 = random.sample(C_list, 3)
                                    A_trial = set(A)
                                    A_trial.discard(x1)
                                    A_trial.discard(x2)
                                    A_trial.discard(x3)
                                    A_trial.add(y1)
                                    A_trial.add(y2)
                                    A_trial.add(y3)
                                    
                                    counts_trial = compute_counts(A_trial)
                                    p, s, cov = get_metrics(counts_trial)
                                    if cov:
                                        # Accept even if worse to escape, but track best
                                        A = A_trial
                                        counts = counts_trial
                                        curr_peak, curr_sq, curr_covered = p, s, cov
                                        found_valid = True
                                        break
                        if found_valid:
                            no_improve_count = 0
                        else:
                            # Stuck hard. Reset to best
                            A = set(best_A)
                            counts = compute_counts(A)
                            curr_peak, curr_sq, curr_covered = best_peak, best_sq, True
                            no_improve_count = 0

    return sorted(list(best_A))
