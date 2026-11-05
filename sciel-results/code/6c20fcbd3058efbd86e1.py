import time
import random
import numpy as np

def construct():
    N = 128
    target_lo = N
    target_hi = 2 * N  # 256
    n_targets = target_hi - target_lo + 1  # 129 targets
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 80  # 10s safety margin before 90s kill
    
    # ---------- Helpers ----------
    def compute_counts(A_set):
        """Compute representation counts for each target in [128, 256].
        Unordered pairs (a,b) with a <= b, both in A, a+b in [128,256]."""
        A = sorted(A_set)
        counts = np.zeros(n_targets, dtype=np.int32)
        for i, a in enumerate(A):
            for j in range(i, len(A)):
                b = A[j]
                s = a + b
                if s >= target_lo and s <= target_hi:
                    counts[s - target_lo] += 1
        return counts
    
    def evaluate(A_set):
        """Return (is_valid, peak_multiplicity)."""
        counts = compute_counts(A_set)
        if np.any(counts == 0):
            return False, 0
        return True, int(np.max(counts))
    
    def init_contiguous():
        """Seed-like contiguous interval: [64, 128]."""
        return set(range(64, N + 1))
    
    def random_init(k):
        """Random subset of size k from [0, N]."""
        return set(random.sample(range(N + 1), k))
    
    def repair(A_set):
        """Ensure every target has at least one representation.
        Greedily add elements if needed. Returns (A_set, num_added)."""
        A = set(A_set)
        # Check coverage
        counts = compute_counts(A)
        missing = np.where(counts == 0)[0]
        if len(missing) == 0:
            return A, 0
        
        added = 0
        max_iters = 200
        iter_count = 0
        while len(missing) > 0 and iter_count < max_iters:
            iter_count += 1
            # For each missing target, find a pair (a,b) not both in A
            # that would cover it, preferring to add one element
            target = target_lo + missing[0]
            best_elem = None
            best_gain = 0
            for a in range(N + 1):
                if a in A:
                    continue
                # Check how many missing targets would be covered by adding a
                gain = 0
                for b in A:
                    s = a + b
                    if target_lo <= s <= target_hi and counts[s - target_lo] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = a
            if best_elem is None:
                break
            A.add(best_elem)
            added += 1
            # Recompute counts
            counts = compute_counts(A)
            missing = np.where(counts == 0)[0]
        
        return A, added
    
    # ---------- Local search with swap moves ----------
    def local_search(A_set, max_time):
        """Improve peak multiplicity via swap moves (remove one, add one)."""
        A = set(A_set)
        valid, peak = evaluate(A)
        if not valid:
            A, _ = repair(A)
            valid, peak = evaluate(A)
            if not valid:
                return A, peak
        
        best_peak = peak
        best_A = set(A)
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - t0 < deadline and no_improve_count < max_no_improve:
            improved = False
            # Get current counts
            counts = compute_counts(A)
            
            # Try swap moves: for each element in A, try replacing with element not in A
            # Prioritize elements in A that contribute to high-count targets
            A_sorted = sorted(A)
            random.shuffle(A_sorted)
            
            for a_out in A_sorted:
                if time.time() - t0 > deadline:
                    break
                if no_improve_count >= max_no_improve:
                    break
                
                # Try each candidate a_in not in A
                # Limit candidates for speed
                candidates = []
                for a_in in range(N + 1):
                    if a_in not in A:
                        candidates.append(a_in)
                random.shuffle(candidates)
                # Limit to speed up
                candidates = candidates[:40]
                
                for a_in in candidates:
                    if time.time() - t0 > deadline:
                        break
                    
                    # Compute effect of swap
                    # Remove a_out: decrement counts for pairs (a_out, b) where b in A, b >= a_out
                    # Add a_in: increment counts for pairs (a_in, b) where b in A (after removal), b >= a_in
                    # Simpler: just compute new counts (slower but simpler)
                    A_new = set(A)
                    A_new.discard(a_out)
                    A_new.add(a_in)
                    
                    counts_new = compute_counts(A_new)
                    if np.any(counts_new == 0):
                        continue
                    
                    peak_new = int(np.max(counts_new))
                    if peak_new < best_peak:
                        best_peak = peak_new
                        best_A = set(A_new)
                        A = set(A_new)
                        counts = counts_new
                        improved = True
                        no_improve_count = 0
                        break
                    elif peak_new == best_peak:
                        # Accept with some probability for diversification
                        if random.random() < 0.3:
                            A = set(A_new)
                            counts = counts_new
                            improved = True
                            no_improve_count = 0
                            break
                
                if time.time() - t0 > deadline:
                    break
            
            if not improved:
                no_improve_count += 1
                # Try a small random perturbation to escape local optima
                if no_improve_count % 10 == 0:
                    A_list = list(A)
                    for _ in range(3):
                        if A_list:
                            a_rm = random.choice(A_list)
                            A.discard(a_rm)
                            a_add = random.randint(0, N)
                            A.add(a_add)
                    valid, peak = evaluate(A)
                    if not valid:
                        A = set(best_A)
                        valid, peak = evaluate(A)
            
            if time.time() - t0 > deadline:
                break
        
        if best_peak < peak:
            return best_A, best_peak
        return A, peak
    
    # ---------- Main search ----------
    best_A = None
    best_peak = float('inf')
    
    # Start with contiguous seed
    A0 = init_contiguous()
    valid0, peak0 = evaluate(A0)
    if valid0:
        best_A = A0
        best_peak = peak0
    
    # Run local search from seed
    A_ls, peak_ls = local_search(A0, deadline - (time.time() - t0))
    if peak_ls < best_peak:
        best_peak = peak_ls
        best_A = set(A_ls)
    
    # Try a few random inits + local search
    random.seed(42)
    n_trials = 10
    for trial in range(n_trials):
        if time.time() - t0 > deadline:
            break
        # Random size around 80-100
        k = random.randint(70, 100)
        A_r = random_init(k)
        A_r, _ = repair(A_r)
        valid_r, peak_r = evaluate(A_r)
        if not valid_r:
            continue
        if peak_r < best_peak:
            best_peak = peak_r
            best_A = set(A_r)
        
        # Local search from this random init
        A_ls2, peak_ls2 = local_search(A_r, deadline - (time.time() - t0))
        if peak_ls2 < best_peak:
            best_peak = peak_ls2
            best_A = set(A_ls2)
    
    # Final: ensure validity and return as sorted list
    if best_A is None:
        # Fallback: contiguous
        best_A = init_contiguous()
    
    # Final validity check
    valid_final, peak_final = evaluate(best_A)
    if not valid_final:
        # Last resort repair
        best_A, _ = repair(best_A)
        valid_final, peak_final = evaluate(best_A)
        if not valid_final:
            # Absolute fallback
            best_A = init_contiguous()
    
    return sorted(best_A)
