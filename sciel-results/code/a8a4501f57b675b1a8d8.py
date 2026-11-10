import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    n = 128
    targets = np.arange(128, 257)
    
    # Precompute pairs for each target sum
    # pairs_for_sum[s] = list of (a, b) such that a+b=s, 0<=a<=b<=128
    pairs_for_sum = [[] for _ in range(257)]
    for a in range(n + 1):
        for b in range(a, n + 1):
            s = a + b
            if s <= 256:
                pairs_for_sum[s].append((a, b))
    
    # We want to select a subset A of [0, 128]
    # Let x_v be 1 if v is in A, 0 otherwise.
    # For each target t, let C_t be the count of representations.
    # C_t = sum_{(a,b) in pairs_for_sum[t]} x_a * x_b
    # Note: if a==b, term is x_a^2 = x_a. If a!=b, term is x_a x_b.
    
    # Objective: Minimize max_t C_t.
    # Strategy: Simulated Annealing on the binary vector x.
    
    # 1. Construct a valid initial solution
    # We need every t in [128, 256] to have C_t >= 1.
    # A simple heuristic: Greedily add elements to cover targets.
    # Start with empty A.
    # While there are uncovered targets:
    #   Find element v that covers the most uncovered targets (or best improves coverage).
    #   Add v to A.
    # Then, try to remove elements to reduce peak.
    
    def compute_counts_vec(x):
        """
        x is a numpy array of 0/1 of length 129.
        Returns counts array of length 257.
        """
        counts = np.zeros(257, dtype=int)
        # Iterate over all pairs
        # This is O(N^2) where N=129. 129*129/2 ~ 8000 pairs.
        # For each pair (a,b), if x[a] and x[b] are 1, increment counts[a+b].
        # We can vectorize this partially or just loop.
        # Looping 8000 times is very fast.
        # But we call this many times.
        # Let's precompute the pairs as indices.
        
        # Actually, we can precompute the contribution of each variable? No, it's quadratic.
        # Let's just loop over the precomputed pairs.
        for s in range(128, 257):
            # Only compute for targets we care about? 
            # But counts are needed for all sums if we want to be general, 
            # but we only care about 128..256 for validity and peak.
            # However, the update logic in SA needs to be fast.
            pass
        
        # Alternative: Precompute a matrix M where M[a,b] = 1 if a+b in targets?
        # No, let's just compute counts for 128..256 directly.
        c = np.zeros(129, dtype=int) # indices 0..128 correspond to sums 128..256
        for s in range(128, 257):
            idx = s - 128
            total = 0
            for a, b in pairs_for_sum[s]:
                if x[a] and x[b]:
                    total += 1
            c[idx] = total
        return c

    def get_metrics(c):
        """
        c is counts for targets 128..256 (length 129).
        Returns (uncovered_count, peak_count).
        """
        uncovered = np.sum(c == 0)
        peak = np.max(c)
        return uncovered, peak

    def score_solution(x):
        """
        Returns a scalar score for the solution.
        We want to minimize peak, but coverage is a hard constraint.
        Score = uncovered * LARGE_PENALTY + peak + 0.1 * sum(c^2)
        """
        c = compute_counts_vec(x)
        uncovered, peak = get_metrics(c)
        sq = np.sum(c**2)
        penalty = 10000
        return uncovered * penalty + peak + 0.01 * sq

    # Initial Solution Construction
    # Greedy Set Cover
    x = np.zeros(n + 1, dtype=int)
    uncovered_targets = set(range(128, 257))
    
    # Precompute for each element v, which targets it helps cover?
    # If we add v, it forms pairs (v, u) for all u in A.
    # This is dynamic.
    
    # Simpler greedy:
    # While uncovered_targets:
    #   For each v not in A:
    #     Simulate adding v. Count how many new targets are covered.
    #     Pick v with max new coverage. Tie-break by min peak increase?
    #     Add v.
    
    # This is O(T * N * N) roughly. 129 * 129 * 129 ~ 2M ops. Fast.
    
    while uncovered_targets:
        best_v = -1
        best_new_cov = -1
        best_peak_after = float('inf')
        
        current_counts = compute_counts_vec(x)
        
        for v in range(n + 1):
            if x[v] == 1:
                continue
            # Simulate adding v
            # New pairs: (v, u) for u in A (including v? no, x[v] is 0 now, so (v,v) is new)
            # Wait, if x[v] becomes 1, pairs involving v are (v, u) for u in A_new.
            # A_new = A_old U {v}.
            # Pairs: (v, u) for u in A_old, and (v, v).
            
            # Which targets are covered by these new pairs?
            # s = v + u for u in A_old.
            # s = 2v.
            
            new_covered = set()
            for u in range(n + 1):
                if x[u] == 1:
                    s = v + u
                    if 128 <= s <= 256:
                        new_covered.add(s)
            s = 2 * v
            if 128 <= s <= 256:
                new_covered.add(s)
            
            # How many of these are currently uncovered?
            new_uncovered_covered = new_covered & uncovered_targets
            num_new = len(new_uncovered_covered)
            
            if num_new > best_new_cov or (num_new == best_new_cov and best_v == -1):
                # Check peak if tied? For now, just pick max new coverage.
                # If we want to be smarter, we could check peak, but it's expensive.
                # Let's stick to max coverage first.
                best_new_cov = num_new
                best_v = v
        
        if best_v == -1:
            break # No improvement possible? Should not happen if full set covers.
            
        x[best_v] = 1
        # Update uncovered_targets
        # Recompute counts to find new uncovered?
        # Or just update based on the pairs added.
        # Easier: recompute counts for the whole set? 129 targets * ~64 pairs avg = 8k ops.
        # Doing this 129 times is 1M ops. Fine.
        
        current_counts = compute_counts_vec(x)
        # Update uncovered_targets
        for i in range(129):
            t = 128 + i
            if current_counts[i] > 0:
                if t in uncovered_targets:
                    uncovered_targets.remove(t)
    
    # Now we have a valid solution.
    # Minimize peak using Simulated Annealing.
    
    curr_x = x.copy()
    curr_counts = compute_counts_vec(curr_x)
    curr_uncovered, curr_peak = get_metrics(curr_counts)
    curr_score = score_solution(curr_x)
    
    best_x = curr_x.copy()
    best_score = curr_score
    best_peak = curr_peak
    best_uncovered = curr_uncovered
    
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of elements for swapping
    elements = list(range(n + 1))
    
    steps = 0
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            
        # Random move: Flip one bit, or swap two bits?
        # Flipping one bit is a good neighborhood.
        v = random.choice(elements)
        
        # Try flipping v
        new_x = curr_x.copy()
        new_x[v] = 1 - new_x[v]
        
        # Compute delta in score efficiently?
        # Score depends on counts.
        # Delta in counts:
        # If adding v (0->1):
        #   Increment counts for s = v+u (u in A_old) and s=2v.
        # If removing v (1->0):
        #   Decrement counts for s = v+u (u in A_new \ {v}) and s=2v.
        
        # Let's just compute new counts. It's fast enough?
        # compute_counts_vec is ~8000 ops.
        # If we do 1000 steps/sec, that's 8M ops/sec. Python is slower, maybe 1M ops/sec.
        # So ~100 steps/sec. 85s -> 8500 steps. Might be enough.
        
        # Optimization: Update counts incrementally.
        # Let's implement incremental update.
        
        new_counts = curr_counts.copy()
        
        if new_x[v] == 1:
            # Adding v
            # Pairs (v, u) for u in curr_x where curr_x[u]==1
            # Note: curr_x[v] is 0.
            for u in range(n + 1):
                if curr_x[u] == 1:
                    s = v + u
                    if 128 <= s <= 256:
                        new_counts[s - 128] += 1
            s = 2 * v
            if 128 <= s <= 256:
                new_counts[s - 128] += 1
        else:
            # Removing v
            # Pairs (v, u) for u in curr_x where curr_x[u]==1 and u != v
            # Note: curr_x[v] is 1.
            for u in range(n + 1):
                if curr_x[u] == 1 and u != v:
                    s = v + u
                    if 128 <= s <= 256:
                        new_counts[s - 128] -= 1
            s = 2 * v
            if 128 <= s <= 256:
                new_counts[s - 128] -= 1
        
        new_uncovered, new_peak = get_metrics(new_counts)
        new_sq = np.sum(new_counts**2)
        new_score = new_uncovered * 10000 + new_peak + 0.01 * new_sq
        
        delta = new_score - curr_score
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            curr_x = new_x
            curr_counts = new_counts
            curr_uncovered = new_uncovered
            curr_peak = new_peak
            curr_score = new_score
            
            if new_uncovered == 0 and (new_peak < best_peak or (new_peak == best_peak and new_score < best_score)):
                best_x = curr_x.copy()
                best_peak = new_peak
                best_score = new_score
                best_uncovered = new_uncovered
                
        T *= cooling
        steps += 1
        
        # Periodic restart or kick if stuck?
        # If we haven't improved in a while, maybe increase T?
        # Or just let it cool.
        
    # Final check: ensure best_x is valid (uncovered == 0)
    # If not, we might need to return a valid solution from the path.
    # But our SA accepts invalid solutions with high penalty, so it should try to fix them.
    # However, if it ends in an invalid state, we should return the best valid one found.
    # best_x is only updated if new_uncovered == 0.
    # So best_x is always valid (if found).
    # What if no valid solution was found? (Unlikely given greedy init)
    # If best_uncovered > 0, we should try to repair best_x or return curr_x if valid?
    # Let's check curr_x validity at the end.
    
    if best_uncovered > 0:
        # Try to repair curr_x or best_x
        # For now, if best is invalid, we have a problem.
        # But greedy init ensures we start valid.
        # SA might move to invalid states.
        # We track best valid solution.
        # If no valid solution found (shouldn't happen), return greedy init.
        pass

    # Return sorted list of elements in best_x
    A = [i for i in range(n + 1) if best_x[i] == 1]
    return sorted(A)

# Quick test to ensure it runs
if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res)
    
    # Verify
    A = set(res)
    counts = np.zeros(257, dtype=int)
    for a in A:
        for b in A:
            if a <= b:
                s = a + b
                if s <= 256:
                    counts[s] += 1
    peak = np.max(counts[128:257])
    min_c = np.min(counts[128:257])
    print(f"Peak: {peak}, Min: {min_c}")
