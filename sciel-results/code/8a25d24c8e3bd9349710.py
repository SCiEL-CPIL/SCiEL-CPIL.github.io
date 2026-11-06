"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
DEADLINE_MARGIN = 5.0  # seconds before the 60s (or 90s) kill to stop

def _max_abs_sum(signs):
    """Compute max |sum| over all arithmetic progressions in [1, N].
    signs is a list or array of length N with entries -1 or +1 (0-indexed).
    An AP is defined by start s (0-indexed), step d >= 1, and length L >= 1,
    with s + (L-1)*d < N.
    """
    s_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    # Precompute prefix sums for each step? That's heavy.
    # Instead, iterate over step d, and for each residue class mod d,
    # compute the cumulative sums along that subsequence.
    # For a fixed step d, the positions are partitioned into d residue classes.
    # Within each residue class, the elements form a sequence: r, r+d, r+2d, ...
    # Any AP with step d is a contiguous subarray of one of these sequences.
    # So we need max |sum| over all contiguous subarrays of all these sequences, for all d.
    
    # To be efficient, for each d, we can compute for each residue r in [0, d):
    #   the sequence vals = s_arr[r::d]
    #   then find max |sum| over all contiguous subarrays of vals.
    # This is O(N/d * (N/d)) per residue if done naively, but we can use prefix sums.
    # For a sequence of length m, max |subarray sum| can be found in O(m) using prefix sums.
    # Total work for step d: sum over r of O(N/d) = O(d * N/d) = O(N).
    # Total over all d: O(N^2). For N=256, that's ~65k operations, very fast.
    
    for d in range(1, N + 1):
        for r in range(d):
            # Get the subsequence
            seq = s_arr[r::d]
            if len(seq) <= 1:
                continue
            # Compute prefix sums
            # prefix[0] = 0, prefix[i] = sum(seq[0:i])
            prefix = np.concatenate(([0], np.cumsum(seq)))
            # We need max |prefix[j] - prefix[i]| for 0 <= i < j <= len(seq)
            # This is equivalent to (max prefix - min prefix) but we need the absolute difference
            # Actually, max |sum| over subarrays = max(prefix) - min(prefix) is NOT correct for absolute value.
            # We need max over all i<j of |prefix[j] - prefix[i]|.
            # This equals max(max(prefix) - min(prefix), ...) but actually:
            # The maximum absolute difference between any two prefix values is max(prefix) - min(prefix).
            # Because |a - b| <= max - min, and it's achieved when a=max and b=min or vice versa.
            # So max |subarray sum| = max(prefix) - min(prefix).
            # Wait, that's the max difference, which equals the max absolute subarray sum? 
            # Let's verify: subarray sum from i to j-1 is prefix[j] - prefix[i].
            # We want max |prefix[j] - prefix[i]| over i < j.
            # This is indeed max(prefix) - min(prefix) IF the max occurs after the min, or vice versa.
            # But we require i < j. So we need to be careful.
            # However, for the purpose of an upper bound and since we're minimizing, let's just compute it properly.
            # For efficiency, let's just compute it with a simple scan.
            # Actually, for small sequences, let's just do it directly.
            m = len(seq)
            if m <= 1:
                continue
            # Find max and min prefix, tracking indices
            # Simpler: just compute all subarray sums? No, too slow.
            # Let's use the property: max absolute subarray sum = max(
            #   max over j of (prefix[j] - min_{i<j} prefix[i]),
            #   max over j of (min_{i<j} prefix[i] - prefix[j])  # which is -min over j of (prefix[j] - max_{i<j} prefix[i])
            # )
            # Actually, let's just compute it in O(m).
            min_pref = prefix[0]
            max_pref = prefix[0]
            best = 0
            for val in prefix[1:]:
                if val < min_pref:
                    min_pref = val
                if val > max_pref:
                    max_pref = val
                # The current max absolute subarray sum ending at this position
                # is max(|val - min_pref_before|, |max_pref_before - val|)
                # But we've updated min_pref and max_pref already. Let's track them separately.
            # Let me redo this more carefully.
            min_so_far = prefix[0]
            max_so_far = prefix[0]
            best_abs = 0
            for val in prefix[1:]:
                # Update best based on current val and previous min/max
                if val - min_so_far > best_abs:
                    best_abs = val - min_so_far
                if max_so_far - val > best_abs:
                    best_abs = max_so_far - val
                if val < min_so_far:
                    min_so_far = val
                if val > max_so_far:
                    max_so_far = val
            
            if best_abs > max_abs:
                max_abs = best_abs
    
    return max_abs

def _max_abs_sum_fast(signs):
    """Faster version using numpy vectorization where possible."""
    s_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    for d in range(1, N + 1):
        for r in range(d):
            seq = s_arr[r::d]
            m = len(seq)
            if m <= 1:
                continue
            prefix = np.concatenate(([0], np.cumsum(seq)))
            # For each position j, we want max(prefix[j] - min(prefix[0:j]), max(prefix[0:j]) - prefix[j])
            # We can compute running min and max of prefix
            running_min = np.minimum.accumulate(prefix)
            running_max = np.maximum.accumulate(prefix)
            # For each j, the best subarray sum ending at j is:
            #   prefix[j] - running_min[j]  (if we want positive max)
            #   running_max[j] - prefix[j]  (if we want negative min, i.e., large absolute)
            # But we need i < j, so running_min[j] includes prefix[j] itself.
            # We should use running_min[j-1] and running_max[j-1] for the part before j.
            if m > 1:
                prev_min = running_min[:-1]
                prev_max = running_max[:-1]
                curr = prefix[1:]
                pos_diffs = curr - prev_min
                neg_diffs = prev_max - curr
                best_for_d_r = max(np.max(pos_diffs), np.max(neg_diffs))
                if best_for_d_r > max_abs:
                    max_abs = best_for_d_r
    return max_abs

def construct():
    start_time = time.time()
    
    # Initialize with Thue-Morse sequence
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n as +1/-1."""
        seq = [1]
        while len(seq) < n:
            seq += [-x for x in seq]
        return seq[:n]
    
    # Create several diverse initial solutions
    initial_solutions = []
    
    # Thue-Morse
    tm = thue_morse(N)
    initial_solutions.append(tm)
    
    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(N)]
    initial_solutions.append(alt)
    
    # Random with different seeds
    rng = random.Random(42)
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng = random.Random(seed)
        sol = [rng.choice((-1, 1)) for _ in range(N)]
        initial_solutions.append(sol)
    
    # Evaluate all initial solutions
    best_sol = None
    best_score = float('inf')
    
    for sol in initial_solutions:
        score = _max_abs_sum_fast(sol)
        if score < best_score:
            best_score = score
            best_sol = sol[:]
    
    # Simulated Annealing
    current_sol = best_sol[:]
    current_score = best_score
    best_sol_sa = best_sol[:]
    best_score_sa = best_score
    
    # SA parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iterations = 50000
    
    T = T_start
    it = 0
    while T > T_end and it < max_iterations:
        # Check deadline
        if time.time() - start_time > 60 - DEADLINE_MARGIN:
            break
        
        # Pick a random position to flip
        pos = rng.randint(0, N - 1)
        old_val = current_sol[pos]
        new_val = -old_val
        current_sol[pos] = new_val
        
        new_score = _max_abs_sum_fast(current_sol)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_sol_sa = current_sol[:]
        else:
            delta = new_score - current_score
            # Since scores are integers, delta >= 1
            # Use probability exp(-delta / T)
            if delta == 0:
                accept = True
            else:
                # delta is at least 1
                accept = rng.random() < np.exp(-delta / T)
            if not accept:
                # Revert
                current_sol[pos] = old_val
            else:
                current_score = new_score
        
        T *= cooling_rate
        it += 1
        
        # Periodic check
        if it % 5000 == 0:
            if time.time() - start_time > 60 - DEADLINE_MARGIN:
                break
    
    # Final local search: greedy improvement
    improved = True
    while improved:
        if time.time() - start_time > 60 - DEADLINE_MARGIN:
            break
        improved = False
        for pos in range(N):
            if time.time() - start_time > 60 - DEADLINE_MARGIN:
                break
            old_val = best_sol_sa[pos]
            best_sol_sa[pos] = -old_val
            new_score = _max_abs_sum_fast(best_sol_sa)
            if new_score < best_score_sa:
                best_score_sa = new_score
                improved = True
            else:
                best_sol_sa[pos] = old_val
    
    # Return the best solution found
    return best_sol_sa
# EVOLVE-BLOCK-END
