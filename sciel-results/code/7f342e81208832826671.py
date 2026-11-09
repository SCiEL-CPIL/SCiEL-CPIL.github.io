import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Precompute all arithmetic progressions
    # An AP is defined by (start, step) where indices are start, start+step, start+2*step, ... < n
    # We need to track all such progressions
    # For efficiency, we'll precompute for each index, which progressions it belongs to
    
    # First, let's understand the structure:
    # For step d, starts are 0, 1, ..., d-1 (but only those where start < n)
    # Actually starts can be 0 to n-1, but for a given step d, starts 0 to d-1 cover all residue classes
    # Wait, no. For step d, an AP starting at s has elements s, s+d, s+2d, ...
    # s can be 0 to n-1, but if s >= d, then s = s' + kd for some k, so the AP starting at s is a subsequence
    # of the AP starting at s' = s mod d.
    # Actually, for our purpose, we need ALL APs including sub-APs.
    # But note: an AP with step d starting at s where s = s0 + kd is a suffix of the AP with step d starting at s0.
    # So the set of all APs is: for each d from 1 to n, for each s from 0 to d-1, the APs are
    # s, s+d, s+2d, ... (the full one) and all its suffixes.
    
    # However, for the purpose of computing the sum, we can just enumerate all (start, step) pairs
    # where start < n and step >= 1, and the AP has at least 1 element.
    # But that's O(n^2) APs, each of length up to n, so total O(n^3) elements.
    # n=256, n^3 = 16.7 million, which might be okay but let's be smarter.
    
    # Better approach: For each step d, group indices by residue mod d.
    # For residue r (0 <= r < d), the indices are r, r+d, r+2d, ...
    # All APs with step d are suffixes of these sequences.
    # So for each (d, r), we have one base sequence, and all APs with step d and start in that residue class
    # are suffixes of it.
    
    # For computing the maximum absolute sum, we need to check all these suffixes.
    # For a fixed (d, r) sequence of length L, the suffix sums can be computed in O(L).
    # Total work: sum over d=1 to n of sum over r=0 to d-1 of L_{d,r}
    # where L_{d,r} is the length of the sequence for residue r mod d.
    # L_{d,r} = floor((n-1-r)/d) + 1 if r < n else 0
    # Sum over r for fixed d: sum_{r=0}^{d-1} (floor((n-1-r)/d) + 1)
    # This is approximately n for each d, so total is O(n^2) = 65536, which is fine.
    
    # Precompute the list of APs as lists of indices
    # But storing all APs explicitly might use too much memory.
    # Instead, we'll compute the max discrepancy on the fly during evaluation.
    
    # For the search, we'll use simulated annealing or greedy hill climbing.
    # Let's use a combination: start with random, then do local search.
    
    # Precompute for each index, the list of (ap_id, position_in_ap) or just compute sums incrementally.
    
    # Alternative: For each step d, precompute the sequences.
    # sequences[d][r] = list of indices with index % d == r, in order
    sequences = {}
    for d in range(1, n + 1):
        sequences[d] = [[] for _ in range(d)]
        for idx in range(n):
            r = idx % d
            sequences[d][r].append(idx)
    
    # Now, for a given sign assignment, compute max |sum| over all APs
    def compute_max_discrepancy(signs):
        max_disc = 0
        for d in range(1, n + 1):
            for r in range(d):
                seq = sequences[d][r]
                if not seq:
                    continue
                # Compute suffix sums
                # The APs with step d and start in this residue class are suffixes of seq
                # Suffix starting at position k in seq: sum of signs[seq[k]], signs[seq[k+1]], ...
                # We can compute this from right to left
                suffix_sum = 0
                for k in range(len(seq) - 1, -1, -1):
                    suffix_sum += signs[seq[k]]
                    abs_val = abs(suffix_sum)
                    if abs_val > max_disc:
                        max_disc = abs_val
        return max_disc
    
    # Initial random assignment
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    best_signs = signs[:]
    best_disc = compute_max_discrepancy(best_signs)
    
    # Local search: try flipping single bits
    # To speed up, we can precompute for each index, which (d, r) it belongs to and its position
    # But for simplicity, let's just recompute the full discrepancy for each flip.
    # With n=256, each evaluation is O(n^2) ~ 65k operations, and we can do maybe 100k flips in 55s?
    # 100k * 65k = 6.5 billion, too slow.
    
    # We need incremental update.
    # When we flip index i from +1 to -1 (or vice versa), the change is -2*signs[i] for all APs containing i.
    # So we need to track the current sum for each AP, and update them.
    
    # Let's restructure: precompute for each index, the list of (d, r, position) where it appears.
    # Actually, for each (d, r), the index i appears if i % d == r, and its position is i // d.
    # So for index i, the (d, r) pairs are: for each d from 1 to n, r = i % d, position = i // d.
    # But we only care about (d, r) where the sequence is non-empty, which is always true if i < n.
    
    # Let's precompute: for each index i, list of (d, r) where it appears.
    # But that's n * n = 65k entries, which is fine.
    
    index_to_aps = [[] for _ in range(n)]
    for d in range(1, n + 1):
        for r in range(d):
            seq = sequences[d][r]
            for pos, idx in enumerate(seq):
                index_to_aps[idx].append((d, r, pos))
    
    # Now, maintain current sums for each (d, r)
    # We'll use a dictionary or a 2D array
    # sums[d][r] = current sum for the AP (d, r)
    # But note: the AP (d, r) is the full sequence, and we need suffix sums.
    # Actually, we need to track the sum of the full sequence, and also the suffix sums.
    # This is tricky for incremental update.
    
    # Alternative: track for each (d, r), the array of prefix sums or the current suffix sums.
    # When we flip index i, which is at position pos in sequence (d, r),
    # all suffixes starting at position k <= pos are affected.
    # Specifically, suffix starting at k gets change -2*signs[i] if k <= pos, else 0.
    
    # This is still O(L) per flip per (d, r), which is O(n) per flip, and n flips per index, so O(n^2) per flip.
    # Total O(n^3) per iteration, too slow.
    
    # Let's try a different approach: use numpy for vectorized operations.
    # Or, use a simpler local search with periodic full recomputation.
    
    # Given time constraints, let's do a greedy approach:
    # Start with random, then for each index, try both signs and keep the one that minimizes max discrepancy.
    # This is O(n^2) per index, O(n^3) total, which is 16.7 million, might be okay.
    
    # Actually, let's do simulated annealing with periodic full recomputation.
    # Or, let's do a simple hill climbing with full recomputation, but limit the number of iterations.
    
    # Given the 55s limit, let's estimate: each full computation is ~65k operations.
    # In Python, 65k operations might take ~0.001s, so we can do ~50k iterations in 55s.
    # 50k iterations of flipping one bit and recomputing fully.
    
    # Let's try that.
    
    current_signs = signs[:]
    current_disc = best_disc
    
    # Simple hill climbing
    flip_count = 0
    while time.time() - start_time < time_limit - 2.0:
        # Pick a random index to flip
        i = rng.randint(0, n - 1)
        old_sign = current_signs[i]
        new_sign = -old_sign
        current_signs[i] = new_sign
        
        new_disc = compute_max_discrepancy(current_signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            best_signs = current_signs[:]
            best_disc = current_disc
        else:
            current_signs[i] = old_sign
        
        flip_count += 1
        if flip_count % 10000 == 0:
            break
    
    return best_signs
