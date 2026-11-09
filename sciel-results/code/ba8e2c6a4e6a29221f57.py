"""Rewritten program for low discrepancy on all arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 52.0  # safety margin before 60s kill
    
    # Precompute all arithmetic progressions:
    # For each step d >= 1 and start s (1-indexed), the AP is s, s+d, s+2d, ...
    # We need to consider ALL contiguous subsequences of each AP.
    # Key: for a fixed (start, step), the AP gives a sequence of indices.
    # The max |sum| over any contiguous subsequence of that sequence is what matters.
    # 
    # To efficiently evaluate: for each (start, step), compute prefix sums of the
    # signs along that AP, then max|sum| = max(prefix) - min(prefix) over that AP.
    #
    # But we need to do this for ALL (start, step) pairs. Let's enumerate them.
    # For step d, starts s = 1, 2, ..., n-d (so that s+d <= n, i.e., at least 2 elements)
    # Actually even length-1 APs matter (single element), so |x[i]| = 1 always.
    # The max is at least 1. We want to minimize the max over ALL APs.
    
    # Let's think about the structure. An AP is defined by (a, d, L) where
    # a >= 1, d >= 1, L >= 1, and a + (L-1)*d <= n.
    # The sum is sum_{k=0}^{L-1} x[a + k*d].
    # We want min over x in {-1,+1}^n of max over all (a,d,L) of |sum|.
    
    # Efficient evaluation:
    # For each step d from 1 to n-1:
    #   For each start a from 1 to n:
    #     Compute the sequence x[a], x[a+d], x[a+2d], ... (as long as index <= n)
    #     Compute all contiguous subsequence sums, track max abs.
    #
    # This is O(n^2 * n) in the worst case which is too slow for repeated evaluation.
    #
    # Better approach: For a fixed d, we can compute for each residue class mod d.
    # For residue r (0-indexed: r = 0, 1, ..., d-1), the sequence is x[r], x[r+d], x[r+2d], ...
    # For each such sequence, the max |contiguous sum| can be computed in O(length)
    # using prefix sums.
    #
    # Total work per evaluation: sum over d=1 to n-1 of (d sequences, each of length ~n/d)
    # = sum over d of n = O(n^2) per evaluation. For n=256, that's ~65536 operations.
    # This is fast enough for local search.
    
    def evaluate(x):
        """Compute max |sum| over all arithmetic progressions."""
        x = np.asarray(x)  # 0-indexed, length n
        max_abs = 1  # singletons always give 1
        for d in range(1, n):
            for r in range(d):
                # Sequence: x[r], x[r+d], x[r+2d], ...
                idx = np.arange(r, n, d)
                if len(idx) < 2:
                    continue
                seq = x[idx]
                # Prefix sums
                ps = np.cumsum(seq)
                # Max contiguous sum = max(ps) - min(ps[0:-1] if len(ps)>1 else ps)
                # Actually: max over i<j of ps[j]-ps[i] where we include ps[0]=0
                # Let's prepend 0 to prefix sums
                ps_full = np.concatenate(([0], ps))
                # max subarray sum = max(ps_full) - min of ps_full up to that point
                # More precisely: max_{i<=j} (ps_full[j] - ps_full[i]) for i<j
                # = max over j of (ps_full[j] - min_{i<j} ps_full[i])
                min_so_far = np.minimum.accumulate(ps_full)
                # But we need min_{i<j}, not min_{i<=j}
                # Let's do it differently:
                # max_contig = max_{j} ps_full[j] - min_{i<j} ps_full[i]
                # min_before_j = min of ps_full[0:j]
                if len(ps_full) > 1:
                    min_before = np.empty(len(ps_full))
                    min_before[0] = ps_full[0]
                    min_before[1:] = np.minimum.accumulate(ps_full[:-1])
                    max_contig = np.max(ps_full - min_before)
                    min_contig = -np.max(-ps_full + min_before)  # by symmetry with negation
                    # Actually min contiguous sum = min over i<j of (ps_full[j]-ps_full[i])
                    # = -(max over i<j of (ps_full[i]-ps_full[j]))
                    # Let's just compute both directly
                    # max_contig = max_{i<j} (ps_full[j] - ps_full[i])
                    # min_contig = min_{i<j} (ps_full[j] - ps_full[i])
                    # We can compute: for each j, ps_full[j] - min_{i<j} ps_full[i] gives max ending at j
                    # and ps_full[j] - max_{i<j} ps_full[i] gives min ending at j
                    max_so_far = np.maximum.accumulate(ps_full)
                    max_before = np.empty(len(ps_full))
                    max_before[0] = ps_full[0]
                    max_before[1:] = np.maximum.accumulate(ps_full[:-1])
                    max_contig = int(np.max(ps_full - min_before))
                    min_contig = int(np.min(ps_full - max_before))
                    cur_max = max(abs(max_contig), abs(min_contig))
                    if cur_max > max_abs:
                        max_abs = cur_max
        return max_abs
    
    # Initial solution: Thue-Morse sequence (known to have good discrepancy properties)
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        tm = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits
            tm[i] = 1 if bin(i).count('1') % 2 == 0 else -1
        return tm
    
    x = thue_morse(n).copy()
    best_x = x.copy()
    best_score = evaluate(x)
    
    # Local search with simulated annealing
    rng = random.Random(42)
    temperature = 1.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(0, n-1)
        x[i] = -x[i]
        new_score = evaluate(x)
        
        if new_score <= best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - best_score
            if delta > 0:
                accept_prob = np.exp(-delta / temperature)
                if rng.random() < accept_prob:
                    pass  # keep the flip
                else:
                    x[i] = -x[i]  # revert
            else:
                # Should not happen since we checked new_score <= best_score above
                pass
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
