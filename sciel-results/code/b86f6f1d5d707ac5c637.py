import numpy as np

def heuristics(demands, capacity):
       demands = np.asarray(demands, dtype=float)
       n = len(demands)
       
       # Calculate pair sums
       sums = demands[:, None] + demands[None, :]
       
       # Identify valid pairs (sum <= capacity)
       valid = sums <= capacity
       
       # Calculate gap for valid pairs
       # For invalid pairs, gap is positive (sum > capacity), so capacity - sum is negative.
       # We only care about gap where sum <= capacity.
       gaps = capacity - sums
       
       # Initialize heuristic with a very small value
       eta = np.full((n, n), 1e-9, dtype=float)
       
       # Apply heuristic for valid pairs
       # We want high eta for small gap.
       # Using exp(-alpha * gap)
       # If gap is 0, eta = 1.
       # If gap is large, eta approaches 0.
       # We need to ensure eta > 0 for valid pairs to not be clamped to 1e-9 unnecessarily if it gets too small?
       # "Heuristic entries that are not positive become 1e-9".
       # So if exp(...) underflows to 0, it becomes 1e-9. That's fine.
       
       # Let's pick alpha. Max gap is 150 - 40 = 110.
       # If alpha = 0.05, exp(-5.5) ~ 0.004.
       # If alpha = 0.1, exp(-11) ~ 1.6e-5.
       # Let's use 0.1 to strongly penalize loose packs.
       
       # We must be careful: if gap is negative (invalid), we shouldn't apply this.
       # valid mask handles this.
       
       # Calculate score for valid pairs
       # To avoid 0 values in exp for large gaps, we can add a small epsilon or just let it go.
       # The prompt says "not positive" -> 1e-9.
       # exp(x) is always positive. So it will never be "not positive".
       # So valid pairs will always have eta > 0.
       
       # We want to encourage placing items that are similar in size to each other?
       # Or maybe we want to penalize mixing sizes?
       # The previous finding suggests: "Introducing item size (d_j) into the numerator and squaring the waste penalty in the denominator creates a stronger gradient for tight packing of large items"
       # Let's try: eta[i, j] = d_j / (1 + (capacity - d_i - d_j)^2) for valid pairs.
       # This rewards placing a large item j after i if the fit is tight (small waste).
       # The d_j in numerator rewards larger items more.
       
       # Let's test this hypothesis.
       # d_j is demands[j].
       # waste = capacity - demands[i] - demands[j].
       # score = demands[j] / (1 + waste^2)
       
       # For valid pairs:
       waste = gaps[valid]
       # We need demands[j] for each valid pair.
       # valid is a boolean matrix.
       # We can extract indices.
       i_idx, j_idx = np.where(valid)
       d_j = demands[j_idx]
       
       scores = d_j / (1.0 + waste * waste)
       
       # Assign
       eta[valid] = scores
       
       # Diagonal: set to 0 or 1e-9.
       np.fill_diagonal(eta, 1e-9)
       
       return eta
