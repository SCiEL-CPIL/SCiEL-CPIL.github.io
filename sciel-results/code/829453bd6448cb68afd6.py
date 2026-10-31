"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import weibull_min

# Weibull parameters for the generated sequences
# We assume a standard distribution for the task.
# If the task generator uses specific parameters, we might need to adjust.
# Common defaults in such tasks: shape (k) = 2, scale (lambda) = 10 or similar.
# Given capacity 100 and 1000 items, mean item size is likely around 10-20.
# Let's estimate parameters based on typical online bin packing benchmarks.
# If mean = 10, and k=2, then lambda = mean / Gamma(1 + 1/k) = 10 / Gamma(1.5) ≈ 10 / 0.886 ≈ 11.28.
# Let's assume k=2, lambda=11.28 for now.
# We can refine these if we had more info, but this is a reasonable structural guess.
# Actually, a safer bet for "Weibull sequences" in these tasks might be k=2, scale=10.
# Let's use k=2, scale=10.
# Mean = 10 * Gamma(1.5) ≈ 8.86.
# Let's check if items fit in capacity 100. Yes.

# Precompute the value function V(r) for remaining space r.
# V(r) = expected number of items that can be packed into space r.
# This is a classic DP problem: V(r) = sum_{x} P(X=x) * (1 + V(r-x)) for x <= r, else 0.
# We approximate this with a grid.

GRID_SIZE = 2000  # 0 to 100 in steps of 0.05
CAPACITY = 100.0
STEP = CAPACITY / (GRID_SIZE - 1)
r_grid = np.arange(0, CAPACITY + STEP, STEP)

# Weibull distribution parameters
K_SHAPE = 2.0
L_SCALE = 10.0
weibull = weibull_min(fscale=L_SCALE, fshape=K_SHAPE)

# Precompute CDF and PDF for the grid
# We only care about items x where x <= r.
# We need to integrate P(X=x) * (1 + V(r-x)) over x from 0 to r.
# This is a convolution-like operation.

# Initialize V
V = np.zeros(GRID_SIZE)

# To speed up, we can precompute the probability mass on the grid
# We approximate the continuous distribution by discretizing the item sizes.
# Let's create a grid for item sizes too.
ITEM_GRID_SIZE = 2000
item_grid = np.arange(0, CAPACITY + STEP, STEP)
# Probability density at each point
pdf_vals = weibull.pdf(item_grid)
# Convert to probability mass by multiplying by step size
# Note: pdf is density, so P(item in [x, x+dx]) ≈ pdf(x) * dx
prob_mass = pdf_vals * STEP
# Normalize to ensure sum is 1 (approx)
prob_mass /= np.sum(prob_mass)

# DP Iteration
# V(r) = sum_{j} prob_mass[j] * (1 + V(r - item_grid[j])) for item_grid[j] <= r
# We iterate r from small to large.
for i in range(GRID_SIZE):
    r = r_grid[i]
    # Find all items that fit in r
    # item_grid[j] <= r
    # Since item_grid is sorted, we can find the cutoff index
    # max_j such that item_grid[max_j] <= r
    # We can use searchsorted
    max_j = np.searchsorted(item_grid, r, side='right') - 1
    if max_j < 0:
        V[i] = 0
        continue
    
    # Sum over j=0 to max_j
    # term = prob_mass[j] * (1 + V[r - item_grid[j]])
    # r - item_grid[j] is the remaining space after placing item j
    # We need to map this remaining space back to the grid index
    # rem_space = r - item_grid[j]
    # index = rem_space / STEP
    # Since r and item_grid are on the same grid, rem_space should also be on the grid (or close)
    # Let's vectorize this for speed
    
    # Get the relevant item sizes and probabilities
    items_fit = item_grid[:max_j+1]
    probs_fit = prob_mass[:max_j+1]
    
    # Calculate remaining space
    rem_spaces = r - items_fit
    
    # Map to grid indices
    # Since rem_spaces are multiples of STEP (approximately), we can divide
    # But floating point might be an issue.
    # Let's use the fact that index = i - j
    # r = i * STEP
    # item_grid[j] = j * STEP
    # rem = (i-j) * STEP
    # index = i - j
    
    indices = i - np.arange(max_j + 1)
    
    # Get V values
    # Note: indices are always >= 0 because j <= i (since item_grid[j] <= r)
    v_vals = V[indices]
    
    # Calculate expected value
    V[i] = np.sum(probs_fit * (1 + v_vals))

# Now V[i] represents the expected number of items that can be packed into space r_grid[i].
# The "waste" or "regret" of leaving space r is the capacity of that space minus the value of what can be packed.
# Actually, we want to minimize the total number of bins.
# Placing an item in a bin with remaining space r creates a new remaining space r' = r - item.
# We want to choose the bin that leads to the best future packing.
# The "value" of a bin with remaining space r is V[r] (expected items).
# When we place an item of size `item` in a bin with remaining space `bins[i]`, the new remaining space is `bins[i] - item`.
# The "loss" in value is V[bins[i]] - (1 + V[bins[i] - item]).
# Wait, the current bin already has V[bins[i]] potential.
# After placing the item, the bin has V[bins[i] - item] potential.
# But we also "used" one item.
# Let's think in terms of "opportunity cost".
# We want to maximize the total number of items packed.
# If we place the item in bin i, we gain 1 item packed now, and the bin's future value becomes V[bins[i] - item].
# The total value contributed by this bin is 1 + V[bins[i] - item].
# We want to maximize this.
# So the score for bin i should be V[bins[i] - item].
# Or more precisely, we want to minimize the "waste" of the current state.
# Let's stick to maximizing the future value of the bin.
# Score = V[bins[i] - item]

def get_v_value(space):
    """Interpolate V value for a given remaining space."""
    if space <= 0:
        return 0.0
    if space >= CAPACITY:
        return V[-1]
    # Find index
    idx = space / STEP
    i0 = int(idx)
    i1 = i0 + 1
    if i1 >= GRID_SIZE:
        return V[-1]
    # Linear interpolation
    frac = idx - i0
    return V[i0] * (1 - frac) + V[i1] * frac

# Vectorized version for speed
def get_v_values_vectorized(spaces):
    """Interpolate V values for an array of remaining spaces."""
    # Clip spaces to [0, CAPACITY]
    spaces = np.clip(spaces, 0, CAPACITY)
    # Calculate indices
    idx = spaces / STEP
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    # Clip indices to valid range
    i0 = np.clip(i0, 0, GRID_SIZE - 1)
    i1 = np.clip(i1, 0, GRID_SIZE - 1)
    # Interpolate
    frac = idx - i0
    return V[i0] * (1 - frac) + V[i1] * frac

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy:
    - Use a precomputed Value Function V(r) which estimates the expected number of future items
      that can be packed into a space of size r, given the Weibull distribution of item sizes.
    - For each bin with remaining capacity `bins[i]`, calculate the new remaining space `bins[i] - item`.
    - The score is the value of this new remaining space, V(bins[i] - item).
    - This effectively chooses the bin that maximizes the potential for future packing,
      thereby minimizing the expected total number of bins used.
    """
    # Start with a very low score for all bins
    scores = np.full(len(bins), -1e18)
    
    # Calculate remaining capacity in each bin after placing the item
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_indices]
    
    # Get the value for each valid remaining space
    # We use the vectorized interpolation
    v_values = get_v_values_vectorized(valid_remaining)
    
    # The score is the value of the remaining space
    # Higher value means better future packing potential
    scores[valid_indices] = v_values
    
    return scores
# EVOLVE-BLOCK-END
