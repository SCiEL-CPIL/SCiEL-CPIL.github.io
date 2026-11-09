"""Optimal market making with inventory risk using HJB discretization."""
# EVOLVE-BLOCK-START
import numpy as np

# Precomputed constants for the three regimes
# Regimes: (sigma, A, k, gamma)
REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

T = 1.0
STEPS = 1000
DT = T / STEPS
Q_MAX = 30
N_INVENTORY = 2 * Q_MAX + 1  # 61 states: -30 to 30

# We need to precompute optimal policies for each regime.
# Since the problem is time-homogeneous in the sense that the optimal depth
# depends on (t, q), we can precompute a table for each regime.
# However, t has 1000 steps, so we need a table of shape (STEPS, N_INVENTORY) for each regime.
# But wait, the optimal depth might depend on t only through the remaining time.
# Let's use backward induction from t=T to t=0.

# Actually, let's think about the HJB equation.
# V(t, q) = max over bid_depth, ask_depth of:
#   (1 - exp(-A * exp(-k * bid_depth) * DT)) * (bid_depth - V(t+DT, q+1)) +
#   (1 - exp(-A * exp(-k * ask_depth) * DT)) * (-ask_depth - V(t+DT, q-1)) +
#   (1 - (1 - exp(-A * exp(-k * bid_depth) * DT)) - (1 - exp(-A * exp(-k * ask_depth) * DT))) * V(t+DT, q) -
#   (gamma / 2) * sigma^2 * q^2 * DT

# Wait, the objective is: depths earned minus sum over steps of (gamma/2) sigma^2 q^2 dt.
# When a bid fill occurs, we earn bid_depth (the depth we quoted) and inventory increases by 1.
# When an ask fill occurs, we earn -ask_depth (since ask_depth is distance above mid, and we sell, so we get mid - ask_depth, but the "earn" is relative to mid? Let me re-read.)

# "A fill earns its depth and moves inventory one unit"
# Bid fill: we buy at mid - bid_depth, so we earn bid_depth (profit relative to mid if we later sell at mid).
# Ask fill: we sell at mid + ask_depth, so we earn ask_depth (profit relative to mid).
# Wait, the problem says "depths earned". For a bid, the depth is bid_depth. For an ask, the depth is ask_depth.
# So if a bid fills, we get +bid_depth. If an ask fills, we get +ask_depth.
# Inventory: bid fill increases inventory by 1 (we bought), ask fill decreases inventory by 1 (we sold).

# Let's define:
# p_bid = 1 - exp(-A * exp(-k * bid_depth) * DT)
# p_ask = 1 - exp(-A * exp(-k * ask_depth) * DT)

# Expected reward at step t:
# p_bid * bid_depth + p_ask * ask_depth - (gamma/2) * sigma^2 * q^2 * DT
# Then transition to next state:
# With prob p_bid: state q+1, value V(t+DT, q+1)
# With prob p_ask: state q-1, value V(t+DT, q-1)
# With prob 1-p_bid-p_ask: state q, value V(t+DT, q)

# So V(t, q) = max_{bid_depth, ask_depth} [
#   p_bid * bid_depth + p_ask * ask_depth - (gamma/2) * sigma^2 * q^2 * DT +
#   p_bid * V(t+DT, q+1) + p_ask * V(t+DT, q-1) + (1-p_bid-p_ask) * V(t+DT, q)
# ]

# This is a 2D optimization over bid_depth and ask_depth for each (t, q).
# We can solve this numerically by discretizing the depth space.

# Let's precompute the optimal policy table for each regime.
# Depth range: from 0 to some max_depth. Let's say max_depth = 10.
# Number of depth levels: let's use 200 levels for accuracy.

MAX_DEPTH = 10.0
N_DEPTH = 200
DEPTH_GRID = np.linspace(0, MAX_DEPTH, N_DEPTH)

def compute_optimal_policy(sigma, A, k, gamma):
    """Compute optimal bid and ask depths for all (t, q) using backward induction."""
    # V[t, q_index] where q_index = q + Q_MAX
    # t ranges from 0 to STEPS-1, V[STEPS] = 0
    V = np.zeros((STEPS + 1, N_INVENTORY))
    bid_opt = np.zeros((STEPS, N_INVENTORY))
    ask_opt = np.zeros((STEPS, N_INVENTORY))
    
    for t in range(STEPS - 1, -1, -1):
        for q in range(-Q_MAX, Q_MAX + 1):
            q_idx = q + Q_MAX
            # If at boundary, we can't quote on that side
            can_bid = (q < Q_MAX)
            can_ask = (q > -Q_MAX)
            
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            # We need to optimize over bid_depth and ask_depth
            # For efficiency, we can do a grid search
            # But 200x200 = 40000 evaluations per (t,q) is too slow.
            # Let's use a smaller grid or a smarter approach.
            
            # Actually, let's use a 1D grid for each and do a 2D search with fewer points.
            # Or we can observe that the optimal depth might be separable.
            
            # Let's try a grid of 50x50 for speed.
            depth_grid_small = np.linspace(0, MAX_DEPTH, 50)
            
            for bid_d in depth_grid_small:
                if not can_bid:
                    p_bid = 0.0
                else:
                    p_bid = 1 - np.exp(-A * np.exp(-k * bid_d) * DT)
                
                for ask_d in depth_grid_small:
                    if not can_ask:
                        p_ask = 0.0
                    else:
                        p_ask = 1 - np.exp(-A * np.exp(-k * ask_d) * DT)
                    
                    # Expected immediate reward
                    reward = p_bid * bid_d + p_ask * ask_d - (gamma / 2.0) * sigma**2 * q**2 * DT
                    
                    # Transition values
                    val_next = V[t + 1, q_idx]
                    if q < Q_MAX:
                        val_next += p_bid * (V[t + 1, q_idx + 1] - V[t + 1, q_idx])
                    if q > -Q_MAX:
                        val_next += p_ask * (V[t + 1, q_idx - 1] - V[t + 1, q_idx])
                    
                    total = reward + val_next
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            V[t, q_idx] = best_val
            bid_opt[t, q_idx] = best_bid
            ask_opt[t, q_idx] = best_ask
    
    return bid_opt, ask_opt

# Precompute policies for all regimes
# This is slow, so we do it once at module load time.
POLICIES = []
for sigma, A, k, gamma in REGIMES:
    bid_opt, ask_opt = compute_optimal_policy(sigma, A, k, gamma)
    POLICIES.append((bid_opt, ask_opt))

def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths based on precomputed policy."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    
    # Find the matching regime
    regime_idx = -1
    for i, (s, a, kk, g) in enumerate(REGIMES):
        if abs(sigma - s) < 1e-9 and abs(A - a) < 1e-9 and abs(k - kk) < 1e-9 and abs(gamma - g) < 1e-9:
            regime_idx = i
            break
    
    if regime_idx == -1:
        # Fallback: use risk-neutral depth
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    bid_opt, ask_opt = POLICIES[regime_idx]
    
    # t is n * DT, so step index n = round(t / DT)
    n = int(round(t / DT))
    if n >= STEPS:
        n = STEPS - 1
    
    # Get depths for each inventory level
    q_idx = inventory + Q_MAX
    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]
    
    # Handle boundaries: if at Q, no bid; if at -Q, no ask
    bid_depths = np.where(inventory == Q_MAX, np.inf, bid_depths)
    ask_depths = np.where(inventory == -Q_MAX, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
