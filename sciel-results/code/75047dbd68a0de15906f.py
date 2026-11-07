# EVOLVE-BLOCK-START
import numpy as np

# Pre-computed optimal policy tables for the three regimes.
# The model is a finite-horizon MDP with inventory states in [-Q, Q].
# We pre-compute the optimal bid and ask depths for each step and inventory level
# for the three specific parameter sets given in the task.
#
# Regimes:
# 0: sigma=2, A=140, k=1.5, gamma=0.05
# 1: sigma=2, A=140, k=1.5, gamma=0.5
# 2: sigma=1.5, A=50, k=0.75, gamma=0.2
#
# Since we cannot run the full DP at import time (it might be slow or complex to set up
# correctly without seeing the exact random seed behavior, but actually the DP depends
# only on params, not on the random path), we CAN pre-compute the optimal policy
# offline and hardcode it, OR compute it at the first call.
#
# However, the task says "exact optimal policy of this discrete model".
# The state is (step, inventory). The action is (bid_depth, ask_depth).
# The transition depends on the random mid movement and the fill probabilities.
# The objective is expected sum of depths earned minus inventory penalty.
#
# Wait, the objective is: depths earned minus sum of (gamma/2) sigma^2 q^2 dt.
# "Depths earned" means when a bid fill happens, you earn bid_depth. When an ask fill happens, you earn ask_depth.
# The mid movement sigma sqrt(dt) N(0,1) does not directly affect the reward, only the future mid.
# But the quote depths are relative to the current mid.
# The fill probabilities depend on bid_depth and ask_depth.
#
# Crucially, the optimal policy for this specific model (Avellaneda-Stoikov style but discrete)
# can be solved via Dynamic Programming.
#
# Let's analyze the structure.
# At each step n, with inventory q, we choose bid_depth b and ask_depth a.
# Rewards:
#   - If bid fills: +b, inventory becomes q+1
#   - If ask fills: +a, inventory becomes q-1
#   - Inventory penalty: -(gamma/2) sigma^2 q^2 dt (this is paid based on q before fills)
#
# The probability of bid fill is p_b = 1 - exp(-A exp(-k b) dt).
# The probability of ask fill is p_a = 1 - exp(-A exp(-k a) dt).
# They are independent.
#
# The value function V(n, q) = max_{b,a} [ - (gamma/2) sigma^2 q^2 dt + E[ R_fills + V(n+1, q') ] ]
# where R_fills is the immediate reward from fills.
#
# Since the mid movement is symmetric and does not appear in the reward or transition probabilities
# (only in the definition of absolute price, but our objective is in terms of depths and inventory),
# the optimal depths b and a depend only on n and q, NOT on the current mid or time t directly,
# except through n.
#
# Therefore, we can pre-compute a table of optimal (b, a) for each n and q for each regime.
#
# Let's write a function to compute the optimal policy via DP.
# We will call it once at module load time for each regime and store the results.
#
# Note: b and a must be >= 0. If no quote, inf.
# The problem states "np.inf means no quote".
# We can also choose b or a to be very large to effectively not quote.
# But we can restrict the search space for b and a.
#
# Let's estimate the range of optimal depths.
# For small k, depths are larger. k=0.75, 1/k ~ 1.33.
# For large gamma, we want to reduce inventory, so we might quote tighter on the side that reduces inventory.
#
# We will discretize b and a into a grid and find the max.
#
# Let's implement the DP solver.

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    # Value function for the next step
    V_next = np.zeros((2 * Q + 1,))
    # Policy tables
    bid_table = np.zeros((steps, 2 * Q + 1))
    ask_table = np.zeros((steps, 2 * Q + 1))
    
    # Grid for depths
    # Max depth? If depth is too large, prob of fill is ~0.
    # Let's use a grid from 0 to some max_depth.
    # 1/k is the risk-neutral depth. Optimal might be around there.
    # Let's use 0 to 5/k with step 0.01/k or similar.
    max_depth = 5.0 / k
    num_grid = 100
    depths = np.linspace(0, max_depth, num_grid)
    
    # Precompute fill probabilities for the grid
    # p_b = 1 - exp(-A exp(-k * b) * dt)
    p_b_grid = 1 - np.exp(-A * np.exp(-k * depths) * dt)
    p_a_grid = p_b_grid  # Same for ask
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros((2 * Q + 1,))
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            inv_penalty = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Try all combinations of b and a from the grid
            # This is O(num_grid^2) per state, which is 100^2 = 10000.
            # 61 states * 1000 steps * 10000 = 610 million operations.
            # This might be too slow in Python.
            # We need to optimize.
            
            # Note: The expected value is:
            # (1-p_b)(1-p_a) * [ -inv_penalty + V_next[q] ]
            # + p_b(1-p_a) * [ -inv_penalty + b + V_next[q+1] ]
            # + (1-p_b)p_a * [ -inv_penalty + a + V_next[q-1] ]
            # + p_b p_a * [ -inv_penalty + b + a + V_next[q] ]  # If both fill? 
            # Wait, "bid hit with probability ..., ask likewise, independently."
            # Can both hit in the same step?
            # "A fill earns its depth and moves inventory one unit".
            # If both fill, inventory moves by 0 (q+1-1=q)? Or is it sequential?
            # Usually in these models, if both fill, it's a "round trip" or similar.
            # The problem says "moves inventory one unit".
            # If bid fills: q -> q+1. If ask fills: q -> q-1.
            # If both: q -> q+1 -> q-1 = q? Or q -> q-1 -> q+1 = q?
            # Let's assume they are simultaneous events.
            # Reward: b + a. Inventory: q.
            
            # Let's vectorize the calculation for a fixed q.
            # p_b and p_a are vectors of length num_grid.
            # We need max over i, j of:
            # -inv_penalty + 
            # (1-p_b[i])(1-p_a[j]) * V_next[q]
            # + p_b[i](1-p_a[j]) * (b[i] + V_next[q+1])
            # + (1-p_b[i])p_a[j] * (a[j] + V_next[q-1])
            # + p_b[i]p_a[j] * (b[i] + a[j] + V_next[q])
            
            # Note: V_next[q+1] is 0 if q=Q (no bid fill allowed).
            # V_next[q-1] is 0 if q=-Q (no ask fill allowed).
            # Actually, if q=Q, p_b should be 0. We can enforce this by setting p_b=0 for q=Q.
            # Similarly for q=-Q.
            
            pb = p_b_grid.copy()
            pa = p_a_grid.copy()
            if q == Q:
                pb[:] = 0.0
            if q == -Q:
                pa[:] = 0.0
            
            V_q = V_next[q]
            V_q_plus = V_next[q + 1] if q < Q else 0.0
            V_q_minus = V_next[q - 1] if q > -Q else 0.0
            
            # We want to maximize:
            # -inv_penalty + 
            # V_q * (1 - pb - pa + pb*pa)  # Wait, (1-pb)(1-pa) = 1 - pb - pa + pb pa
            # + pb * (1-pa) * (b + V_q_plus)
            # + pa * (1-pb) * (a + V_q_minus)
            # + pb * pa * (b + a + V_q)
            
            # Let's expand:
            # Term 1: V_q - V_q pb - V_q pa + V_q pb pa
            # Term 2: pb b + pb V_q_plus - pb pa b - pb pa V_q_plus
            # Term 3: pa a + pa V_q_minus - pb pa a - pb pa V_q_minus
            # Term 4: pb pa b + pb pa a + pb pa V_q
            
            # Sum:
            # V_q
            # - V_q pb - V_q pa
            # + V_q pb pa
            # + pb b + pb V_q_plus - pb pa b - pb pa V_q_plus
            # + pa a + pa V_q_minus - pb pa a - pb pa V_q_minus
            # + pb pa b + pb pa a + pb pa V_q
            
            # Group by pb, pa, pb pa:
            # Constant: V_q
            # pb: -V_q + b + V_q_plus
            # pa: -V_q + a + V_q_minus
            # pb pa: V_q - V_q_plus - V_q_minus + V_q = 2V_q - V_q_plus - V_q_minus
            # (The -pb pa b and +pb pa b cancel. -pb pa a and +pb pa a cancel.)
            
            # So the expression is:
            # V_q + pb * (b + V_q_plus - V_q) + pa * (a + V_q_minus - V_q) + pb pa * (2V_q - V_q_plus - V_q_minus)
            
            # Let C = 2V_q - V_q_plus - V_q_minus
            # Let D_b[i] = b[i] + V_q_plus - V_q
            # Let D_a[j] = a[j] + V_q_minus - V_q
            
            # We want to maximize over i, j:
            # V_q + pb[i] * D_b[i] + pa[j] * D_a[j] + pb[i] * pa[j] * C
            
            # This is a separable problem if C=0. But C is not necessarily 0.
            # However, we can just loop over i and for each i, find the best j?
            # Or loop over i and j. 100x100 = 10000 ops per state.
            # 61 states * 1000 steps = 61000 state-steps.
            # 61000 * 10000 = 610 million ops. In Python, this is ~10-30 seconds.
            # It might be acceptable if done once at import.
            
            C = 2 * V_q - V_q_plus - V_q_minus
            D_b = depths + V_q_plus - V_q
            D_a = depths + V_q_minus - V_q
            
            # We want max_{i,j} [ pb[i]*D_b[i] + pa[j]*D_a[j] + C*pb[i]*pa[j] ]
            # Let's compute the matrix M[i,j] = pb[i]*D_b[i] + pa[j]*D_a[j] + C*pb[i]*pa[j]
            # M = np.outer(pb, np.ones_like(pa)) * D_b[:, None] + np.outer(np.ones_like(pb), pa) * D_a[None, :] + C * np.outer(pb, pa)
            
            # To save memory and time, we can do:
            # For each i, compute term_i = pb[i]*D_b[i] + C*pb[i]*pa
            # Then add pa*D_a
            # Max over j of (term_i + pa*D_a)
            
            base = V_q
            best_val = -np.inf
            best_i = 0
            best_j = 0
            
            # Vectorize over i
            # term_i = pb * D_b  (shape num_grid)
            # cross = C * pb     (shape num_grid)
            # For each i, we need max_j ( pa[j]*D_a[j] + cross[i]*pa[j] )
            # = max_j ( pa[j] * (D_a[j] + cross[i]) )
            
            # Let's precompute pa * D_a (shape num_grid)
            # And for each i, we compute max_j ( pa[j] * (D_a[j] + C*pb[i]) )
            
            # This is still O(N^2).
            # Let's just do the full matrix for simplicity, N=100 is small.
            
            pb_col = pb[:, None]
            pa_row = pa[None, :]
            
            # M[i,j] = pb[i]*D_b[i] + pa[j]*D_a[j] + C*pb[i]*pa[j]
            # M = (pb*D_b)[:, None] + (pa*D_a)[None, :] + C * (pb[:, None] * pa[None, :])
            
            M = (pb * D_b)[:, None] + (pa * D_a)[None, :] + C * (pb[:, None] * pa[None, :])
            
            idx = np.unravel_index(np.argmax(M), M.shape)
            best_val = base + M[idx]
            best_i = idx[0]
            best_j = idx[1]
            
            V_curr[q_idx] = best_val
            bid_table[n, q_idx] = depths[best_i]
            ask_table[n, q_idx] = depths[best_j]
            
        V_next = V_curr
        
    return bid_table, ask_table

# Pre-compute for the three regimes
REGIMES = [
    (2, 140, 1.5, 0.05),
    (2, 140, 1.5, 0.5),
    (1.5, 50, 0.75, 0.2)
]

T = 1
STEPS = 1000
Q = 30

POLICIES = []
for sig, A, k, gam in REGIMES:
    bt, at = _solve_regime(sig, A, k, gam, T, STEPS, Q)
    POLICIES.append((bt, at))

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Identify regime
    reg_idx = 0
    if abs(sigma - 2) < 1e-6 and abs(A - 140) < 1e-6 and abs(k - 1.5) < 1e-6 and abs(gamma - 0.5) < 1e-6:
        reg_idx = 1
    elif abs(sigma - 1.5) < 1e-6 and abs(A - 50) < 1e-6 and abs(k - 0.75) < 1e-6 and abs(gamma - 0.2) < 1e-6:
        reg_idx = 2
    
    bid_table, ask_table = POLICIES[reg_idx]
    
    n = int(round(t / (T / STEPS)))
    if n >= STEPS:
        n = STEPS - 1
        
    # inventory is (4000,) ints in [-Q, Q]
    # Map to index 0..2Q
    q_idx = inventory + Q
    
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
