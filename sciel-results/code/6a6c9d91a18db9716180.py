"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We will pre-compute the optimal policy using Dynamic Programming (Bellman Equation)
# for each regime. The state is (step, inventory). The action is (bid_depth, ask_depth).
# Since the action space is continuous, we need to discretize it or find the optimal
# depth analytically.

# Let's analyze the Bellman equation:
# V_n(q) = max_{b, a} { (1 - exp(-A exp(-k b) dt)) * (b - V_{n+1}(q+1)) 
#                       + (1 - exp(-A exp(-k a) dt)) * (a - V_{n+1}(q-1))
#                       - (gamma / 2) * sigma^2 * q^2 * dt
#                       + exp(-A exp(-k b) dt) * exp(-A exp(-k a) dt) * V_{n+1}(q)
#                       + ... terms for only bid fill, only ask fill, no fill }
#
# Wait, the inventory changes:
# If bid fill: q -> q+1 (if q < Q)
# If ask fill: q -> q-1 (if q > -Q)
# If both: q -> q (one +1, one -1)
# If none: q -> q
#
# The objective is: depths earned minus sum over steps of (gamma / 2) sigma^2 q^2 dt
# "depths earned" means if bid is hit, we earn b (the depth below mid). If ask is hit, we earn a (depth above mid).
# Wait, the problem says "depths earned". Usually market making earns the spread.
# Let's re-read: "A fill earns its depth".
# So if bid at mid - b is hit, we buy at mid - b. The "depth earned" is b?
# Or is it the profit relative to mid?
# "bid_depths, ask_depths ... distances below and above mid to quote this step"
# "A fill earns its depth" -> if bid is hit, we earn b. If ask is hit, we earn a.
# This implies the reward is simply b + a if both hit, b if only bid, a if only ask.
# And the cost is (gamma / 2) sigma^2 q^2 dt.
#
# Let p_b = 1 - exp(-A exp(-k b) dt)
# Let p_a = 1 - exp(-A exp(-k a) dt)
#
# Expected reward at step n:
# R = p_b * (1 - p_a) * (b - V_{n+1}(q+1))  # bid only
#   + p_a * (1 - p_b) * (a - V_{n+1}(q-1))  # ask only
#   + p_b * p_a * (b + a - V_{n+1}(q))      # both
#   + (1 - p_b) * (1 - p_a) * (0 - V_{n+1}(q)) # none
#
# Wait, the value function V_n(q) should include the future values.
# V_n(q) = max_{b,a} { E[reward_n + V_{n+1}(q')] } - (gamma/2) sigma^2 q^2 dt
#
# Let's simplify the expected value:
# E[V_{n+1}] = (1-p_b)(1-p_a) V_{n+1}(q) + p_b(1-p_a) V_{n+1}(q+1) + (1-p_b)p_a V_{n+1}(q-1) + p_b p_a V_{n+1}(q)
#            = V_{n+1}(q) + p_b(1-p_a)(V_{n+1}(q+1) - V_{n+1}(q)) + (1-p_b)p_a(V_{n+1}(q-1) - V_{n+1}(q))
#
# E[reward] = p_b(1-p_a)b + (1-p_b)p_a a + p_b p_a (b+a)
#           = p_b b + p_a a
#
# So:
# V_n(q) = max_{b,a} { p_b b + p_a a + V_{n+1}(q) + p_b(1-p_a)(V_{n+1}(q+1) - V_{n+1}(q)) + (1-p_b)p_a(V_{n+1}(q-1) - V_{n+1}(q)) } - (gamma/2) sigma^2 q^2 dt
#
# This separates into a term depending only on b and a term depending only on a!
# max_{b} { p_b b + p_b(1-p_a)(V_{n+1}(q+1) - V_{n+1}(q)) }
# max_{a} { p_a a + (1-p_b)p_a(V_{n+1}(q-1) - V_{n+1}(q)) }
#
# Wait, p_a depends on a, and the first term has (1-p_a). So they are coupled through p_a and p_b.
# However, if we fix p_a, the optimal b can be found. If we fix p_b, the optimal a can be found.
# Since b and a are independent actions (except through the coupling in the probability terms),
# we can solve for the optimal b and a by finding the fixed point or by noting that
# for small dt, p_b and p_a are small, so 1-p_a ≈ 1 and 1-p_b ≈ 1.
#
# Let's assume dt is small enough that p_b, p_a << 1.
# Then:
# max_b { p_b b + p_b (V_{n+1}(q+1) - V_{n+1}(q)) } = p_b (b + V_{n+1}(q+1) - V_{n+1}(q))
# max_a { p_a a + p_a (V_{n+1}(q-1) - V_{n+1}(q)) } = p_a (a + V_{n+1}(q-1) - V_{n+1}(q))
#
# Let D_b = V_{n+1}(q+1) - V_{n+1}(q)
# Let D_a = V_{n+1}(q-1) - V_{n+1}(q)
#
# We want to maximize p_b (b + D_b) + p_a (a + D_a)
# where p_b = 1 - exp(-A exp(-k b) dt)
#
# Let f(b) = (1 - exp(-A exp(-k b) dt)) * (b + D_b)
# We want to maximize f(b).
# f'(b) = -exp(-A exp(-k b) dt) * A * exp(-k b) * (-k) * dt * (b + D_b) + (1 - exp(-A exp(-k b) dt))
# f'(b) = k A dt exp(-k b) exp(-A exp(-k b) dt) (b + D_b) + 1 - exp(-A exp(-k b) dt)
#
# This is a 1D optimization problem. We can solve it numerically for each q and n.
# Since q is in [-30, 30], there are 61 states. n is in [0, 1000].
# 61 * 1000 = 61000 optimizations. Each optimization is a 1D root finding or grid search.
# This is feasible.
#
# Let's implement this DP.
# We need to be careful with the boundary conditions: no bid fill at q=Q, no ask fill at q=-Q.
# At q=Q, we cannot increase inventory, so we should not quote a bid (or b=inf).
# At q=-Q, we cannot decrease inventory, so we should not quote an ask (or a=inf).
#
# We'll precompute the optimal b and a for each (n, q).
# We'll store them in arrays: opt_b[n, q+Q], opt_a[n, q+Q].
#
# Time budget: 330s. 61000 * (cost of 1D opt).
# If 1D opt takes 100us, total is 6.1s. This is fine.
#
# Let's refine the 1D optimization.
# We want to maximize g(x) = (1 - exp(-A exp(-k x) dt)) * (x + D)
# where x is the depth (b or a), and D is the inventory shift in value.
#
# Note: if D is very negative, we might want x=0 (no quote) or even negative?
# But x must be > 0 (or >= 0). The problem says "distances below and above mid", so x >= 0.
# Also, "Values below -3/k, -inf, NaN or a wrong shape make the whole result invalid."
# So x must be >= -3/k? No, "distances below ... to quote" implies positive distance.
# But the constraint is "Values below -3/k ... invalid". So x can be as low as -3/k?
# Wait, "distances below and above mid to quote this step". A distance is usually positive.
# But if x is negative, it means quoting above mid for bid? That would be a cross.
# The problem says "np.inf means no quote".
# Let's assume x >= 0. If the optimal x is < 0, we set it to 0 or inf?
# Actually, if x=0, p = 1 - exp(-A dt) which is high.
# If D is very negative, we don't want to fill. So we want p to be small, i.e., x large.
# So x will be positive.
#
# Let's use scipy.optimize.minimize_scalar or a simple grid + Newton.
# Since we have many calls, a precomputed grid for x might be faster.
# x range: [0, X_max]. What is X_max?
# If x is large, p ≈ 0, g(x) ≈ 0.
# If x=0, p = 1 - exp(-A dt), g(0) = p * D.
# If D > 0, we want to fill, so x small.
# If D < 0, we want to avoid fill, so x large.
# But g(x) -> 0 as x -> inf.
# If D < 0, g(0) < 0, and g(inf) = 0. So the max is at x=inf (no quote)?
# Wait, if we don't quote, we get 0 reward and stay at q.
# If we quote at x, we get p(x)(x+D).
# If D < 0, and x is small, x+D might be negative.
# We want to maximize p(x)(x+D).
# If D < 0, we can choose x such that x+D > 0, i.e., x > -D.
# Then g(x) > 0.
# So even if D < 0, we might want to quote at a large x to earn a small positive amount.
#
# Let's find the maximum of g(x) for x in [0, inf].
# g(x) = (1 - exp(-A exp(-k x) dt)) (x + D)
#
# We can solve g'(x) = 0.
# Let's use a grid search for x in [0, 10] and then refine.
# 10 is a reasonable upper bound since exp(-k*10) is small for k>=0.75.
#
# We will precompute a grid of x values and the corresponding p(x).
# Then for each (n, q), we compute D_b and D_a, and find the best x on the grid.
#
# Let's implement this.

_policy_cache = {}

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid and ask depths for all (step, inventory) states."""
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    
    # Value function: V[n, q] for n in [0, n_steps], q in [0, n_inv-1]
    # V[n_steps, q] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_inv))
    
    # Optimal depths
    opt_b = np.zeros((n_steps, n_inv))
    opt_a = np.zeros((n_steps, n_inv))
    
    # Precompute grid for depth x
    # x in [0, 20]
    x_grid = np.linspace(0, 20, 4001)
    p_grid = 1 - np.exp(-A * np.exp(-k * x_grid) * dt)
    
    for n in range(n_steps - 1, -1, -1):
        for i in range(n_inv):
            q = i - Q
            
            # Inventory shift in value
            # D_b = V[n+1, q+1] - V[n+1, q]  (if q < Q)
            # D_a = V[n+1, q-1] - V[n+1, q]  (if q > -Q)
            
            D_b = 0.0
            if q < Q:
                D_b = V[n+1, i+1] - V[n+1, i]
            else:
                # No bid fill allowed, so we set b = inf (no quote)
                opt_b[n, i] = np.inf
                D_b = 0.0  # Won't be used if b=inf
            
            D_a = 0.0
            if q > -Q:
                D_a = V[n+1, i-1] - V[n+1, i]
            else:
                # No ask fill allowed, so we set a = inf (no quote)
                opt_a[n, i] = np.inf
                D_a = 0.0
            
            # Find optimal b
            if q < Q:
                # Maximize p_grid * (x_grid + D_b)
                vals_b = p_grid * (x_grid + D_b)
                idx_b = np.argmax(vals_b)
                opt_b[n, i] = x_grid[idx_b]
                
            # Find optimal a
            if q > -Q:
                vals_a = p_grid * (x_grid + D_a)
                idx_a = np.argmax(vals_a)
                opt_a[n, i] = x_grid[idx_a]
            
            # Compute V[n, i]
            b = opt_b[n, i]
            a = opt_a[n, i]
            
            p_b = 1 - np.exp(-A * np.exp(-k * b) * dt) if np.isfinite(b) else 0.0
            p_a = 1 - np.exp(-A * np.exp(-k * a) * dt) if np.isfinite(a) else 0.0
            
            # Expected reward
            reward = p_b * b + p_a * a
            
            # Expected next value
            # V_next = (1-p_b)(1-p_a) V[i] + p_b(1-p_a) V[i+1] + (1-p_b)p_a V[i-1] + p_b p_a V[i]
            #        = V[i] + p_b(1-p_a)(V[i+1]-V[i]) + (1-p_b)p_a(V[i-1]-V[i])
            
            V_next = V[n+1, i]
            if q < Q:
                V_next += p_b * (1 - p_a) * (V[n+1, i+1] - V[n+1, i])
            if q > -Q:
                V_next += (1 - p_b) * p_a * (V[n+1, i-1] - V[n+1, i])
            
            cost = (gamma / 2) * sigma**2 * q**2 * dt
            V[n, i] = reward + V_next - cost
    
    return opt_b, opt_a


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _policy_cache[key]
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    
    # inventory is an array of ints in [-Q, Q]
    # Convert to index: i = q + Q
    idx = inventory + Q
    
    bid_depths = opt_b[n, idx]
    ask_depths = opt_a[n, idx]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
