# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 270.0
_POLICY_CACHE = {}


def _compute_policy(params):
    """
    Computes the optimal bid and ask depths for each inventory level and time step
    using a backward induction (HJB discretization) approach.
    
    Returns:
        bid_depths: array of shape (steps, Q+1) where Q is max inventory
        ask_depths: array of shape (steps, Q+1)
    """
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps
    
    # Inventory range is [-Q, Q]
    # We index inventory q by i = q + Q, so i in [0, 2Q]
    num_q = 2 * Q + 1
    
    # V[i] represents the optimal expected future reward starting from inventory q = i - Q
    # We solve backwards from step steps-1 to 0.
    # V_next is the value function for the next time step.
    V_next = np.zeros(num_q)
    
    # Store optimal depths for each step and inventory
    # bid_depths[step, i] is the optimal bid depth for inventory q = i - Q at time step 'step'
    bid_depths = np.zeros((steps, num_q))
    ask_depths = np.zeros((steps, num_q))
    
    # Precompute exp(-k * x) for a grid of x to speed up optimization?
    # Or just use scalar optimization.
    
    # We need to solve for optimal x >= 0 maximizing:
    # (1 - exp(-A * exp(-k*x))) * (x + C)
    # where C is the value difference.
    
    # Let f(x) = (1 - exp(-A * exp(-k*x))) * (x + C)
    # f'(x) = A * exp(-k*x) * exp(-A * exp(-k*x)) * (x + C) + (1 - exp(-A * exp(-k*x)))
    # Set f'(x) = 0.
    
    # We can use scipy.optimize.minimize_scalar on -f(x) or find root of f'(x).
    # Given the time budget, we might precompute a lookup table for optimal x given C.
    # C can range roughly from -max_depth to +max_depth.
    # Max depth is bounded. Let's say x in [0, 10].
    # C range: V(q+1) - V(q). V is roughly -gamma/2 * sigma^2 * q^2 * T_remaining.
    # So C is roughly -gamma * sigma^2 * q * T_remaining.
    # For q=30, T=1, sigma=2, gamma=0.5: C ~ -0.5 * 4 * 30 = -60.
    # So C can be large negative.
    
    # If C is very negative, we want to minimize fill probability to avoid losing money?
    # Wait, if C < 0, (x+C) is negative.
    # f(x) = p(x) * (x + C).
    # If C < -x, then x+C < 0.
    # As x -> infinity, p(x) -> 1, f(x) -> x + C -> infinity.
    # This implies if C < 0, we should quote at infinite depth?
    # No, the model says "distances below and above mid to quote this step".
    # If we quote at infinite depth, we never get filled?
    # p(x) = 1 - exp(-A exp(-k x)). As x -> inf, exp(-k x) -> 0, p(x) -> 0.
    # Ah, I made a mistake in the limit.
    # As x -> 0, exp(-k x) -> 1, p(x) -> 1 - exp(-A).
    # As x -> inf, exp(-k x) -> 0, p(x) -> 0.
    
    # So p(x) is decreasing in x.
    # f(x) = p(x) (x + C).
    
    # If C > 0:
    # x=0: f(0) = (1-exp(-A)) * C > 0.
    # x->inf: f(x) -> 0 * inf. Limit is 0?
    # Let's check derivative.
    # f'(x) = p'(x)(x+C) + p(x).
    # p'(x) = -A exp(-kx) exp(-A exp(-kx)) < 0.
    # If C is large positive, we want small x to get high p.
    # If C is large negative, we want large x to make p small, so p(x)(x+C) is small negative (close to 0).
    # Actually, if C < 0, x+C can be negative.
    # We want to maximize f(x).
    # If we can make f(x) close to 0 from below, that's better than a large negative number.
    # As x -> inf, p(x) ~ A exp(-kx).
    # f(x) ~ A exp(-kx) (x+C) -> 0 from below (if x+C > 0 eventually? No, if C is fixed, x+C is positive for large x).
    # Wait, if C = -10, x=100, x+C=90. p(x) is tiny. f(x) is tiny negative.
    # At x=0, f(0) = (1-exp(-A)) * (-10) = large negative.
    # So for C < 0, we want large x.
    
    # Is there an upper bound on x?
    # The task says "Values below -3/k ... make the whole result invalid".
    # It doesn't explicitly bound x from above, but "distances" should be reasonable.
    # However, if x is very large, p(x) is effectively 0.
    # We can cap x at some value where p(x) is negligible, e.g., 1e-6.
    # A exp(-k x) = 1e-6 => exp(-k x) = 1e-6 / A => x = (1/k) ln(A / 1e-6).
    # For A=140, k=1.5, x = (1/1.5) ln(1.4e8) ~ 0.66 * 18.7 ~ 12.
    # So x in [0, 15] should be enough.
    
    # Let's precompute a lookup table for optimal x given C.
    # C range: let's sample C from -50 to 50.
    # Number of points: 1000.
    
    c_min = -50.0
    c_max = 50.0
    num_c = 1000
    c_grid = np.linspace(c_min, c_max, num_c)
    optimal_x_grid = np.zeros(num_c)
    
    # Define the objective to maximize
    def obj(x, C):
        p = 1.0 - np.exp(-A * np.exp(-k * x))
        return p * (x + C)
    
    # We will solve for each C in c_grid
    # Use scipy.optimize.minimize_scalar on -obj
    from scipy.optimize import minimize_scalar
    
    for i, C in enumerate(c_grid):
        res = minimize_scalar(lambda x: -obj(x, C), bounds=(0, 15), method='bounded')
        optimal_x_grid[i] = res.x
        
    # Now, for each step and inventory, we compute C and look up x.
    # V is a function of q.
    # V_next is known.
    # We need to compute V_current.
    # V_current(q) = max_{bid, ask} [ ... ]
    # This is coupled because bid and ask decisions are made simultaneously?
    # "bid hit with probability ..., ask likewise, independently."
    # The objective is "depths earned minus sum ...".
    # The decision at step n is bid_depth and ask_depth.
    # The reward is:
    # If bid hits: +bid_depth. Inventory q -> q+1.
    # If ask hits: +ask_depth. Inventory q -> q-1.
    # If both hit: +bid_depth + ask_depth. Inventory q -> q (bid then ask? or ask then bid? "moves inventory one unit" for each fill. If both, net 0 change? Or +1 then -1? Order doesn't matter for net change if both occur. But does it matter for value? V(q+1-1) = V(q).)
    # Wait, "A fill earns its depth and moves inventory one unit".
    # If both hit, we earn bid_depth + ask_depth. Inventory change is +1 (from bid) and -1 (from ask). Net 0.
    # So the state transition is:
    # P(bid only) = p_b (1 - p_a) -> q+1
    # P(ask only) = (1 - p_b) p_a -> q-1
    # P(both) = p_b p_a -> q
    # P(neither) = (1-p_b)(1-p_a) -> q
    
    # Expected reward = p_b * bid_depth + p_a * ask_depth + p_b(1-p_a) V(q+1) + (1-p_b)p_a V(q-1) + (p_b p_a + (1-p_b)(1-p_a)) V(q)
    # = p_b bid_depth + p_a ask_depth + V(q) + p_b(1-p_a)(V(q+1)-V(q)) + (1-p_b)p_a(V(q-1)-V(q))
    # = V(q) + p_b [bid_depth + (1-p_a) DeltaV_plus] + p_a [ask_depth + (1-p_b) DeltaV_minus]
    # where DeltaV_plus = V(q+1) - V(q)
    # DeltaV_minus = V(q-1) - V(q)
    
    # Note that p_a depends on ask_depth, p_b on bid_depth.
    # This is a joint optimization.
    # However, often in market making, the spread is wide enough that fills are rare, or we can approximate independence or solve jointly.
    # Given the "Dual-Variable" hint, maybe we can solve for optimal bid and ask separately if we assume the other side's fill probability is small or fixed?
    # Or we can do a 2D optimization.
    # 2D optimization for 1000 steps * 61 inventory levels = 61,000 optimizations.
    # Each optimization is fast, but 61,000 * (cost of 2D opt) might be too slow.
    
    # Let's check the cost.
    # If we use the precomputed 1D lookup, we need to handle the coupling.
    # Approximation: Assume p_a and p_b are small, so (1-p_a) ~ 1 and (1-p_b) ~ 1.
    # Then reward ~ V(q) + p_b [bid_depth + DeltaV_plus] + p_a [ask_depth + DeltaV_minus].
    # This decouples bid and ask.
    # Optimal bid depth maximizes p_b(x_b) (x_b + DeltaV_plus).
    # Optimal ask depth maximizes p_a(x_a) (x_a + DeltaV_minus).
    # This is exactly what the 1D lookup does.
    
    # Is the approximation good?
    # A=140, k=1.5.
    # If x=1/k=0.66, p = 1 - exp(-140 * exp(-1)) = 1 - exp(-51) ~ 1.
    # So fills are very likely if we quote tight.
    # If we quote wide, p is small.
    # The optimal x will balance depth earned vs inventory risk.
    # If inventory risk is high (DeltaV_plus is negative for bid if we are long? No, DeltaV_plus = V(q+1)-V(q).
    # V(q) is concave in q (penalty -q^2). So V(q+1) < V(q) for q>0?
    # V(q) ~ -gamma/2 sigma^2 q^2.
    # V(q+1) - V(q) ~ -gamma sigma^2 q.
    # So if q > 0, DeltaV_plus < 0. We don't want to buy more.
    # So we quote bid far away (large x_b) to reduce p_b.
    # If q < 0, DeltaV_plus > 0. We want to buy.
    
    # With the approximation, we can use the 1D lookup.
    # Let's try this first. It's much faster.
    
    # We need to map C to x.
    # We have c_grid and optimal_x_grid.
    # We can use np.interp.
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Compute V_current
        V_current = np.zeros(num_q)
        for i in range(num_q):
            q = i - Q
            
            # Boundary conditions
            # No bid fill at q = Q. So we must not quote bid? Or bid depth = inf?
            # "no bid fill at q = Q".
            # If we quote bid, p_b > 0, but fill is blocked?
            # "no bid fill at q = Q" implies the probability of fill is 0, or the transition doesn't happen.
            # If transition doesn't happen, we still pay the spread? No, "depths earned" only on fill.
            # But we might still want to quote to earn depth if fill was possible?
            # But fill is NOT possible. So quoting bid yields 0 expected reward from bid side.
            # So optimal bid depth is such that p_b is 0? i.e., x_b = inf.
            # Or simply, we set bid_depth = np.inf?
            # The task says "np.inf means no quote".
            # So at q=Q, bid_depth should be np.inf.
            # At q=-Q, ask_depth should be np.inf.
            
            if q == Q:
                bid_depths[n, i] = np.inf
            else:
                # Calculate C for bid
                # C_bid = V(q+1) - V(q)
                C_bid = V_next[i + 1] - V_next[i]
                # Lookup optimal x
                # Clip C to c_grid range
                C_bid_clipped = np.clip(C_bid, c_min, c_max)
                x_bid = np.interp(C_bid_clipped, c_grid, optimal_x_grid)
                bid_depths[n, i] = x_bid
            
            if q == -Q:
                ask_depths[n, i] = np.inf
            else:
                # Calculate C for ask
                # C_ask = V(q-1) - V(q)
                C_ask = V_next[i - 1] - V_next[i]
                C_ask_clipped = np.clip(C_ask, c_min, c_max)
                x_ask = np.interp(C_ask_clipped, c_grid, optimal_x_grid)
                ask_depths[n, i] = x_ask
            
            # Calculate V_current[i]
            # We need p_b and p_a for the optimal x.
            x_b = bid_depths[n, i]
            x_a = ask_depths[n, i]
            
            # Handle inf
            if np.isinf(x_b):
                p_b = 0.0
            else:
                p_b = 1.0 - np.exp(-A * np.exp(-k * x_b))
            
            if np.isinf(x_a):
                p_a = 0.0
            else:
                p_a = 1.0 - np.exp(-A * np.exp(-k * x_a))
            
            # Expected reward
            # Reward = p_b * x_b + p_a * x_a + (1-p_a)*p_b * (V(q+1)-V(q)) + (1-p_b)*p_a * (V(q-1)-V(q)) + ... V(q)
            # Wait, the value function V includes the future.
            # V_current(q) = max ( ... )
            # The term is:
            # p_b * x_b + p_a * x_a
            # + p_b(1-p_a) V_next(q+1) + (1-p_b)p_a V_next(q-1) + (p_b p_a + (1-p_b)(1-p_a)) V_next(q)
            # Note: V_next(q) is the value at next step for same inventory.
            
            term_vq = V_next[i]
            term_vq1 = V_next[i+1] if q < Q else 0.0 # If q=Q, p_b=0 so this term vanishes, but let's be safe
            term_vqm1 = V_next[i-1] if q > -Q else 0.0
            
            # Actually, if q=Q, p_b=0.
            # If q=-Q, p_a=0.
            
            V_current[i] = (p_b * x_b + p_a * x_a +
                            p_b * (1 - p_a) * term_vq1 +
                            (1 - p_b) * p_a * term_vqm1 +
                            (1 - (p_b * (1 - p_a) + (1 - p_b) * p_a)) * term_vq)
            
            # Simplify probability of staying at q:
            # 1 - p_b(1-p_a) - (1-p_b)p_a = 1 - (p_b - p_b p_a) - (p_a - p_b p_a) = 1 - p_b - p_a + 2 p_b p_a
            # My formula above: 1 - (p_b - p_b p_a + p_a - p_b p_a) = 1 - p_b - p_a + 2 p_b p_a. Correct.
            
            # Add inventory penalty
            # Objective: depths earned minus sum (gamma/2) sigma^2 q^2 dt
            # The penalty is incurred at step n for inventory q.
            # So V_current[i] -= (gamma / 2.0) * sigma**2 * q**2 * dt
        
        V_next = V_current
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Returns bid and ask depths for the given state.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe default
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy(params)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    steps = params['steps']
    Q = params['Q']
    T = params['T']
    dt = T / steps
    
    # Determine current step
    # t is n * dt
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    
    # Map inventory to index
    # inventory is array of ints in [-Q, Q]
    # index = inventory + Q
    indices = inventory + Q
    
    # Retrieve depths
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
