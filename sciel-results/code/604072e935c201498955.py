"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 320.0  # Safety margin before 330s hard kill
_CACHE = {}

def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via dynamic programming.
    
    Key Insight: The Bellman equation can be decoupled.
    Let V_next[q] be the value at time t+1 for inventory q.
    Let C_q = (gamma/2) * sigma^2 * q^2 * dt.
    
    V_t[q] = max_{db, da} { 
        P_b(db) * (1 - P_a(da)) * (V_next[q+1] - V_next[q]) 
        + P_a(da) * (1 - P_b(db)) * (V_next[q-1] - V_next[q])
        + (P_b(db)*P_a(da) + (1-P_b(db))*(1-P_a(da))) * V_next[q]
        + P_b(db)*db + P_a(da)*da 
        - C_q
    }
    
    Expanding and grouping by P_b and P_a:
    Let D_b = V_next[q+1] - V_next[q]
    Let D_a = V_next[q-1] - V_next[q]
    
    Term(P_b, P_a) = P_b * (1-P_a) * D_b + P_a * (1-P_b) * D_a 
                    + (P_b P_a + 1 - P_b - P_a + P_b P_a) * V_next[q]
                    + P_b db + P_a da - C_q
                    
    = P_b D_b - P_b P_a D_b + P_a D_a - P_b P_a D_a 
      + P_b P_a V_next[q] + V_next[q] - P_b V_next[q] - P_a V_next[q] + P_b P_a V_next[q]
      + P_b db + P_a da - C_q
      
    This is still coupled in the P_b P_a term. However, notice that the objective is 
    linear in P_b given P_a, and linear in P_a given P_b?
    
    Let's rewrite as:
    f(P_b, P_a) = P_b [ (1-P_a)D_b - P_a D_a + 2 P_a V_next[q] - V_next[q] + db ] 
                  + P_a [ D_a - P_b D_a + P_b V_next[q] - V_next[q] + da ] 
                  + V_next[q] - C_q
    
    Wait, the db and da are functions of P_b and P_a.
    db(P_b) = -1/k * ln( -ln(1-P_b) / (A dt) )
    da(P_a) = -1/k * ln( -ln(1-P_a) / (A dt) )
    
    This function f(P_b, P_a) is concave in P_b and P_a separately?
    If so, we can use coordinate descent or solve for one given the other.
    Given the small number of states (61) and steps (1000), we can do a very fast 
    1D optimization for each dimension sequentially until convergence, or simply 
    assume that for a fixed P_a, the optimal P_b can be found, and vice versa.
    
    Actually, a simpler structural change:
    The current code uses a 100x100 grid. This is 10,000 evaluations per state per step.
    61 states * 1000 steps * 10,000 = 610,000,000 evaluations.
    This is the bottleneck.
    
    New Strategy:
    Use the fact that the optimal depth is often small.
    We can use a 1D search for db and da separately, using a simple heuristic for the 
    "other" side's probability, or iterate.
    
    Better: Use the analytic derivative.
    Let's define the value function for a fixed P_a:
    g(P_b) = P_b * (1 - P_a) * D_b + P_b * db(P_b) + P_b * ( -P_a D_a + 2 P_a V_next[q] - V_next[q] ) 
             + [Terms not involving P_b]
             
    Wait, the cross term - P_b P_a D_a and + P_b P_a V_next[q] are linear in P_b.
    So g(P_b) is linear in P_b plus P_b * db(P_b).
    db(P_b) is convex? 
    d = -1/k ln( -ln(1-p) / L ). 
    As p -> 0, -ln(1-p) ~ p, so d ~ -1/k ln(p/L). d -> infinity.
    As p -> 1, -ln(1-p) -> infinity, d -> -infinity.
    
    The term P_b * db(P_b) goes to 0 as P_b -> 0 and -infinity as P_b -> 1.
    So the function is concave in P_b?
    Let's check the second derivative of h(p) = p * (-1/k ln(-ln(1-p)/L)).
    
    Regardless, we can find the maximum of g(P_b) using a 1D optimizer like brentq on the derivative, 
    or a simple golden section search. This reduces the complexity from O(G^2) to O(G) or O(log G).
    
    Even better: Since we only have 61 states, we can just use a finer grid, say 500x500, 
    IF we can make it fast enough. 500*500 = 250,000. 61 * 1000 * 250,000 = 15 billion. Too slow.
    
    So we MUST decouple.
    
    Let's use the property that the optimal P_b and P_a are likely to be close to a 
    symmetric solution or can be found via fixed point iteration.
    
    Algorithm:
    1. Initialize P_b = P_a = 0.5 (or some heuristic).
    2. For each state q:
       a. Given P_a, maximize g(P_b) over P_b in [0, 1].
       b. Given P_b, maximize h(P_a) over P_a in [0, 1].
       c. Iterate until convergence.
       
    To maximize g(P_b):
    g(P_b) = P_b * [ (1-P_a)D_b - P_a D_a + 2 P_a V_next[q] - V_next[q] ] + P_b * db(P_b)
    Let C1 = (1-P_a)D_b - P_a D_a + 2 P_a V_next[q] - V_next[q]
    g(P_b) = C1 * P_b + P_b * db(P_b)
    
    We want to find P_b that maximizes this.
    db(P_b) = -1/k * ln( -ln(1-P_b) / (A dt) )
    
    Let L = A * dt.
    db(P_b) = -1/k * ln( -ln(1-P_b) / L )
    
    Derivative w.r.t P_b:
    d/db * dP_b/dp ? No, db is a function of p.
    Let u = -ln(1-p). du/dp = 1/(1-p).
    db = -1/k ln(u/L).
    d(db)/dp = -1/k * (1/u) * du/dp = -1/k * (1/u) * 1/(1-p) = -1/(k * u * (1-p))
    
    g'(p) = C1 + db(p) + p * d(db)/dp
    g'(p) = C1 - 1/k ln(u/L) - p / (k * u * (1-p))
    
    Set g'(p) = 0:
    C1 = 1/k ln(u/L) + p / (k * u * (1-p))
    
    This is a 1D root finding problem for p in (0, 1).
    We can use Newton's method or bisection.
    Since g(p) is concave (likely), bisection on the derivative is safe and fast.
    
    This reduces the per-state cost to ~50 iterations of bisection * 2 (for bid and ask) * 61 states * 1000 steps.
    50 * 2 * 61 * 1000 = 6,100,000 operations. This is very fast.
    
    Let's implement this.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    V = np.zeros((N + 1, n_q))
    bid_table = np.zeros((N, n_q))
    ask_table = np.zeros((N, n_q))
    
    cost_per_q = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    L = A * dt
    
    # Precompute depth from probability
    # p in [0, 1]
    # d = -1/k * ln( -ln(1-p) / L )
    # Handle p=0: d -> inf. But we want d=0? 
    # If p=0, depth is infinity. But we want to minimize depth? 
    # No, we maximize reward. 
    # If P_b is very small, db is large. 
    # The term P_b * db is small. 
    # The term C1 * P_b is small. 
    # So p=0 is a candidate.
    
    # We will solve for p in [0, 1-eps].
    
    def solve_p(C1, other_p, D_target, V_same, is_bid=True):
        # Maximize g(p) = C1 * p + p * db(p)
        # where C1 depends on other_p.
        # Wait, C1 is passed in.
        
        # g'(p) = C1 - 1/k ln(u/L) - p / (k * u * (1-p))
        # u = -ln(1-p)
        
        # We want to find p such that g'(p) = 0.
        # g'(0) = C1 - 1/k ln(0/L) ... wait.
        # As p -> 0, u -> 0, ln(u/L) -> -inf, so -1/k ln(u/L) -> +inf.
        # So g'(0) -> +inf.
        
        # As p -> 1, u -> inf, ln(u/L) -> inf, -1/k ln(u/L) -> -inf.
        # p/(k u (1-p)) -> +inf (since 1-p -> 0).
        # So g'(1) -> -inf.
        
        # So there is a root in (0, 1).
        
        # Bisection
        lo = 1e-12
        hi = 1 - 1e-12
        
        for _ in range(50):
            mid = (lo + hi) / 2
            u = -np.log1p(-mid)
            # Avoid log of 0 or negative
            if u <= 0:
                u = 1e-300
            
            term1 = C1
            term2 = -1.0 / k * np.log(u / L)
            term3 = -mid / (k * u * (1 - mid))
            
            deriv = term1 + term2 + term3
            
            if deriv > 0:
                lo = mid
            else:
                hi = mid
        
        p_opt = (lo + hi) / 2
        
        # Convert p_opt to depth
        u = -np.log1p(-p_opt)
        if u <= 0:
            return 0.0, 0.0 # p=0, depth=inf? No, if p=0, depth is infinite, but reward 0.
                            # But if p is very small, depth is large.
                            # We want to return the depth.
                            # If p_opt is close to 0, depth is large.
                            # But if C1 is negative, we might want p=0.
                            # Let's check the value at p=0.
                            # g(0) = 0.
                            # g(p_opt) = C1 * p_opt + p_opt * db(p_opt)
                            # If g(p_opt) < 0, we should choose p=0 (depth=inf? No, no quote).
                            # But the problem says "no quote" is np.inf.
                            # However, in the DP, we assume we always quote.
                            # If the optimal depth is "infinite", it means we don't quote.
                            # But our formula for depth gives large depth for small p.
                            # We should clip depth to a reasonable max? 
                            # The task says "distances below and above mid".
                            # If we don't quote, we return np.inf.
                            # But the DP value assumes a quote.
                            # Actually, if we don't quote, P=0, depth=inf, reward=0.
                            # In our formula, if p=0, reward=0.
                            # So if the max is at p=0, we return inf depth?
                            # But the evaluator might not like inf depth if it expects a number.
                            # "np.inf means no quote".
                            # So we can return np.inf if p_opt is very small.
                            
                            # Let's check if g(p_opt) is positive.
        
        db_val = -1.0 / k * np.log(u / L)
        
        # Check if it's better to not quote (p=0)
        # g(0) = 0
        # g(p_opt) = C1 * p_opt + p_opt * db_val
        
        val_opt = C1 * p_opt + p_opt * db_val
        
        if val_opt < 0:
            # Better to not quote? 
            # But we must return a depth. 
            # If we return a huge depth, P is small, reward is small.
            # If we return np.inf, P=0, reward=0.
            # So if val_opt < 0, we should return np.inf?
            # But the DP step assumes we are in the state and making a decision.
            # If we decide not to quote, the value is just V_same - cost.
            # But our Bellman equation included the quote terms.
            # We should have taken max( Quote_EV, No_Quote_EV ).
            # No_Quote_EV = V_next[q] - C_q.
            # Quote_EV = ...
            
            # Let's handle this in the main loop by comparing.
            return p_opt, db_val, val_opt
        
        return p_opt, db_val, val_opt
    
    # Main DP Loop
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback
            for tt in range(t, -1, -1):
                base_d = 1.0 / k
                bid_table[tt] = np.clip(base_d + 0.01 * q_vals, 0.0, None)
                ask_table[tt] = np.clip(base_d - 0.01 * q_vals, 0.0, None)
                V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        V_next = V[t + 1, :]
        
        V_bid_next = np.zeros(n_q)
        V_bid_next[:n_q-1] = V_next[1:n_q]
        V_bid_next[n_q-1] = V_next[n_q-1]
        
        V_ask_next = np.zeros(n_q)
        V_ask_next[1:] = V_next[:n_q-1]
        V_ask_next[0] = V_next[0]
        
        V_same_next = V_next.copy()
        costs = cost_per_q
        
        D_b = V_bid_next - V_same_next
        D_a = V_ask_next - V_same_next
        
        # For each state i
        for i in range(n_q):
            # No bid fill at q=Q (i=n_q-1)
            # No ask fill at q=-Q (i=0)
            
            # We will use coordinate descent.
            # Init p_b, p_a
            p_b = 0.1
            p_a = 0.1
            
            # Iterate
            for _ in range(10):
                # Solve for p_b given p_a
                if i < n_q - 1:
                    C1_b = (1 - p_a) * D_b[i] - p_a * D_a[i] + 2 * p_a * V_same_next[i] - V_same_next[i]
                    # Wait, the term - P_b P_a D_a + 2 P_b P_a V_next[q] - P_b V_next[q]
                    # Grouped by P_b:
                    # P_b [ (1-P_a)D_b - P_a D_a + 2 P_a V_same - V_same ]
                    # Yes.
                    
                    p_b_new, db_new, val_b_new = solve_p(C1_b, p_a, D_b[i], V_same_next[i], True)
                else:
                    p_b_new = 0.0
                    db_new = 0.0
                    val_b_new = 0.0
                
                # Solve for p_a given p_b
                if i > 0:
                    C1_a = (1 - p_b) * D_a[i] - p_b * D_b[i] + 2 * p_b * V_same_next[i] - V_same_next[i]
                    p_a_new, da_new, val_a_new = solve_p(C1_a, p_b, D_a[i], V_same_next[i], False)
                else:
                    p_a_new = 0.0
                    da_new = 0.0
                    val_a_new = 0.0
                
                p_b = p_b_new
                p_a = p_a_new
            
            # Calculate final EV
            # EV = P_b(1-P_a)V_bid + P_a(1-P_b)V_ask + (P_b P_a + (1-P_b)(1-P_a))V_same + P_b db + P_a da - cost
            
            if i < n_q - 1 and i > 0:
                EV = (p_b * (1 - p_a) * V_bid_next[i] + 
                      p_a * (1 - p_b) * V_ask_next[i] + 
                      (p_b * p_a + (1 - p_b) * (1 - p_a)) * V_same_next[i] + 
                      p_b * db_new + 
                      p_a * da_new - 
                      costs[i])
            elif i == n_q - 1:
                # No bid
                EV = (p_a * V_ask_next[i] + (1 - p_a) * V_same_next[i] + p_a * da_new - costs[i])
                # Wait, if no bid, P_b=0.
                # EV = 0 + P_a(1-0)V_ask + (0 + 1*1-P_a)V_same + 0 + P_a da - cost
                # = P_a V_ask + (1-P_a) V_same + P_a da - cost
                # This matches.
            elif i == 0:
                # No ask
                EV = (p_b * V_bid_next[i] + (1 - p_b) * V_same_next[i] + p_b * db_new - costs[i])
            
            # Compare with No Quote
            EV_no_quote = V_same_next[i] - costs[i]
            
            if EV_no_quote > EV:
                # Choose no quote
                # Depth = inf
                bid_table[t, i] = np.inf
                ask_table[t, i] = np.inf
                V[t, i] = EV_no_quote
            else:
                bid_table[t, i] = db_new
                ask_table[t, i] = da_new
                V[t, i] = EV
                
    _CACHE[key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    """
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    bid_table, ask_table = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    Q = params["Q"]
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    # Ensure non-negative and valid
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
