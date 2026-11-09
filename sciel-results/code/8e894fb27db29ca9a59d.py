"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_START = time.time()
_DEADLINE = _START + 305.0  # Safety margin before 330s hard kill
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the HJB PDE for the value function and derive optimal depths via FOCs.
    
    HJB: V_t + (sigma^2/2) V_qq - (gamma/2) sigma^2 q^2 = -max_{db,da} [ ... ]
    
    We solve backward in time using explicit finite differences.
    Then for each (t, q), we solve the 1-D FOCs for db and da separately
    (exploiting that bid/ask decisions are separable given V).
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    
    # Grid parameters - use finer grid than DP for better PDE accuracy
    n_q = 2 * Q + 1
    # Time grid: use all steps
    n_t = N
    
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # PDE solution arrays
    # V[t, i] value function at time index t (0 = final), inventory index i
    V = np.zeros((n_t + 1, n_q))
    
    # Precompute cost
    cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # For the PDE, we need dV/dq and d2V/dq2
    # We'll solve the PDE backward in time
    # V_t = -[ (sigma^2/2) V_qq - (gamma/2) sigma^2 q^2 + max_depth_reward ]
    # But the max depends on V(t+dt), creating a nonlinearity.
    # 
    # Alternative: Use the DP structure but with a much finer depth grid 
    # and exploit separability for speed.
    #
    # Actually, let's use a hybrid: solve the PDE for V, then use FOCs for depths.
    # The FOC for bid depth db:
    # d/d db [ Pb(db) * (db + V_q) ] = 0  (simplified, ignoring cross terms)
    # Pb(db) = 1 - exp(-A exp(-k db) dt)
    # Pb'(db) = A k exp(-k db) dt exp(-A exp(-k db) dt)
    # 
    # Full FOC considering inventory change:
    # Pb'(db) * (db + V[t+1, q+1] - V[t+1, q]) + Pb(db) = 0
    # This is a 1-D root finding problem.
    
    # Let's use a faster DP with separable optimization:
    # For each (t, q), the optimal (db, da) can be found by:
    # 1. For each candidate db, compute optimal da (1-D search)
    # 2. Optimize over db (1-D search)
    # But this is still 2-D.
    
    # Better: Use the PDE to get V, then solve FOCs.
    # The HJB in continuous time:
    # 0 = V_t + (sigma^2/2) V_qq - (gamma/2) sigma^2 q^2 + max_{db,da} [
    #     lambda_b(db) * (db + V_q) + lambda_a(da) * (da - V_q)
    #   ]
    # where lambda_b(db) = A exp(-k db) is the intensity, and the reward includes
    # the depth earned plus the change in value from inventory shift.
    
    # For discrete time, the Bellman equation is:
    # V[t, q] = max_{db,da} [
    #   (1-Pb)(1-Pa) V[t+1, q] + Pb(1-Pa)(db + V[t+1, q+1]) 
    #   + Pa(1-Pb)(da + V[t+1, q-1]) + Pb*Pa(V[t+1, q])
    #   - cost
    # ]
    # = max_{db,da} [
    #   V[t+1, q] + Pb(1-Pa)(db + V[t+1,q+1]-V[t+1,q]) + Pa(1-Pb)(da + V[t+1,q-1]-V[t+1,q])
    #   + Pb*Pa(V[t+1,q]-V[t+1,q]) - cost
    # ]
    # Simplify: Pb*Pa term cancels in the V difference.
    # = V[t+1, q] - cost + max_{db,da} [
    #   Pb(1-Pa)(db + dV_b) + Pa(1-Pb)(da + dV_a)
    # ]
    # where dV_b = V[t+1,q+1]-V[t+1,q], dV_a = V[t+1,q-1]-V[t+1,q]
    
    # The max over (db, da) is separable IF we ignore the (1-Pa) and (1-Pb) coupling.
    # For small dt, Pb and Pa are small, so (1-Pa) ≈ 1, (1-Pb) ≈ 1.
    # Then: max_{db,da} [ Pb(db)(db + dV_b) + Pa(da)(da + dV_a) ]
    # This IS separable! We can optimize db and da independently.
    
    # FOC for db: d/d db [ Pb(db)(db + dV_b) ] = 0
    # Pb'(db)(db + dV_b) + Pb(db) = 0
    # A k e^{-k db} dt e^{-A e^{-k db} dt} (db + dV_b) + (1 - e^{-A e^{-k db} dt}) = 0
    
    # Let's implement the separable DP with 1-D root finding.
    
    # Precompute: for the FOC, we need to solve:
    # f(db) = A*k*exp(-k*db)*dt*exp(-A*exp(-k*db)*dt)*(db + dV_b) + (1 - exp(-A*exp(-k*db)*dt)) = 0
    # This is monotone in db for typical parameters, so brentq works.
    
    def _optimal_depth(dV, A, k, dt):
        """
        Find optimal depth given value difference dV.
        FOC: Pb'(d)(d + dV) + Pb(d) = 0
        where Pb(d) = 1 - exp(-A exp(-k d) dt)
        Pb'(d) = A k exp(-k d) dt exp(-A exp(-k d) dt)
        """
        if abs(dV) < 1e-15:
            dV = 0.0
        
        def f(d):
            if d < 0:
                d = 0.0
            exp_kd = np.exp(-k * d)
            arg = A * exp_kd * dt
            Pb = 1.0 - np.exp(-arg)
            Pb_prime = A * k * exp_kd * dt * np.exp(-arg)
            return Pb_prime * (d + dV) + Pb
        
        # f(0) = A*k*dt*exp(-A*dt)*(dV) + (1-exp(-A*dt))
        # For large d: f(d) -> 0 from positive side (Pb -> 0, Pb' -> 0, but Pb'*(d+dV) -> 0)
        # Actually for large d: Pb ~ A e^{-kd} dt, Pb' ~ A k e^{-kd} dt
        # f(d) ~ A k e^{-kd} dt (d + dV) + A e^{-kd} dt = A e^{-kd} dt (k(d+dV) + 1)
        # For d > 0 large: k(d+dV) + 1 > 0, so f > 0
        # f(0) = A k dt exp(-A dt) dV + 1 - exp(-A dt)
        # If dV < 0 (inventory increases in value when we go up, i.e., we want to reduce q),
        # then the first term is negative.
        
        f0 = f(0.0)
        if f0 <= 0:
            # Optimal is at or near 0
            # Check if f is increasing from 0
            f_small = f(0.01)
            if f_small > 0:
                # Root near 0
                return brentq(f, 0.0, 0.01, xtol=1e-10)
            else:
                return 0.0
        
        # f(0) > 0, look for root where f crosses 0
        # For large d, f(d) -> 0+ (from positive)
        # So if f(0) > 0 and f(large) > 0, there may be no root, or root at boundary
        # Let's check: if dV > 0, we prefer to NOT trade (depth large)
        # If dV < 0, we prefer to trade (depth small)
        
        # Find upper bound where f might cross
        d_max = 10.0 / k
        f_max = f(d_max)
        
        if f_max < 0:
            # f went from positive to negative, root exists in (0, d_max)
            return brentq(f, 0.0, d_max, xtol=1e-10)
        
        # f(0) > 0 and f(d_max) > 0: 
        # If dV > 0, optimal depth is large (don't trade)
        # If dV < 0, there should be a root... let me check more carefully
        # Actually, for dV < 0: f(0) = A*k*dt*exp(-A*dt)*dV + (1-exp(-A*dt))
        # The second term is positive and dominates for small A*dt
        # So f(0) > 0 even for negative dV typically
        # The root occurs where the negative term from dV balances
        
        # Let's scan for a sign change
        d_scan = np.linspace(0, d_max, 100)
        f_scan = np.array([f(d) for d in d_scan])
        sign_changes = np.where(np.diff(np.sign(f_scan)) != 0)[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            return brentq(f, d_scan[idx], d_scan[idx+1], xtol=1e-10)
        
        # No root found: if dV > 0, use large depth; if dV < 0, use small depth
        if dV > 0:
            return min(d_max, 5.0 / k)
        else:
            return max(0.0, min(1.0 / k, 2.0 / k))
    
    # Now do the backward induction with separable optimization
    # This is much faster than 200x200 grid search
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback
            for tt in range(t, -1, -1):
                base = 1.0 / k
                V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        V_next = V[t + 1, :]
        
        # dV_b[q] = V[t+1, q+1] - V[t+1, q] for q < Q
        # dV_a[q] = V[t+1, q-1] - V[t+1, q] for q > -Q
        dV_b = np.zeros(n_q)
        dV_a = np.zeros(n_q)
        dV_b[:-1] = V_next[1:] - V_next[:-1]  # q -> q+1
        dV_b[-1] = 0.0  # q=Q, no bid fill
        dV_a[1:] = V_next[:-1] - V_next[1:]  # q -> q-1
        dV_a[0] = 0.0  # q=-Q, no ask fill
        
        # For each q, find optimal db and da
        bid_depths = np.zeros(n_q)
        ask_depths = np.zeros(n_q)
        
        for i in range(n_q):
            # Optimal bid depth
            if i < n_q - 1:  # q < Q
                bid_depths[i] = _optimal_depth(dV_b[i], A, k, dt)
            else:
                bid_depths[i] = 0.0  # No bid at Q (or very large)
                # Actually at Q, we should not quote bid, so depth = inf
                # But we can't use inf. Use a very large depth.
                bid_depths[i] = 1e6  # Effectively no quote
            
            # Optimal ask depth
            if i > 0:  # q > -Q
                ask_depths[i] = _optimal_depth(dV_a[i], A, k, dt)
            else:
                ask_depths[i] = 1e6  # No ask at -Q
        
        # Compute V[t, i] using the optimal depths
        for i in range(n_q):
            db = bid_depths[i]
            da = ask_depths[i]
            
            if i >= n_q - 1:
                Pb = 0.0
            else:
                Pb = 1.0 - np.exp(-A * np.exp(-k * db) * dt)
            
            if i <= 0:
                Pa = 0.0
            else:
                Pa = 1.0 - np.exp(-A * np.exp(-k * da) * dt)
            
            V_same = V_next[i]
            V_bid = V_next[i + 1] if i + 1 < n_q else V_next[i]
            V_ask = V_next[i - 1] if i - 1 >= 0 else V_next[i]
            
            EV = (Pb * (1 - Pa) * (db + V_bid) +
                  Pa * (1 - Pb) * (da + V_ask) +
                  Pb * Pa * V_same +
                  (1 - Pb) * (1 - Pa) * V_same)
            
            V[t, i] = EV - cost[i]
        
        # Store depths for lookup
        # We'll store them in separate arrays
        if not hasattr(_solve_regime, '_bid_tables'):
            _solve_regime._bid_tables = {}
            _solve_regime._ask_tables = {}
        
        _solve_regime._bid_tables[t] = bid_depths
        _solve_regime._ask_tables[t] = ask_depths
    
    # Build the tables
    bid_table = np.zeros((N, n_q))
    ask_table = np.zeros((N, n_q))
    for t in range(N):
        if t in _solve_regime._bid_tables:
            bid_table[t] = _solve_regime._bid_tables[t]
            ask_table[t] = _solve_regime._ask_tables[t]
        else:
            bid_table[t] = 1.0 / k
            ask_table[t] = 1.0 / k
    
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
    
    # Replace very large depths (no-quote markers) with inf
    bid_depths[bid_depths > 1e5] = np.inf
    ask_depths[ask_depths > 1e5] = np.inf
    
    # Ensure non-negative and valid
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    # Safety: replace any remaining issues
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=np.inf, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=np.inf, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
