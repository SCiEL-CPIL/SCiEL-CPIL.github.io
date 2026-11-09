"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_START = time.time()
_DEADLINE = _START + 315.0  # Safety margin before 330s hard kill
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via dynamic programming with analytic depth selection.
    
    The Bellman equation separates into independent conditions for bid and ask depths:
    bid: Pb * (1 - Pa) * (V_bid_next - V_same_next) + Pb * db = 0
    ask: Pa * (1 - Pb) * (V_ask_next - V_same_next) + Pa * da = 0
    
    We solve for db and da analytically (or via root finding) for each inventory state.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Value function: V[t, i] where i indexes q from -Q to Q
    V = np.zeros((N + 1, n_q))
    # Optimal depths tables
    bid_table = np.zeros((N, n_q))
    ask_table = np.zeros((N, n_q))
    
    # Precompute cost per inventory level: (gamma/2) * sigma^2 * q^2 * dt
    cost_per_q = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Precompute helper functions for probability
    # P(d) = 1 - exp(-A * exp(-k*d) * dt)
    # Let lambda(d) = A * exp(-k*d) * dt
    # P = 1 - exp(-lambda)
    
    # We need to solve for d given a target probability P_target.
    # 1 - exp(-lambda) = P_target => exp(-lambda) = 1 - P_target => lambda = -ln(1 - P_target)
    # A * exp(-k*d) * dt = -ln(1 - P_target)
    # exp(-k*d) = -ln(1 - P_target) / (A * dt)
    # -k*d = ln(-ln(1 - P_target) / (A * dt))
    # d = -1/k * ln(-ln(1 - P_target) / (A * dt))
    
    # Define a safe inverse probability function
    def inv_prob(p):
        # Clamp p to (0, 1)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        # lambda = -ln(1 - p)
        lam = -np.log1p(-p)
        # d = -1/k * ln(lam / (A * dt))
        # Note: if lam / (A*dt) < 1, ln is negative, so d is positive.
        # If lam / (A*dt) > 1, ln is positive, d is negative. 
        # But depth must be >= 0.
        # Let's check the range.
        # At d=0, lambda = A*dt. P_max = 1 - exp(-A*dt).
        # So P_target must be <= P_max for a valid non-negative depth.
        # If P_target > P_max, the optimal depth is 0 (or we clip to 0).
        # If P_target < 0, impossible.
        
        val = lam / (A * dt)
        # If val <= 0, impossible (shouldn't happen)
        # If val >= 1, then ln(val) >= 0, so d <= 0. We clip to 0.
        # If val < 1, ln(val) < 0, so d > 0.
        
        d = -1.0 / k * np.log(val)
        return np.maximum(d, 0.0)
    
    # Precompute P_max for each state? No, P_max is constant for all states.
    P_max = 1.0 - np.exp(-A * dt)
    
    # We will iterate backwards in time.
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Fallback: use current V and set remaining to safe defaults
            for tt in range(t, -1, -1):
                if tt < N:
                    # Use symmetric depth based on inventory
                    base_d = 1.0 / k
                    bid_table[tt] = np.clip(base_d + 0.01 * q_vals, 0.0, None)
                    ask_table[tt] = np.clip(base_d - 0.01 * q_vals, 0.0, None)
                    V[tt] = V[tt + 1] if tt + 1 <= N else 0.0
            break
        
        V_next = V[t + 1, :]  # shape (n_q,)
        
        # V after bid fill (q -> q+1): for i from 0 to n_q-2, next is i+1
        V_bid_next = np.zeros(n_q)
        V_bid_next[:n_q-1] = V_next[1:n_q]
        V_bid_next[n_q-1] = V_next[n_q-1]  # placeholder, will be masked
        
        # V after ask fill (q -> q-1): for i from 1 to n_q-1, next is i-1
        V_ask_next = np.zeros(n_q)
        V_ask_next[1:] = V_next[:n_q-1]
        V_ask_next[0] = V_next[0]  # placeholder, will be masked
        
        V_same_next = V_next.copy()
        
        costs = cost_per_q  # shape (n_q,)
        
        # The Bellman equation is:
        # V[t, i] = max_{db, da} [ Pb(1-Pa)V_bid + Pa(1-Pb)V_ask + (PbPa + (1-Pb)(1-Pa))V_same + Pb*db + Pa*da - cost ]
        
        # Let's define:
        # DeltaV_bid = V_bid_next[i] - V_same_next[i]
        # DeltaV_ask = V_ask_next[i] - V_same_next[i]
        
        # The term involving Pb:
        # Pb * (1 - Pa) * DeltaV_bid + Pb * db
        # The term involving Pa:
        # Pa * (1 - Pb) * DeltaV_ask + Pa * da
        
        # This is coupled because Pb appears in the Pa term and vice versa.
        # However, we can solve for db given da, and da given db.
        # Or, we can iterate.
        # Given the small number of steps and states, a simple fixed-point iteration or nested solve might work.
        # But let's look closer.
        
        # For a fixed da (and thus Pa), the optimal db maximizes:
        # Pb * ( (1 - Pa) * DeltaV_bid + db )
        # Let f(Pb) = Pb * C_b + g(Pb) where C_b = (1 - Pa) * DeltaV_bid
        # and g(Pb) is the inverse mapping from Pb to db.
        # db = inv_prob(Pb)
        # So we maximize h(Pb) = Pb * C_b + inv_prob(Pb)
        
        # Similarly for Pa.
        
        # Let's try to solve analytically.
        # db = -1/k * ln( -ln(1-Pb) / (A dt) )
        # This is convex?
        
        # Alternative: Since we have only 61 states, we can do a 1D root find for each state.
        # But the coupling is the issue.
        
        # Let's use the fact that the optimal depth is often close to a simple heuristic.
        # Or, we can just use a fine grid but much smaller, since we don't need 200x200.
        # Actually, let's stick to the grid but make it adaptive or smaller.
        # The previous score was 1.348 with grid_size=200.
        # Let's try to speed up the grid search by reducing the search space.
        # The optimal depth is usually small.
        
        # Let's try a different approach: 
        # Since the problem is separable in expectation if we ignore the cross-term (1-Pb) in the Pa part?
        # No, they are coupled.
        
        # Let's just optimize the grid search implementation.
        # The previous implementation created huge 3D arrays.
        # We can do this state-by-state with 1D arrays, which is memory efficient and fast enough.
        
        # For each state i:
        #   Search db in [0, max_d], da in [0, max_d]
        #   Compute EV for all pairs.
        #   This is O(n_q * grid^2). 61 * 200^2 = 2.4M operations per step. 1000 steps = 2.4B ops.
        #   This might be too slow in Python loops.
        #   Vectorizing over states (n_q) is better.
        
        # Let's revert to vectorized grid but optimize memory.
        # Instead of full 3D arrays, we can compute EV in chunks or use broadcasting cleverly.
        
        # Actually, the previous code was:
        # EV = ... (n_q, grid, grid)
        # This is 61 * 200 * 200 * 8 bytes = ~19.5 MB.
        # Creating this array 1000 times is 19.5 GB of memory traffic.
        # This is likely the bottleneck.
        
        # Let's try to reduce grid size to 100 and see if it fits in time.
        # Or, use a coarser grid for early steps and finer for late steps?
        # Or, simply use a smaller grid size like 50.
        
        grid_size = 100
        max_depth = 4.0 / k  # Reduce max depth, optimal is usually small
        depths = np.linspace(0.0, max_depth, grid_size)
        
        exp_term = np.exp(-k * depths)
        P = 1.0 - np.exp(-A * exp_term * dt)
        
        # Precompute for this t
        # We need to find best (db, da) for each i.
        # EV[i, j, l] = Pb[j]*(1-Pa[l])*V_bid[i] + Pa[l]*(1-Pb[j])*V_ask[i] + ...
        
        # Let's compute the terms separately.
        # Term 1: Pb[j] * (1 - Pa[l]) * V_bid_next[i]
        # Term 2: Pa[l] * (1 - Pb[j]) * V_ask_next[i]
        # Term 3: (Pb[j]*Pa[l] + (1-Pb[j])*(1-Pa[l])) * V_same_next[i]
        # Term 4: Pb[j] * depths[j]
        # Term 5: Pa[l] * depths[l]
        # Term 6: -costs[i]
        
        # We can rewrite EV as:
        # EV[i, j, l] = V_bid[i] * Pb[j] * (1-Pa[l]) + V_ask[i] * Pa[l] * (1-Pb[j]) + 
        #               V_same[i] * (Pb[j]*Pa[l] + 1 - Pb[j] - Pa[l] + Pb[j]*Pa[l]) +
        #               Pb[j]*depths[j] + Pa[l]*depths[l] - costs[i]
        
        # Let's group by Pb and Pa.
        # This is still coupled.
        
        # Let's just use a smaller grid and hope it's fast enough.
        # With grid_size=100, memory is ~5 MB.
        # Operations: 61 * 100 * 100 = 610k per step. 1000 steps = 610M ops.
        # This should be fast in NumPy.
        
        Pb = P[None, :, None] # (1, grid, 1)
        Pa = P[None, None, :] # (1, 1, grid)
        db_grid = depths[None, :, None]
        da_grid = depths[None, None, :]
        
        # Broadcast to (n_q, grid, grid)
        # We do this in a loop over i to save memory? No, broadcasting is fast.
        
        # V_bid_next[i], V_ask_next[i], V_same_next[i], costs[i] are scalars for each i.
        # We can create arrays of shape (n_q, 1, 1) and broadcast.
        
        V_bid_b = V_bid_next[:, None, None]
        V_ask_b = V_ask_next[:, None, None]
        V_same_b = V_same_next[:, None, None]
        costs_b = costs[:, None, None]
        
        # Compute EV
        # (1-Pa) = 1 - Pa
        # (1-Pb) = 1 - Pb
        
        EV = (V_bid_b * Pb * (1 - Pa) + 
              V_ask_b * Pa * (1 - Pb) + 
              V_same_b * (Pb * Pa + (1 - Pb) * (1 - Pa)) +
              Pb * db_grid + 
              Pa * da_grid - 
              costs_b)
        
        # Handle boundaries
        # For i = n_q-1 (q=Q), bid fill is impossible. So Pb should be 0.
        # For i = 0 (q=-Q), ask fill is impossible. So Pa should be 0.
        
        # We can just set the corresponding EV to -inf for invalid combos?
        # Or set Pb=0 for i=n_q-1 and Pa=0 for i=0 in the calculation.
        
        # Let's create masked Pb and Pa.
        Pb_masked = Pb.copy()
        Pa_masked = Pa.copy()
        
        # For i=n_q-1, all Pb are 0.
        # This is tricky with broadcasting.
        # Let's just compute EV and then set invalid entries to -inf.
        
        # Invalid:
        # i = n_q-1, any j (bid depth > 0 is invalid? No, depth 0 is valid, but prob is 0).
        # Actually, if q=Q, bid fill prob is 0. So Pb=0.
        # If Pb=0, the term Pb*db is 0. The term Pb*(1-Pa)*V_bid is 0.
        # The term (1-Pb)*(1-Pa)*V_same is (1-Pa)*V_same.
        # So it's effectively just the ask side.
        # The grid search will naturally pick db=0 if it's optimal?
        # Not necessarily, because db adds to the reward.
        # But if Pb=0, db doesn't matter for the reward (Pb*db=0).
        # So any db is equally good. We should pick db=0 to be safe/consistent.
        
        # Let's enforce Pb=0 for i=n_q-1 and Pa=0 for i=0.
        
        # We can do this by creating full arrays of Pb and Pa with masks.
        # Pb_full[i, j, l] = P[j] if i < n_q-1 else 0
        # Pa_full[i, j, l] = P[l] if i > 0 else 0
        
        Pb_full = np.broadcast_to(Pb, (n_q, grid_size, grid_size)).copy()
        Pa_full = np.broadcast_to(Pa, (n_q, grid_size, grid_size)).copy()
        
        Pb_full[n_q-1, :, :] = 0.0
        Pa_full[0, :, :] = 0.0
        
        # Recompute EV with masked probabilities
        EV = (V_bid_b * Pb_full * (1 - Pa_full) + 
              V_ask_b * Pa_full * (1 - Pb_full) + 
              V_same_b * (Pb_full * Pa_full + (1 - Pb_full) * (1 - Pa_full)) +
              Pb_full * db_grid + 
              Pa_full * da_grid - 
              costs_b)
        
        # Find optimal
        EV_flat = EV.reshape(n_q, -1)
        best_flat_idx = np.argmax(EV_flat, axis=1)
        best_db_idx = best_flat_idx // grid_size
        best_da_idx = best_flat_idx % grid_size
        
        bid_table[t, :] = depths[best_db_idx]
        ask_table[t, :] = depths[best_da_idx]
        
        V[t, :] = EV_flat[np.arange(n_q), best_flat_idx]
        
    _CACHE[key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    Uses exact precomputed DP tables for the current regime.
    """
    if time.time() > _DEADLINE:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    bid_table, ask_table = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    # t is n * dt, so step index n = int(round(t / dt))
    dt = params["T"] / params["steps"]
    n = int(round(t / dt))
    n = max(0, min(n, params["steps"] - 1))
    
    # inventory is int array in [-Q, Q]
    # Index into table: i = q + Q
    Q = params["Q"]
    idx = inventory + Q
    
    # Clip indices to valid range (should already be valid)
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[n, idx]
    ask_depths = ask_table[n, idx]
    
    # Ensure non-negative depths and no NaN/inf
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
