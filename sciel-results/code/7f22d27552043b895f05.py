"""Bermudan put exercise policy via continuous-time put price with early-exercise correction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math


def _black_scholes_put_price(S, K, r, sigma, T):
    """Vectorized Black-Scholes put price."""
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    # Handle T=0
    mask = T > 1e-12
    out = np.where(mask, 0.0, K - S)
    if not np.any(mask):
        return out
    T_m = T[mask]
    S_m = S[mask]
    d1 = (np.log(S_m / K) + (r + 0.5 * sigma**2) * T_m) / (sigma * np.sqrt(T_m))
    d2 = d1 - sigma * np.sqrt(T_m)
    from scipy.stats import norm
    N = norm.cdf
    price = K * np.exp(-r * T_m) * N(-d2) - S_m * N(-d1)
    out[mask] = price
    return out


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    
    Strategy: Use the continuous-time Black-Scholes put price as a baseline for
    the continuation value, then apply a Bermudan early-exercise correction.
    
    For a Bermudan put, the optimal exercise rule at each date t_i is:
    Exercise if (K - S) >= V_cont(S, t_i)
    
    Where V_cont is the continuation value. We approximate V_cont using the
    Black-Scholes put price to the next exercise date, adjusted for the fact
    that we can only exercise at discrete dates.
    
    A simple but effective approach: at each exercise date, compare the immediate
    exercise payoff (K - S)+ with the discounted expected value of holding to the
    next exercise date, approximated by the BS put price to that date.
    """
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin

    # Extract parameters
    K = float(contract.get('strike', contract.get('K', 100.0)))
    r = float(contract.get('rate', contract.get('r', 0.05)))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', contract.get('maturity', 1.0)))
    n_dates = int(contract.get('n_dates', contract.get('n_exercise_dates', 12)))
    
    # Exercise dates: 12 monthly dates, no time-0 exercise
    # Dates are at T/12, 2T/12, ..., 12T/12 = T
    dt = T / n_dates
    exercise_times = np.arange(1, n_dates + 1) * dt
    
    # We need to determine the exercise rule for each date.
    # At date i (1-indexed, time t_i), we decide whether to exercise.
    # If we don't exercise, we hold to the next exercise date (or maturity).
    
    # Key insight: For a Bermudan put, the optimal policy can be approximated
    # by comparing the intrinsic value (K-S)+ with the "Bermudan continuation value."
    # 
    # A good approximation: The continuation value at time t_i for a put that can
    # be exercised at dates t_{i+1}, ..., t_n is approximately the Black-Scholes
    # put price with maturity t_{i+1} (i.e., to the next exercise date), because
    # the main value of waiting is the time value to the next opportunity.
    #
    # However, this underestimates the continuation value because it ignores the
    # option to exercise at later dates. A better approximation uses the BS put
    # price to the final maturity T, but that overestimates because it includes
    # time value beyond the last exercise date that can't be captured.
    #
    # Best practical approach: Use the BS put price to the NEXT exercise date as
    # the continuation value. This is conservative (may exercise too early) but
    # simple. Alternatively, we can use a blend.
    
    # Let's use a more refined approach:
    # At each exercise date t_i, the continuation value is the discounted expected
    # value of max(intrinsic at t_{i+1}, continuation at t_{i+1}).
    # We approximate this by the BS put price with maturity = next exercise date.
    # This is a common heuristic for Bermudan options.
    
    # Actually, a better structural approach:
    # The Bermudan put value at time t_i is >= BS put price to maturity T
    # (since Bermudan is less flexible than European... wait, no).
    # Actually, Bermudan put value is between European put to T and American put to T.
    # 
    # For the exercise decision at t_i:
    # Exercise if (K - S)+ >= V_i(S), where V_i is the continuation value.
    # 
    # Approximation: V_i(S) ≈ BS_put(S, K, r, sigma, t_{i+1} - t_i)
    # i.e., the BS put price to the next exercise date.
    
    # Let's precompute the exercise thresholds for each date.
    # At date i (0-indexed from 0 to n_dates-1, corresponding to times t_1...t_n):
    # The next exercise date is t_{i+2} (or maturity if i is the last date).
    # For the last exercise date (i = n_dates-1), there's no next date, so we
    # always exercise if in the money (since it's maturity).
    
    # Actually, let me reconsider. The 12 exercise dates are at:
    # t_1 = T/12, t_2 = 2T/12, ..., t_12 = T
    # At t_12 = T (maturity), we exercise if in the money.
    # At t_11, we compare (K-S)+ with the value of waiting to t_12.
    # The value of waiting to t_12 is the expected discounted payoff at t_12.
    # Approximate this as BS put price with maturity dt = T/12.
    
    # Let's compute the exercise threshold for each date.
    # For date i (1-indexed), the time to next exercise is dt.
    # For the last date, we always exercise if ITM.
    
    # Precompute: for each exercise date i (1 to n_dates), what is the threshold
    # below which we exercise?
    # Exercise if S <= threshold_i, where threshold_i is defined by:
    # (K - S) = BS_put(S, K, r, sigma, dt)  for i < n_dates
    # For i = n_dates: threshold = K (exercise if S < K)
    
    # Let's solve for the threshold numerically.
    # For a given dt, find S* such that (K - S*) = BS_put(S*, K, r, sigma, dt)
    # This is the early exercise boundary.
    
    def find_exercise_boundary(K, r, sigma, dt):
        """Find the critical stock price below which we exercise a put."""
        if dt <= 1e-12:
            return K
        
        # The boundary is where intrinsic value equals continuation value.
        # (K - S) = BS_put(S, K, r, sigma, dt)
        # For small dt, the boundary is slightly below K.
        
        # Use bisection. The boundary is in (0, K).
        lo = 1e-8
        hi = K
        
        def f(S):
            """f(S) = (K - S) - BS_put(S, K, r, sigma, dt)"""
            intrinsic = max(K - S, 0.0)
            if S <= 0:
                return K - 0  # BS put at S=0 is K*exp(-r*dt)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * dt) / (sigma * math.sqrt(dt))
            d2 = d1 - sigma * math.sqrt(dt)
            from scipy.stats import norm
            N = norm.cdf
            bs_put = K * math.exp(-r * dt) * N(-d2) - S * N(-d1)
            return (K - S) - bs_put
        
        # f(0) = K - K*exp(-r*dt) > 0 (for r > 0)
        # f(K) = 0 - BS_put(K, ...) < 0
        # So there's a root in (0, K).
        
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        
        return (lo + hi) / 2
    
    # Compute boundaries for each exercise date
    # Date i (1-indexed): time to next exercise is dt, except last date.
    # Actually, for date i < n_dates, time to next exercise is dt.
    # For date i = n_dates (maturity), no next date, always exercise if ITM.
    
    boundaries = np.zeros(n_dates)
    for i in range(n_dates):
        if i < n_dates - 1:
            # Not the last date, time to next exercise is dt
            boundaries[i] = find_exercise_boundary(K, r, sigma, dt)
        else:
            # Last date (maturity), exercise if in the money
            boundaries[i] = K
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, K):
            self.boundaries = boundaries
            self.K = K
            self.n_dates = len(boundaries)
        
        def exercise(self, current_state, time):
            """
            Determine exercise decision.
            
            Parameters:
            - current_state: array of current stock prices
            - time: current time (scalar or array?)
            
            Returns:
            - boolean vector: True if exercise at this date for this path
            """
            current_state = np.asarray(current_state, dtype=float)
            
            # Determine which exercise date we're at based on time
            # The exercise dates are at t_1, t_2, ..., t_n
            # time should correspond to one of these dates
            
            # Find the exercise index
            # time is a scalar or array. Let's handle both.
            if np.isscalar(time) or np.ndim(time) == 0:
                t = float(time)
                # Find which exercise date this is
                # Exercise dates are at (i+1)*dt for i=0,...,n-1
                idx = int(round(t / dt)) - 1
                idx = max(0, min(idx, self.n_dates - 1))
                boundary = self.boundaries[idx]
                return current_state <= boundary
            else:
                # time is an array
                time_arr = np.asarray(time, dtype=float)
                idxs = np.round(time_arr / dt).astype(int) - 1
                idxs = np.clip(idxs, 0, self.n_dates - 1)
                boundaries_arr = self.boundaries[idxs]
                return current_state <= boundaries_arr
    
    return ExercisePolicy(boundaries, K)
# EVOLVE-BLOCK-END
