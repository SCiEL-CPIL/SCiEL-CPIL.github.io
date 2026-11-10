"""Rewritten program: high-resolution binomial-tree backward induction for Bermudan put."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy via backward induction on a binomial tree.
    Returns an object with exercise(current_state, time) -> boolean vector.
    """
    t_start = time.time()
    DEADLINE = 150.0
    MARGIN = 5.0  # safety margin

    # Parse contract
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_ex = int(contract.get('n_exercise', 12))
    S0 = float(contract.get('S0', 100.0))

    # Exercise dates: 12 monthly, no time-0. t_i = i * T / n_ex for i=1..12
    # Exercise at t_1, t_2, ..., t_12 (= T)
    ex_dates = np.linspace(0, T, n_ex + 1)[1:]  # 12 dates
    dt = T / n_ex

    # Build a binomial tree with enough steps to be accurate.
    # Use many steps (e.g., 1200) to match/exceed reference precision.
    # But we only exercise at 12 dates, so we can use a coarser tree aligned
    # to those dates, OR a fine tree. A fine tree (1200 steps) is safe.
    # However, a 1200-step tree over 12 exercise dates means each exercise
    # date has 100 sub-steps. We do backward induction on the full tree but
    # only allow exercise at the 12 dates.
    N = 1200  # total binomial steps
    sub = N // n_ex  # steps per exercise interval (100)

    # Binomial parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # We'll do backward induction on the full N-step tree.
    # State value V[j] at each node. We store the terminal payoff, then
    # step backward. At exercise dates (every `sub` steps), we take max(payoff, continue).

    # Terminal payoff at T: max(K - S, 0)
    # At step N, node j (0..N), S = S0 * u^j * d^(N-j)
    # We'll work with log-spaced or just compute S at each node.

    # To be efficient, we track the value array of size (N+1).
    # Start at terminal.
    # j from 0..N
    S_terminal = S0 * (u ** np.arange(N + 1)) * (d ** (N - np.arange(N + 1)))
    V = np.maximum(K - S_terminal, 0.0)

    # Backward induction
    # Step from N down to 1. At steps that correspond to exercise dates, allow exercise.
    # Exercise dates are at steps: sub, 2*sub, 3*sub, ..., n_ex*sub = N
    # (since we exercise at t_1..t_12, and t_12 = T = step N)
    # So at step index s (going backward from N to 1), we are at time (N-s)*dt.
    # We exercise if (N-s) is a multiple of sub and >0... let's think.
    # At backward step s (we are computing V at step s from V at step s+1),
    # the time at step s is t = s*dt. We exercise at t = k*dt where k in {sub,2sub,...,N}.
    # So when computing V at step s, if s is a multiple of sub (and s>0), we can exercise.
    for s in range(N, 0, -1):
        if time.time() - t_start > DEADLINE - MARGIN:
            break  # use current V
        # V at step s, nodes 0..s-1 (since at step s there are s nodes? no.)
        # At step s (s nodes? Actually at step s, there are s+1 nodes j=0..s).
        # We go from step s+1 (s+2 nodes) to step s (s+1 nodes).
        # V_up = V[j+1], V_down = V[j]
        V_next = V  # size s+2
        V_s = disc * (p_up * V_next[1:] + (1 - p_up) * V_next[:-1])
        # Now check exercise at step s if it's an exercise date
        if s % sub == 0:
            # Compute S at step s, nodes j=0..s
            S_s = S0 * (u ** np.arange(s + 1)) * (d ** (s - np.arange(s + 1)))
            payoff = np.maximum(K - S_s, 0.0)
            V_s = np.maximum(V_s, payoff)
        V = V_s

    # Now V at step 0 is the option value. But we need exercise thresholds at each date.
    # The exercise policy: at exercise date k (step k*sub), exercise if payoff > continue value.
    # We need the threshold spot at each exercise date.
    # During backward induction, at exercise step s=k*sub, we had V_s = max(continue, payoff).
    # The threshold is where payoff = continue. We can recompute: at each exercise date,
    # find the node where we switch from exercise to continue.

    # Let's recompute the continue values at each exercise date to extract thresholds.
    # We'll redo backward induction but record the "continue" value at each exercise date.
    # Actually, more efficient: recompute forward the S grid and use the V arrays.
    # But we overwrote V. Let's do a clean pass recording thresholds.

    thresholds = np.zeros(n_ex)  # threshold spot for each of 12 exercise dates
    V = np.maximum(K - S_terminal, 0.0)
    for s in range(N, 0, -1):
        if time.time() - t_start > DEADLINE - MARGIN:
            # Fallback: use deep ITM threshold (exercise always when ITM)
            for k in range(n_ex):
                if thresholds[k] == 0.0:
                    thresholds[k] = K
            break
        V_next = V
        V_s = disc * (p_up * V_next[1:] + (1 - p_up) * V_next[:-1])
        if s % sub == 0:
            k = s // sub  # exercise date index 1..n_ex
            S_s = S0 * (u ** np.arange(s + 1)) * (d ** (s - np.arange(s + 1)))
            payoff = np.maximum(K - S_s, 0.0)
            # continue value is V_s (before max), exercise where payoff > V_s
            # threshold: largest S where payoff >= V_s (exercise region is low S)
            # For a put, exercise when S is low. Find boundary.
            # exercise_mask = payoff >= V_s
            # The threshold is the highest S at which we exercise.
            ex_mask = payoff >= V_s
            if np.any(ex_mask):
                # nodes where we exercise; find the max S among them
                S_ex = S_s[ex_mask]
                thresholds[k - 1] = np.max(S_ex)
            else:
                thresholds[k - 1] = 0.0  # never exercise
            V_s = np.maximum(V_s, payoff)
        V = V_s

    # Ensure thresholds are monotonic non-increasing over time? For a put,
    # the exercise boundary typically increases over time (you're willing to
    # exercise at higher spots as expiration nears). Actually for puts, the
    # threshold (spot below which you exercise) tends to increase toward K as T->0.
    # We'll keep as computed; the tree already encodes optimal behavior.

    # Build policy object
    ex_dates_arr = ex_dates  # 12 dates
    th_arr = thresholds.copy()

    class ExercisePolicy:
        def __init__(self, dates, ths):
            self.dates = dates
            self.ths = ths

        def exercise(self, current_state, time_val):
            """
            current_state: array of spot prices (shape (n_paths,) or similar)
            time_val: current time (scalar)
            Returns boolean vector: True if exercise.
            """
            S = np.asarray(current_state, dtype=float)
            t = float(time_val)
            # Find the nearest exercise date (within tolerance)
            # We exercise only if t is (close to) an exercise date.
            # Find index of exercise date closest to t.
            idx = int(np.argmin(np.abs(self.dates - t)))
            dt_diff = abs(self.dates[idx] - t)
            # Tolerance: allow small numerical mismatch
            if dt_diff > 1e-6:
                # Not at an exercise date; don't exercise
                return np.zeros(S.shape, dtype=bool)
            th = self.ths[idx]
            return S < th

    return ExercisePolicy(ex_dates_arr, th_arr)

# EVOLVE-BLOCK-END
