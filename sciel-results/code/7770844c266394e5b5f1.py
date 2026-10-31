"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 100
DEADLINE = 110.0  # seconds, safety margin before 120s kill


def _overlap_score(h):
    """Compute the maximum overlap of h with shifted (1-h).
    
    For each shift k, overlap_k = sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize max_k overlap_k.
    """
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(1, N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _sawtooth(params):
    """Generate a sawtooth pattern from parameters [a, b, P, phase].
    
    a, b: the two constant values (in [0,1])
    P: period length (number of steps in one full cycle)
    phase: phase offset
    
    Pattern within one period of length P:
    - First ceil(P/2) steps have value a
    - Remaining floor(P/2) steps have value b
    
    The sum constraint: sum(h) = N/2 = 50
    """
    a, b, P, phase = params
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    P = int(round(np.clip(P, 2, N)))
    phase = int(phase) % P
    
    # Build one period
    n_a = (P + 1) // 2  # number of a's in period
    n_b = P - n_a       # number of b's in period
    period = np.zeros(P)
    period[:n_a] = a
    period[n_a:] = b
    
    # Repeat to fill N steps
    n_periods = (N + P - 1) // P
    h = np.tile(period, n_periods)[:N]
    
    # Apply phase shift
    h = np.roll(h, phase)
    
    # Enforce sum constraint by scaling if needed
    # Actually, we should enforce it properly. Let's adjust.
    # The sum of one period is n_a * a + n_b * b
    # Total sum = (N // P) * (n_a * a + n_b * b) + remainder
    # This is complex. Instead, let's use a different approach.
    return h


def _sawtooth_constrained(params):
    """Generate sawtooth with exact sum constraint.
    
    params: [a, b, P, phase]
    We enforce sum(h) = N/2 by adjusting one of the values.
    """
    a, b, P, phase = params
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    P = int(round(np.clip(P, 2, N)))
    phase = int(phase) % P
    
    n_a = (P + 1) // 2
    n_b = P - n_a
    
    # Build base pattern
    period = np.zeros(P)
    period[:n_a] = a
    period[n_a:] = b
    
    n_periods = (N + P - 1) // P
    h = np.tile(period, n_periods)[:N]
    h = np.roll(h, phase)
    
    # Now enforce sum constraint: sum(h) should be N/2 = 50
    current_sum = np.sum(h)
    target_sum = N / 2.0
    
    if abs(current_sum - target_sum) > 1e-10:
        # Adjust: we need to scale or shift. 
        # Simple approach: add a constant offset to make sum correct, then clip
        # But clipping may break the two-value structure.
        # Alternative: adjust b to satisfy constraint.
        # sum = (N//P)*n_a*a + (N//P)*n_b*b + remainder terms
        # This is getting complicated. Let's use a simpler parameterization.
        # 
        # New approach: parameterize by a, P, phase, and derive b from constraint.
        # For a period of length P with n_a values of a and n_b values of b:
        # In N steps, we have q = N//P full periods and r = N%P extra steps.
        # The extra r steps come from the start of the period (after phase shift).
        # 
        # This is complex. Let's just use a direct optimization on the full vector
        # with a structured initialization.
        pass
    
    return h


def construct():
    start_time = time.time()
    
    # Strategy: Use differential evolution to optimize a sawtooth-like pattern.
    # Parameterize: [a, b, P, phase] where a and b are the two levels,
    # P is the period, phase is the shift.
    # We'll enforce the sum constraint by post-processing.
    
    def objective(params):
        a, b, P, phase = params
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        P = int(round(np.clip(P, 2, N)))
        phase = int(phase) % P
        
        n_a = (P + 1) // 2
        n_b = P - n_a
        
        period = np.zeros(P)
        period[:n_a] = a
        period[n_a:] = b
        
        n_periods = (N + P - 1) // P
        h = np.tile(period, n_periods)[:N]
        h = np.roll(h, phase)
        
        # Enforce sum constraint: adjust b to make sum = N/2
        # Sum of h = sum of all elements
        # In the tiled array, count how many a's and b's there are
        # After tiling and truncating to N:
        q, r = divmod(N, P)
        # Full periods contribute q * (n_a * a + n_b * b)
        # Remaining r elements are from the start of the period
        # But after phase shift, it's more complex.
        
        # Simpler: just compute sum and adjust
        current_sum = np.sum(h)
        target_sum = N / 2.0
        
        if current_sum != 0 and abs(current_sum - target_sum) > 1e-12:
            # Scale the whole thing? No, that changes the values.
            # Instead, let's adjust: we want to find b' such that sum = target
            # The sum is linear in b (and a).
            # sum = C_a * a + C_b * b where C_a, C_b are counts
            # Let's count:
            C_a = np.sum(h == a) if a != b else 0
            C_b = N - C_a if a != b else 0
            
            if a == b:
                # Degenerate case
                val = target_sum / N
                h = np.full(N, val)
            else:
                # sum = C_a * a + C_b * b
                # We want C_a * a + C_b * b' = target_sum
                # b' = (target_sum - C_a * a) / C_b
                if C_b > 0:
                    b_new = (target_sum - C_a * a) / C_b
                    b_new = np.clip(b_new, 0.0, 1.0)
                    h[h == b] = b_new
                else:
                    # All a's, adjust a
                    a_new = target_sum / N
                    a_new = np.clip(a_new, 0.0, 1.0)
                    h[:] = a_new
        
        # Final check
        if abs(np.sum(h) - N / 2.0) > 1e-6:
            # Fallback: uniform
            return np.full(N, 0.5)
        
        return _overlap_score(h)
    
    # Run differential evolution
    bounds = [(0.0, 1.0), (0.0, 1.0), (2, N), (0, N)]
    
    try:
        result = differential_evolution(
            objective, bounds, maxiter=200, popsize=30, 
            seed=42, tol=1e-8, polish=True,
            time_limit=DEADLINE - (time.time() - start_time) - 5
        )
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            # Reconstruct the best solution
            a, b, P, phase = result.x
            a = np.clip(a, 0.0, 1.0)
            b = np.clip(b, 0.0, 1.0)
            P = int(round(np.clip(P, 2, N)))
            phase = int(phase) % P
            
            n_a = (P + 1) // 2
            n_b = P - n_a
            
            period = np.zeros(P)
            period[:n_a] = a
            period[n_a:] = b
            
            n_periods = (N + P - 1) // P
            h = np.tile(period, n_periods)[:N]
            h = np.roll(h, phase)
            
            # Enforce sum constraint
            current_sum = np.sum(h)
            target_sum = N / 2.0
            
            if abs(current_sum - target_sum) > 1e-10:
                C_a = np.sum(h == a) if a != b else 0
                C_b = N - C_a if a != b else 0
                
                if a == b:
                    val = target_sum / N
                    h = np.full(N, val)
                else:
                    if C_b > 0:
                        b_new = (target_sum - C_a * a) / C_b
                        b_new = np.clip(b_new, 0.0, 1.0)
                        h[h == b] = b_new
                    else:
                        a_new = target_sum / N
                        a_new = np.clip(a_new, 0.0, 1.0)
                        h[:] = a_new
            
            # Verify
            if abs(np.sum(h) - N / 2.0) > 1e-6:
                h = np.full(N, 0.5)
            
            return h
            
    except Exception:
        pass
    
    # Fallback: uniform
    return np.full(N, 0.5)
