"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # We need to find a signed step function (values in {-1, 1}) of length N=400
    # that minimizes the upper bound on C3, which is related to the autoconvolution.
    # The task states: "score is the upper bound on C3" and "lower is better".
    # The seed program returns np.ones(N), which gives a score of 1.0.
    # This implies the score for all-ones is 1.0.
    # We need to understand what the "upper bound on C3" is.
    
    # From the context of "Third autocorrelation inequality" and "autoconvolution upper bound",
    # C3 is likely related to the third-order autocorrelation or a specific bound involving it.
    # A common metric in such problems is the peak sidelobe level of the autocorrelation,
    # or the maximum absolute value of the autocorrelation away from the main peak.
    
    # Let's assume the score is the maximum absolute value of the circular autocorrelation
    # excluding the zero-lag term, normalized by N or something similar.
    # For all-ones, the autocorrelation is N at lag 0 and 0 elsewhere (for circular).
    # Wait, for all-ones of length N, the linear autocorrelation is N, N-1, ..., 1, 1, ..., N.
    # The circular autocorrelation is N * delta[k]. So max sidelobe is 0.
    # But the score is 1.0 for all-ones. So it's not simply the max sidelobe.
    
    # Let's re-read: "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    # "score is the upper bound on C3"
    
    # Perhaps C3 is defined as the sum of squares of the autocorrelation sidelobes?
    # Or perhaps it's the L-infinity norm of the autocorrelation sidelobes?
    
    # Let's try to infer from the seed. If all-ones gives 1.0, and we want to minimize,
    # we need a metric that is 1.0 for all-ones.
    
    # One possibility: The metric is (max |autocorr[k]| for k != 0) / (max |autocorr[0]|)
    # For all-ones, this is 0/400 = 0. That doesn't match 1.0.
    
    # Another possibility: The metric is the sum of |autocorr[k]| for k != 0, divided by something.
    # For all-ones, sum of |autocorr[k]| for k != 0 is 0. That doesn't match 1.0.
    
    # Let's consider that the "upper bound on C3" might be a specific formula.
    # In some contexts, C3 might be related to the third moment or a cubic form.
    
    # Given the ambiguity, let's try a different approach.
    # The task says "Choose a signed step function... to improve an autoconvolution upper bound."
    # And "The evaluator measures the raw value (lower is better)."
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    
    # This is key: score = seed_value / value.
    # If seed_value is the value for the seed program (all-ones), and value is the value for our program,
    # then score = 1.0 means value = seed_value.
    # We want score > 1.0, which means value < seed_value.
    
    # So we need to compute the "value" (upper bound on C3) for our sequence, and it should be lower than the seed's value.
    # The seed's value is whatever the metric gives for all-ones.
    
    # Let's assume the metric is the maximum absolute value of the autocorrelation sidelobes.
    # For all-ones, this is 0. So seed_value = 0. Then score = 0/value, which is problematic.
    
    # Maybe the metric includes the main peak? Or maybe it's the L2 norm of the sidelobes?
    # For all-ones, L2 norm of sidelobes is 0.
    
    # Perhaps the metric is defined differently. Let's look at the "Third autocorrelation inequality".
    # This might refer to a specific inequality in signal processing or combinatorics.
    
    # Another idea: The "upper bound on C3" might be a bound on the third-order autocorrelation,
    # which for a binary sequence is related to the number of occurrences of patterns.
    
    # Given the time constraint and ambiguity, let's try a heuristic search that minimizes
    # the peak sidelobe level of the autocorrelation, as this is a common objective.
    # We'll use simulated annealing or hill climbing.
    
    # First, let's define the autocorrelation and a candidate metric.
    # We'll use the maximum absolute value of the circular autocorrelation sidelobes.
    
    def autocorr_sidelobe_max(x):
        n = len(x)
        # Circular autocorrelation
        ac = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))).real
        # Remove the main peak (lag 0)
        ac = np.concatenate([ac[1:], ac[:1]])  # Shift so lag 0 is at the end
        return np.max(np.abs(ac[:-1]))  # Max of sidelobes
    
    # For all-ones, this is 0. So if the seed value is 0, we can't improve.
    # This suggests the metric is not just the sidelobe max.
    
    # Let's try the sum of squares of the sidelobes (energy).
    def autocorr_sidelobe_energy(x):
        n = len(x)
        ac = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))).real
        ac = np.concatenate([ac[1:], ac[:1]])
        return np.sum(ac[:-1]**2)
    
    # For all-ones, this is 0.
    
    # Maybe the metric is the maximum of the autocorrelation including the main peak?
    # For all-ones, that's N=400.
    
    # Or perhaps the metric is normalized: max_sidelobe / N?
    # For all-ones, that's 0.
    
    # I'm stuck on the exact metric. Let's try a different approach.
    # The task mentions "autoconvolution upper bound". Autoconvolution is the same as autocorrelation for real sequences.
    # Maybe C3 is a specific coefficient or bound.
    
    # Let's try to search for a sequence that minimizes the peak sidelobe level,
    # as this is a standard problem. We'll use a simple hill climbing approach.
    
    # Initialize with random signs
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    # Compute initial score
    def compute_score(x):
        # We'll use the peak sidelobe level as a proxy for the "upper bound on C3"
        # This is a guess, but it's a reasonable metric to minimize.
        ac = np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))).real
        ac = np.concatenate([ac[1:], ac[:1]])
        return np.max(np.abs(ac[:-1]))
    
    current_score = compute_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Hill climbing
    for i in range(N):
        if time.time() > deadline:
            break
        # Try flipping bit i
        x[i] *= -1
        new_score = compute_score(x)
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            x[i] *= -1  # Revert
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        x = rng.choice([-1, 1], size=N)
        current_score = compute_score(x)
        
        for i in range(N):
            if time.time() > deadline:
                break
            x[i] *= -1
            new_score = compute_score(x)
            if new_score < current_score:
                current_score = new_score
            else:
                x[i] *= -1
                continue
            # Accept and continue
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
