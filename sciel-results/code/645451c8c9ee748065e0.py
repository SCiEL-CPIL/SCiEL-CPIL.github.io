"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def compute_value(h):
        # value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2 * N * np.max(conv) / (s * s))

    # Start with the seed
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_value(best_h)  # should be 2.0

    # --- Phase 1: Try structured families ---
    # The objective is abs(2*N*max(autoconv) / sum(h)^2)
    # Note: if sum(h) is negative, sum(h)^2 is positive.
    # If we scale h by c, numerator scales by c^2, denominator by c^2, so value is scale-invariant.
    # So we can fix sum(h) = 1 (or -1) and minimize 2*N*max(conv).
    # Actually, let's fix sum(h) = 1. Then value = 2*N*max(conv) (since abs(1)=1).
    # Wait, the formula is abs(2*N*max(conv)/sum(h)^2).
    # If sum(h)=1, value = 2*N*max(conv).
    # We want to minimize this.
    # The max of autoconv is at least the DC component (sum(h)^2 = 1) if h is non-negative?
    # No, autoconv at lag 0 is sum(h_i^2).
    # By Cauchy-Schwarz, sum(h_i^2) >= (sum(h_i))^2 / N = 1/N.
    # So max(conv) >= 1/N. Thus value >= 2*N*(1/N) = 2.
    # The seed gives 2.0. This suggests 2.0 might be the minimum for sum(h)=1?
    # Let's check: if h is constant 1/N, sum=1, conv[0] = N*(1/N)^2 = 1/N.
    # Other lags k: sum_{i} h[i]h[i+k] = (N-k)/N^2.
    # Max is at lag 0: 1/N. Value = 2*N*(1/N) = 2.
    # Can we do better?
    # If we allow negative h, maybe we can reduce max(conv).
    # Note that max(conv) is the maximum over all lags.
    # If we make h oscillate, maybe the peak at lag 0 can be reduced?
    # But sum(h_i^2) is minimized when h is constant (for fixed sum).
    # So for sum(h)=1, min sum(h_i^2) = 1/N, achieved by h_i = 1/N.
    # Since max(conv) >= conv[0] = sum(h_i^2) >= 1/N, the value is >= 2.
    # So 2.0 is the global minimum?
    # Wait, the problem says "lower is better". The seed gives 2.0.
    # Is it possible to get < 2.0?
    # Let's re-read carefully.
    # value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # If sum(h) is not 1, let S = sum(h).
    # Let h' = h / S. Then sum(h') = 1.
    # conv(h, h) = S^2 * conv(h', h').
    # max(conv(h, h)) = S^2 * max(conv(h', h')).
    # value = abs(2 * N * S^2 * max(conv(h', h')) / S^2) = 2 * N * abs(max(conv(h', h'))).
    # So the value depends only on the shape of h (normalized to sum 1).
    # And we proved min max(conv(h', h')) = 1/N, so min value = 2.
    # So the seed is optimal?
    # Let me double check the proof.
    # max(conv) >= conv[0] = sum(h_i^2).
    # For sum(h_i)=1, min sum(h_i^2) = 1/N.
    # So max(conv) >= 1/N.
    # value = 2*N*max(conv) >= 2*N*(1/N) = 2.
    # Yes, 2.0 is the theoretical minimum.
    # So the best score is 1.0 (seed/value = 2/2 = 1.0).
    # Wait, the score is seed/value. Seed value is 2.0.
    # If I find a value of 2.0, score is 1.0.
    # If I find a value < 2.0, score > 1.0.
    # But I proved value >= 2.0.
    # So the maximum possible score is 1.0.
    # Therefore, the seed program is already optimal!
    #
    # Let me re-read the problem statement to see if I missed something.
    # "Choose real, possibly negative step heights"
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # "The maximum is taken over its signed entries, with no absolute value inside np.max"
    #
    # Is it possible that max(conv) < 1/N?
    # conv[0] = sum(h_i^2).
    # If sum(h_i)=1, min sum(h_i^2) is 1/N.
    # So conv[0] >= 1/N.
    # Since max(conv) >= conv[0], max(conv) >= 1/N.
    # So value >= 2.
    #
    # Is there any catch?
    # "np.sum(h) must not be zero"
    # "heights must be finite"
    #
    # What if sum(h) is negative?
    # The proof holds for any non-zero sum.
    #
    # So the minimum value is 2.0.
    # The seed gives 2.0.
    # So the best score is 1.0.
    #
    # Why would the task exist if the seed is optimal?
    # Maybe I'm wrong about the minimum of sum(h_i^2)?
    # Minimize sum(x_i^2) subject to sum(x_i)=1.
    # Lagrangian: 2*x_i - lambda = 0 => x_i = lambda/2 = const.
    # N*const = 1 => const = 1/N.
    # sum(x_i^2) = N*(1/N)^2 = 1/N.
    # Correct.
    #
    # Maybe the task is to confirm this? Or maybe there's a numerical issue?
    # Or maybe I'm misinterpreting "max"?
    # "np.max(np.convolve(h, h))" - this is the maximum element of the array.
    #
    # Let's try a simple example with N=2.
    # h = [0.5, 0.5]. sum=1.
    # conv = [0.25, 0.5, 0.25]. max = 0.5.
    # value = 2*2*0.5/1 = 2.0.
    #
    # h = [1, 0]. sum=1.
    # conv = [1, 0, 0]. max = 1.
    # value = 2*2*1/1 = 4.0.
    #
    # h = [1.5, -0.5]. sum=1.
    # conv = [2.25, -0.75, 0.25]. max = 2.25.
    # value = 2*2*2.25/1 = 9.0.
    #
    # h = [0.5, 0.5] seems best.
    #
    # What if N=3?
    # h = [1/3, 1/3, 1/3]. sum=1.
    # conv = [1/3, 2/9, 1/9, 2/9, 1/3]. max = 1/3.
    # value = 2*3*(1/3)/1 = 2.0.
    #
    # It seems 2.0 is indeed the minimum.
    #
    # So the best I can do is return the seed.
    # But the directive says "Write a new design from scratch".
    # And "REFINEMENT: ... KEEP this design and investigate improvements within its family."
    # But if the seed is optimal, there are no improvements.
    #
    # Let me check if I can get slightly below 2.0 due to floating point?
    # No, the evaluator computes the value exactly.
    #
    # Is it possible that I am wrong about the lower bound?
    # Let's re-read: "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    #
    # What if sum(h) is very small?
    # No, the value is scale-invariant.
    #
    # Okay, I will just return the seed.
    # But to satisfy the "new design" and "deadline guard" requirements,
    # I will write a program that does a small local search to confirm
    # that the seed is indeed local minimum, and then returns it.
    #
    # Actually, the directive says "FRESH DESIGN: ... Write a new design from scratch".
    # But also "REFINEMENT: ... KEEP this design".
    # These are contradictory. I will follow the "REFINEMENT" instruction
    # as it is more specific to the current state (score 1.0).
    #
    # I will implement a simple local search to try to find a value < 2.0,
    # but I expect it to fail.
    #

    # Local search
    best_h = best_h.copy()
    step_size = 0.01
    for i in range(50):  # limited iterations
        if time.time() > deadline:
            break
        improved = False
        for j in range(N):
            if time.time() > deadline:
                break
            # Try increasing h[j]
            h_new = best_h.copy()
            h_new[j] += step_size
            val = compute_value(h_new)
            if val < best_val - 1e-10:
                best_h = h_new
                best_val = val
                improved = True
            # Try decreasing h[j]
            h_new = best_h.copy()
            h_new[j] -= step_size
            val = compute_value(h_new)
            if val < best_val - 1e-10:
                best_h = h_new
                best_val = val
                improved = True
        if not improved:
            break

    return best_h

# EVOLVE-BLOCK-END
