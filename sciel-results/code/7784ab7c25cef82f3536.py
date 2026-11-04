"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    rng = np.random.RandomState(42)

    # Precompute FFT size
    fft_size = 1 << (2 * N - 1).bit_length()

    def get_cost(h):
        # Compute aperiodic autocorrelation via FFT
        F = np.fft.rfft(h, n=fft_size)
        corr = np.fft.irfft(F * np.conj(F), n=fft_size)[:N]
        # Sum of squares of all lags (including lag 0)
        return np.sum(corr**2)

    # Phase 1: Simulated Annealing from random initialization
    h = rng.choice([-1.0, 1.0], size=N)
    current_cost = get_cost(h)
    best_cost = current_cost
    best_h = h.copy()

    temperature = 500.0
    cooling_rate = 0.9999
    min_temperature = 1.0

    # Phase 1 deadline: use 60% of total time
    phase1_deadline = start_time + (deadline - start_time) * 0.6

    while time.time() < phase1_deadline:
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val

        new_cost = get_cost(h)
        delta = new_cost - current_cost

        if delta < 0:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            if temperature > min_temperature:
                prob = np.exp(-delta / temperature)
                if rng.random_sample() < prob:
                    current_cost = new_cost
                else:
                    h[i] = old_val
            else:
                h[i] = old_val

        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature

    # Phase 2: Multi-start local search (hill climbing) from various initializations
    # Use remaining time for multiple restarts
    remaining_time = deadline - time.time()
    
    # Estimate how many restarts we can afford
    # Each restart does a quick hill climb
    num_restarts = max(1, int(remaining_time / 2.0))

    for r in range(num_restarts):
        if time.time() >= deadline:
            break

        # New random initialization
        h_restart = rng.choice([-1.0, 1.0], size=N)
        current_cost = get_cost(h_restart)

        # Quick local search on this restart
        restart_deadline = min(deadline, time.time() + max(0.5, (deadline - time.time()) / (num_restarts - r + 1)))
        
        while time.time() < restart_deadline:
            i = rng.randint(N)
            old_val = h_restart[i]
            h_restart[i] = -old_val

            new_cost = get_cost(h_restart)

            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_h = h_restart.copy()
            else:
                h_restart[i] = old_val

    return best_h
# EVOLVE-BLOCK-END
