import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def get_min_A(h):
        """Calculate the minimum periodic autocorrelation value."""
        H = np.fft.fft(h)
        A = np.fft.ifft(H * np.conj(H)).real
        return np.min(A)

    # Initialize with a random binary sequence with sum N/2
    h = np.zeros(N, dtype=np.float64)
    indices = np.random.choice(N, N // 2, replace=False)
    h[indices] = 1.0

    best_min_A = get_min_A(h)
    best_h = h.copy()

    # Local search to maximize the minimum autocorrelation
    while time.time() < deadline:
        # Compute current autocorrelations
        H = np.fft.fft(best_h)
        A = np.fft.ifft(H * np.conj(H)).real
        current_min_A = np.min(A)

        # Identify ones and zeros
        ones = np.where(best_h > 0.5)[0]
        zeros = np.where(best_h < 0.5)[0]

        if len(ones) == 0 or len(zeros) == 0:
            break

        improved = False
        # Try a number of random flips
        # Adaptive number of trials based on remaining time?
        # Fixed 100 trials per iteration
        for _ in range(100):
            if time.time() > deadline:
                break

            i1 = np.random.choice(ones)
            i2 = np.random.choice(zeros)

            # Flip bits
            best_h[i1] = 1.0 - best_h[i1]
            best_h[i2] = 1.0 - best_h[i2]

            new_min_A = get_min_A(best_h)

            if new_min_A > current_min_A:
                current_min_A = new_min_A
                improved = True
                # Update sets
                ones = np.where(best_h > 0.5)[0]
                zeros = np.where(best_h < 0.5)[0]
            else:
                # Revert
                best_h[i1] = 1.0 - best_h[i1]
                best_h[i2] = 1.0 - best_h[i2]

        if not improved:
            # Perturb to escape local minima
            # Randomly flip 10 pairs
            for _ in range(10):
                i1 = np.random.choice(ones)
                i2 = np.random.choice(zeros)
                best_h[i1] = 1.0 - best_h[i1]
                best_h[i2] = 1.0 - best_h[i2]
            ones = np.where(best_h > 0.5)[0]
            zeros = np.where(best_h < 0.5)[0]

    return best_h
