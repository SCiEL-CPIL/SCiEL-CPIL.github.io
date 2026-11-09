import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (150s limit)

    def compute_peak_sidelobe(h):
        """
        Computes the maximum absolute value of the autocorrelation
        at non-zero lags using linear autocorrelation.
        """
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        # Linear autocorrelation: R[0..n-1] are positive lags, R[nfft-n+1..nfft-1] negative lags
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        return np.max(np.abs(all_sidelobes))

    def compute_power_spectrum(h):
        """Compute the power spectrum for spectral analysis."""
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        return np.abs(H) ** 2

    def thue_morse_sequence(n):
        """Generate Thue-Morse sequence of length n, mapped to ±1."""
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits set in binary representation of i
            x = i
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            seq[i] = 1 if cnt % 2 == 0 else -1
        return seq

    def m_sequence_like(n):
        """Generate a pseudo-random sequence with good autocorrelation properties."""
        # Use a simple LFSR-like approach or just random with structure
        np.random.seed(123)
        seq = np.random.choice([-1, 1], n)
        # Apply some structure: ensure balanced
        if seq.sum() > 0:
            idx = np.where(seq == 1)[0]
            if len(idx) > 0:
                seq[idx[0]] = -1
        return seq

    best_h = None
    best_score = float('inf')

    # Generate diverse initial sequences
    initial_sequences = []
    initial_sequences.append(thue_morse_sequence(N))
    initial_sequences.append(m_sequence_like(N))
    np.random.seed(42)
    for _ in range(20):
        initial_sequences.append(np.random.choice([-1, 1], N))

    def greedy_local_search(h, max_iterations=5000, deadline_check=True):
        """
        Greedy local search with multiple move types:
        1. Single bit flip
        2. Swap two bits
        3. Swap adjacent bits
        """
        current_score = compute_peak_sidelobe(h)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            if deadline_check and time.time() > deadline - 5:
                break

            # Try random single bit flips
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1

            # Try random swaps
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                i, j = np.random.choice(N, 2, replace=False)
                h[i], h[j] = h[j], h[i]
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i], h[j] = h[j], h[i]

            # Try adjacent swaps
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N - 1)
                h[i], h[i+1] = h[i+1], h[i]
                new_score = compute_peak_sidelobe(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i], h[i+1] = h[i+1], h[i]

        return h, current_score

    def simulated_annealing(h, iterations=3000, T_start=2.0, T_end=0.05):
        """
        Simulated annealing with a mix of flip and swap moves.
        """
        current_score = compute_peak_sidelobe(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 10:
                break

            # Choose move type
            move_type = np.random.choice(['flip', 'swap', 'adj_swap'], p=[0.5, 0.3, 0.2])

            if move_type == 'flip':
                i = np.random.randint(N)
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i] *= -1

            elif move_type == 'swap':
                i, j = np.random.choice(N, 2, replace=False)
                h[i], h[j] = h[j], h[i]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i], h[j] = h[j], h[i]

            else:  # adj_swap
                i = np.random.randint(N - 1)
                h[i], h[i+1] = h[i+1], h[i]
                new_score = compute_peak_sidelobe(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i], h[i+1] = h[i+1], h[i]

            # Cooling
            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    # Main search loop: try each initial sequence with SA + local polish
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 15:
            break

        h = init_h.copy()

        # Phase 1: Simulated annealing
        h, sa_score = simulated_annealing(h, iterations=2000, T_start=2.0, T_end=0.05)

        # Phase 2: Greedy local search
        h, local_score = greedy_local_search(h, max_iterations=3000)

        # Update best
        final_score = compute_peak_sidelobe(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination
        if best_score < 0.3:
            break

    if best_h is None:
        best_h = np.random.choice([-1, 1], N)

    return best_h
