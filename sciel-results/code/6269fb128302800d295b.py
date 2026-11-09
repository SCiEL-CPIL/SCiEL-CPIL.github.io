import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak third-order autocorrelation).
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin (150s limit)

    def compute_triple_conv_peak(h):
        """Compute peak of triple convolution (h*h*h)."""
        n = len(h)
        nfft = 1 << (3 * n - 2).bit_length()
        H = np.fft.rfft(h, n=nfft)
        triple = np.fft.irfft(H ** 3, n=nfft)
        # Valid part: indices n-1 to 3n-3 (inclusive), so length 2n-2
        valid = triple[n-1:3*n-2]
        return np.max(np.abs(valid))

    def compute_triple_conv(h):
        """Compute full triple convolution (valid part only)."""
        n = len(h)
        nfft = 1 << (3 * n - 2).bit_length()
        H = np.fft.rfft(h, n=nfft)
        triple = np.fft.irfft(H ** 3, n=nfft)
        valid = triple[n-1:3*n-2]
        return valid

    def greedy_local_search(h, max_iterations=5000, deadline_check=True):
        """Greedy local search with single bit flips and block flips."""
        current_score = compute_triple_conv_peak(h)
        improved = True
        iterations = 0
        no_improve_count = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            no_improve_count = 0

            if deadline_check and time.time() > deadline - 5:
                break

            # Try random single bit flips
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                old_val = h[i]
                h[i] *= -1
                new_score = compute_triple_conv_peak(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                else:
                    h[i] = old_val
                    no_improve_count += 1

            # If stuck, try a block flip
            if no_improve_count >= 30 and np.random.random() < 0.1:
                block_len = np.random.choice([2, 3, 4, 5])
                i = np.random.randint(N - block_len)
                old_block = h[i:i+block_len].copy()
                h[i:i+block_len] *= -1
                new_score = compute_triple_conv_peak(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i:i+block_len] = old_block

        return h, current_score

    def simulated_annealing(h, iterations=3000, T_start=2.0, T_end=0.01):
        """Simulated annealing with single-bit and occasional block flips."""
        current_score = compute_triple_conv_peak(h)
        T = T_start

        for iteration in range(iterations):
            if time.time() > deadline - 15:
                break

            # 90% single flip, 10% block flip
            if np.random.random() < 0.9:
                i = np.random.randint(N)
                old_val = h[i]
                h[i] *= -1
                new_score = compute_triple_conv_peak(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i] = old_val
            else:
                block_len = np.random.choice([2, 3, 4])
                i = np.random.randint(N - block_len)
                old_block = h[i:i+block_len].copy()
                h[i:i+block_len] *= -1
                new_score = compute_triple_conv_peak(h)
                delta = new_score - current_score
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    h[i:i+block_len] = old_block

            T = T_start * (T_end / T_start) ** (iteration / iterations)

        return h, current_score

    best_h = None
    best_score = float('inf')

    # Generate initial sequences
    np.random.seed(42)
    initial_sequences = []

    # Random sequences
    for _ in range(8):
        initial_sequences.append(np.random.choice([-1, 1], N))

    # Thue-Morse
    tm = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        tm[i] = 1 if cnt % 2 == 0 else -1
    initial_sequences.append(tm)

    # Alternating
    initial_sequences.append(np.array([1, -1] * (N // 2)))

    # Block patterns with varying lengths
    initial_sequences.append(np.tile([1, 1, -1, -1], N // 4))
    initial_sequences.append(np.tile([1, -1, -1, 1, 1, -1], N // 6))
    initial_sequences.append(np.tile([1, 1, 1, -1, -1, -1], N // 6))

    # Hadamard-like: use Walsh functions
    def walsh_row(n, k):
        """Generate the k-th row of the Hadamard matrix of size n."""
        row = np.ones(n)
        for j in range(n):
            # Compute the bitwise AND of j and k, count bits
            x = j & k
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            row[j] = 1 if cnt % 2 == 0 else -1
        return row

    # Use several Hadamard rows
    for k in [1, 2, 3, 5, 7]:
        initial_sequences.append(walsh_row(N, k).astype(np.int8))

    # Main search loop
    for init_idx, init_h in enumerate(initial_sequences):
        if time.time() > deadline - 25:
            break

        h = init_h.copy()

        # Phase 1: Simulated annealing
        h, sa_score = simulated_annealing(h, iterations=2000, T_start=2.0, T_end=0.01)

        # Phase 2: Greedy local search
        h, local_score = greedy_local_search(h, max_iterations=3000)

        # Update best
        final_score = compute_triple_conv_peak(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

        # Early termination if we find a very good solution
        if best_score < 0.3:
            break

    if best_h is None:
        best_h = np.random.choice([-1, 1], N)

    # Final polish: run one more round of greedy search on the best
    if time.time() < deadline - 10:
        h = best_h.copy()
        h, final_score = greedy_local_search(h, max_iterations=2000)
        if final_score < best_score:
            best_score = final_score
            best_h = h.copy()

    return best_h
