import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with a+3d <= 256
    # Store as 0-indexed
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    def count_mono(x):
        """Count monochromatic 4-APs for binary vector x (0-indexed, len 256)."""
        xi = x[aps_arr[:, 0]]
        xj = x[aps_arr[:, 1]]
        xk = x[aps_arr[:, 2]]
        xl = x[aps_arr[:, 3]]
        eq_ij = 1 - xi - xj + 2 * xi * xj
        eq_ik = 1 - xi - xk + 2 * xi * xk
        eq_il = 1 - xi - xl + 2 * xi * xl
        mono = eq_ij * eq_ik * eq_il
        return int(mono.sum())

    def thue_morse(n):
        """Return Thue-Morse sequence of length n (0-indexed)."""
        tm = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm

    def greedy_local_search(x, max_iters=500):
        """Greedy local search: flip bits that reduce mono count."""
        x = x.copy().astype(np.int8)
        current = count_mono(x)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(N):
                old = x[i]
                x[i] = 1 - old
                new = count_mono(x)
                if new < current:
                    current = new
                    improved = True
                else:
                    x[i] = old
        return x, current

    def simulated_annealing(x0, start_temp=1.0, end_temp=0.01, n_iters=15000):
        """Simulated annealing on binary vector."""
        x = x0.copy().astype(np.int8)
        current = count_mono(x)
        best = x.copy()
        best_count = current
        T = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / n_iters)
        
        for i in range(n_iters):
            T *= cooling
            pos = np.random.randint(N)
            x[pos] = 1 - x[pos]
            new = count_mono(x)
            delta = new - current
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = new
                if current < best_count:
                    best = x.copy()
                    best_count = current
            else:
                x[pos] = 1 - x[pos]
        
        return best, best_count

    rng = np.random.RandomState(42)

    initializations = []

    # Thue-Morse
    tm = thue_morse(N)
    initializations.append(tm.astype(np.int8))

    # Random initializations
    for _ in range(3):
        initializations.append(rng.randint(0, 2, N).astype(np.int8))

    best_overall = None
    best_overall_count = float('inf')

    for init in initializations:
        if time.time() > deadline - 20:
            break

        # Run simulated annealing
        sa_result, sa_count = simulated_annealing(init, start_temp=1.0, end_temp=0.01, n_iters=15000)

        # Polish with greedy local search
        polished, polished_count = greedy_local_search(sa_result, max_iters=300)

        if polished_count < best_overall_count:
            best_overall_count = polished_count
            best_overall = polished.copy()

    # Final polish
    if best_overall is not None:
        final, final_count = greedy_local_search(best_overall, max_iters=500)
        if final_count < best_overall_count:
            best_overall = final
            best_overall_count = final_count

    if best_overall is None:
        return [(i // 2) % 2 for i in range(256)]

    return list(best_overall.astype(int))
