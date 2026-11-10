import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p**0.5) + 1))]
    n_primes = len(primes)

    WORK_L = 5000
    x = np.arange(1, WORK_L + 1)

    def compute_score(residues):
        counts = np.zeros(WORK_L, dtype=np.int32)
        for i in range(n_primes):
            counts += (x % primes[i] == residues[i]).astype(np.int32)
        mask = counts >= 2
        if mask.all():
            return WORK_L
        return int(np.argmax(~mask))

    # Initial solution from seed
    residues = np.array([((i + 2) // 2) % primes[i] for i in range(n_primes)], dtype=np.int32)
    current_score = compute_score(residues)
    best_score = current_score
    best_residues = residues.copy()

    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    rng = np.random.default_rng(42)

    # Simulated Annealing phase
    while time.time() - start_time < DEADLINE - 30:
        i = rng.integers(0, n_primes)
        old_r = residues[i]
        new_r = rng.integers(0, primes[i])
        if new_r == old_r:
            continue

        residues[i] = new_r
        new_score = compute_score(residues)

        delta = new_score - current_score
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            residues[i] = old_r

        T = max(T_min, T * cooling)

    # Partial greedy improvement phase
    if time.time() - start_time < DEADLINE - 5:
        for i in range(n_primes):
            if time.time() - start_time > DEADLINE - 3:
                break
            best_r = residues[i]
            best_s = compute_score(residues)
            for _ in range(20):
                r = rng.integers(0, primes[i])
                if r == best_r:
                    continue
                residues[i] = r
                s = compute_score(residues)
                if s > best_s:
                    best_s = s
                    best_r = r
            residues[i] = best_r
            if best_s > best_score:
                best_score = best_s
                best_residues = residues.copy()

    return list(best_residues)
