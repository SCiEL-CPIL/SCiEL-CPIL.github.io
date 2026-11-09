import numpy as np
import time

def construct():
    deadline = time.time() + 85

    n = 128
    target_lo = 128
    target_hi = 256

    def compute_reps_fft(mask):
        f = mask.astype(np.float64)
        conv = np.fft.irfft(np.fft.rfft(f, n=512) * np.fft.rfft(f, n=512), n=512)
        conv = np.round(conv).astype(np.int64)
        reps = np.zeros(257, dtype=np.int64)
        for t in range(target_lo, target_hi + 1):
            if t % 2 == 0:
                half = t // 2
                if 0 <= half <= n:
                    reps[t] = (conv[t] + (1 if mask[half] else 0)) // 2
                else:
                    reps[t] = conv[t] // 2
            else:
                reps[t] = conv[t] // 2
        return reps

    def is_valid(reps):
        return bool(np.all(reps[target_lo:target_hi+1] > 0))

    def peak_mult(reps):
        return int(np.max(reps[target_lo:target_hi+1]))

    # Seed solution
    mask = np.zeros(n + 1, dtype=bool)
    mask[64:129] = True
    reps = compute_reps_fft(mask)
    best_mask = mask.copy()
    best_peak = peak_mult(reps)

    np.random.seed(42)

    def sa(mask, max_iters=200000):
        current = mask.copy()
        current_reps = compute_reps_fft(current)
        if not is_valid(current_reps):
            return current, peak_mult(current_reps)
        current_peak = peak_mult(current_reps)
        best = current.copy()
        best_peak = current_peak

        T = 3.0
        cooling = 0.99995

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < 0.01:
                break

            A_indices = np.where(current)[0]
            not_A = np.where(~current)[0]

            r = np.random.random()
            if r < 0.4 and len(A_indices) > 15:
                idx = A_indices[np.random.randint(len(A_indices))]
                trial = current.copy()
                trial[idx] = False
            elif r < 0.7 and len(not_A) > 0:
                idx = not_A[np.random.randint(len(not_A))]
                trial = current.copy()
                trial[idx] = True
            else:
                if len(A_indices) == 0 or len(not_A) == 0:
                    continue
                rm = A_indices[np.random.randint(len(A_indices))]
                add = not_A[np.random.randint(len(not_A))]
                trial = current.copy()
                trial[rm] = False
                trial[add] = True

            trial_reps = compute_reps_fft(trial)
            if not is_valid(trial_reps):
                continue

            trial_peak = peak_mult(trial_reps)
            delta = trial_peak - current_peak

            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current = trial
                current_reps = trial_reps
                current_peak = trial_peak
                if current_peak < best_peak:
                    best = current.copy()
                    best_peak = current_peak

            T *= cooling

        return best, best_peak

    # Run SA from seed
    if time.time() < deadline - 10:
        sa_result, sa_peak = sa(best_mask, max_iters=150000)
        if sa_peak < best_peak:
            best_mask = sa_result
            best_peak = sa_peak

    # Random restarts with smaller sets
    for restart in range(15):
        if time.time() > deadline - 5:
            break

        k = 18 + np.random.randint(0, 15)
        mask = np.zeros(n + 1, dtype=bool)
        mask[0] = True
        mask[128] = True
        remaining = k - 2
        if remaining > 0:
            candidates = np.array([i for i in range(1, 128)])
            chosen = np.random.choice(candidates, size=min(remaining, len(candidates)), replace=False)
            mask[chosen] = True

        # Greedily fix validity
        reps = compute_reps_fft(mask)
        for t in range(target_lo, target_hi + 1):
            if time.time() > deadline - 5:
                break
            if reps[t] > 0:
                continue
            found = False
            for a in range(0, min(128, t // 2) + 1):
                b = t - a
                if b > 128 or b < a:
                    continue
                if not mask[a] or not mask[b]:
                    mask[a] = True
                    mask[b] = True
                    reps = compute_reps_fft(mask)
                    found = True
                    break
            if not found:
                if t <= 128:
                    mask[0] = True
                    mask[t] = True
                else:
                    mask[128] = True
                    mask[t - 128] = True
                reps = compute_reps_fft(mask)

        if not is_valid(compute_reps_fft(mask)):
            continue

        sa_result, sa_peak = sa(mask, max_iters=80000)
        if sa_peak < best_peak:
            best_mask = sa_result
            best_peak = sa_peak

    return list(np.where(best_mask)[0])
