import time
import random
import numpy as np
import math

def _primes_up_to(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    N = 50000

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill

    # State
    r = np.zeros(n_primes, dtype=np.int32)
    cov = np.zeros(N + 1, dtype=np.int8)

    # Helper to rebuild cov from r
    def build_cov():
        cov[:] = 0
        for i, p in enumerate(primes):
            rr = r[i]
            if rr == 0:
                cov[p::p] += 1
            else:
                cov[rr::p] += 1
        cov[0] = 100  # Ignore 0

    # Helper to get L
    def get_L():
        # Find first index in 1..N where cov < 2
        bad = np.nonzero(cov[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0]

    # Helper to apply change
    def apply_change(i, new_r):
        p = primes[i]
        old_r = r[i]
        if old_r == new_r:
            return
        # Remove old
        if old_r == 0:
            cov[p::p] -= 1
        else:
            cov[old_r::p] -= 1
        # Add new
        r[i] = new_r
        if new_r == 0:
            cov[p::p] += 1
        else:
            cov[new_r::p] += 1

    # Helper to revert
    def revert_change(i, old_r):
        p = primes[i]
        new_r = r[i]
        if old_r == new_r:
            return
        # Remove new
        if new_r == 0:
            cov[p::p] -= 1
        else:
            cov[new_r::p] -= 1
        # Add old
        r[i] = old_r
        if old_r == 0:
            cov[p::p] += 1
        else:
            cov[old_r::p] += 1

    # --- Phase 1: Greedy L-extension ---
    # Start with all residues = 0, then try to extend L
    build_cov()
    L = get_L()

    random.seed(42)
    max_extension_attempts = 500
    attempts = 0
    no_progress = 0

    while L < N and time.time() < deadline and attempts < max_extension_attempts:
        attempts += 1
        x = L + 1
        candidates = []

        for i, p in enumerate(primes):
            target_r = x % p
            if r[i] == target_r:
                continue

            old_r = r[i]

            # Check if removing old_r coverage breaks any y < x
            # y must be in [1, x-1] and y % p == old_r
            # cov[y] must be >= 3

            if old_r == 0:
                if x <= p:
                    candidates.append(i)
                    continue
                # Check min cov in p, 2p, ... < x
                if x > p:
                    min_cov = np.min(cov[p:x:p])
                    if min_cov >= 3:
                        candidates.append(i)
            else:
                if old_r >= x:
                    candidates.append(i)
                    continue
                # Check min cov in old_r, old_r+p, ... < x
                if old_r >= 1:
                    min_cov = np.min(cov[old_r:x:p])
                    if min_cov >= 3:
                        candidates.append(i)

        if candidates:
            i = random.choice(candidates)
            apply_change(i, x % primes[i])
            new_L = get_L()
            if new_L > L:
                L = new_L
                no_progress = 0
            else:
                no_progress += 1
                if no_progress > 20:
                    break
        else:
            break

    best_r = r.copy()
    best_L = L

    # --- Phase 2: Backward refinement (polish) ---
    if time.time() < deadline:
        def _build_count_array(residues, primes, max_L):
            counts = np.zeros(max_L + 1, dtype=np.int32)
            for i in range(len(primes)):
                rr = residues[i]
                p = primes[i]
                if rr == 0:
                    rr_eff = p
                else:
                    rr_eff = rr
                if rr_eff <= max_L:
                    positions = np.arange(rr_eff, max_L + 1, p)
                    counts[positions] += 1
            return counts

        def _find_L_from_counts(counts, max_L):
            deficient = np.where(counts[1:max_L + 1] < 2)[0]
            if len(deficient) == 0:
                return max_L
            return int(deficient[0])

        max_L_check = min(20000, N)
        residues = best_r.tolist()
        current_score = _find_L_from_counts(_build_count_array(residues, primes, max_L_check), max_L_check)

        improved = True
        passes = 0
        while improved and time.time() < deadline and passes < 3:
            passes += 1
            improved = False
            for idx in range(n_primes - 1, -1, -1):
                if time.time() > deadline:
                    break
                p = primes[idx]
                old_r = residues[idx]

                # Build counts without this prime
                counts = _build_count_array(residues, primes, max_L_check)
                r_eff = p if old_r == 0 else old_r
                if r_eff <= max_L_check:
                    positions = np.arange(r_eff, max_L_check + 1, p)
                    counts[positions] -= 1

                best_r_local = old_r
                best_local_score = _find_L_from_counts(counts, max_L_check)

                # Try all residues for this prime
                for new_r in range(p):
                    if time.time() > deadline:
                        break
                    r_eff_new = p if new_r == 0 else new_r
                    if r_eff_new <= max_L_check:
                        positions = np.arange(r_eff_new, max_L_check + 1, p)
                        counts[positions] += 1

                    new_score = _find_L_from_counts(counts, max_L_check)

                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_r_local = new_r

                    if r_eff_new <= max_L_check:
                        positions = np.arange(r_eff_new, max_L_check + 1, p)
                        counts[positions] -= 1

                if best_r_local != old_r:
                    residues[idx] = best_r_local
                    improved = True

        # Update best
        new_score = _find_L_from_counts(_build_count_array(residues, primes, max_L_check), max_L_check)
        if new_score > best_L:
            best_L = new_score
            best_r = np.array(residues, dtype=np.int32)

    # --- Phase 3: Simulated Annealing ---
    if time.time() < deadline:
        r = best_r.copy()
        build_cov()
        L = get_L()
        current_L = L
        best_L_sa = L
        best_r_sa = r.copy()

        T = 5.0
        cooling_rate = 0.998
        min_T = 0.01
        restart_count = 0
        max_restarts = 10

        while time.time() < deadline and T > min_T:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = r[i]
            new_r = random.randint(0, p - 1)

            if old_r == new_r:
                continue

            apply_change(i, new_r)
            new_L = get_L()

            delta = new_L - current_L

            if delta > 0 or random.random() < math.exp(delta / T):
                current_L = new_L
                if current_L > best_L_sa:
                    best_L_sa = current_L
                    best_r_sa = r.copy()
            else:
                revert_change(i, old_r)

            T *= cooling_rate

            if T < min_T:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart from best
                r = best_r_sa.copy()
                build_cov()
                current_L = best_L_sa
                L = current_L
                T = 5.0

        if best_L_sa > best_L:
            best_L = best_L_sa
            best_r = best_r_sa.copy()

    # Final verification with full range
    def _score_full(residues_list, primes, max_L=100000):
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(len(primes)):
            rr = residues_list[i]
            p = primes[i]
            if rr == 0:
                rr_eff = p
            else:
                rr_eff = rr
            if rr_eff <= max_L:
                positions = np.arange(rr_eff, max_L + 1, p)
                counts[positions] += 1
        deficient = np.where(counts[1:max_L + 1] < 2)[0]
        if len(deficient) == 0:
            return max_L
        return int(deficient[0])

    final_L = _score_full(best_r.tolist(), primes, 100000)

    return [int(x) for x in best_r.tolist()]
