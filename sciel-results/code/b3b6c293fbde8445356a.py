import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s kill

    # Precompute all APs of length >= 2 within [0, N)
    ap_terms = []
    for d in range(1, N):
        for s in range(0, N - d):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # Build incidence matrix M (num_aps x N) as float32
    M = np.zeros((num_aps, N), dtype=np.float32)
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)

    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # C = M.T @ M, K = diag(C)
    C = (M.T @ M).astype(np.float32)
    K = np.diag(C).copy()

    def get_hard_max(sums):
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N + 1

    # Multi-restart SA with 2-flip moves
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000,
             6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
             17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000]

    for seed in seeds:
        if time.time() > deadline - 6:
            break

        rng = random.Random(seed)

        # Initialize with random signs
        signs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            signs[i] = 1.0 if rng.random() < 0.5 else -1.0

        sums = M @ signs
        G = C @ signs

        current_hard_max = get_hard_max(sums)
        if current_hard_max < best_hard_max:
            best_hard_max = current_hard_max
            best_signs = signs.copy()

        # SA parameters
        T = 8.0
        T_min = 0.3
        T_decay = 0.995

        steps = 0
        no_improve = 0
        last_improve_hard = current_hard_max

        # Main SA loop with 1-flip and 2-flip moves
        while time.time() < deadline - 3:
            # Choose move type: 70% single flip, 30% double flip
            use_double = rng.random() < 0.3

            if use_double:
                # 2-flip move
                k1 = rng.randint(0, N - 1)
                k2 = rng.randint(0, N - 1)
                if k1 == k2:
                    use_double = False
                else:
                    # Compute delta for both flips
                    delta1 = -2.0 * signs[k1]
                    delta2 = -2.0 * signs[k2]

                    # Combined effect on sums
                    # New sums = old sums + delta1 * M[:,k1] + delta2 * M[:,k2]
                    # delta_J = 2*(delta1*G[k1] + delta2*G[k2]) + delta1^2*K[k1] + delta2^2*K[k2] + 2*delta1*delta2*(M[:,k1]@M[:,k2])
                    cross = 0.0
                    # Compute cross term: sum over APs that contain both k1 and k2
                    # M[:,k1] @ M[:,k2] = number of APs containing both
                    # We can compute this from C: C[k1,k2] = sum_p M[p,k1]*M[p,k2]
                    cross = C[k1, k2]

                    delta_J = (2.0 * delta1 * G[k1] + 2.0 * delta2 * G[k2] +
                               delta1 * delta1 * K[k1] + delta2 * delta2 * K[k2] +
                               2.0 * delta1 * delta2 * cross)

                    accept = False
                    if delta_J < 0:
                        accept = True
                    else:
                        if delta_J / T > 80.0:
                            accept = False
                        else:
                            accept = rng.random() < math.exp(-delta_J / T)

                    if accept:
                        signs[k1] = -signs[k1]
                        signs[k2] = -signs[k2]
                        sums[pos_to_aps_arr[k1]] += delta1
                        sums[pos_to_aps_arr[k2]] += delta2
                        G += delta1 * C[:, k1] + delta2 * C[:, k2]
                        steps += 1
                    else:
                        use_double = False

            if not use_double:
                # 1-flip move
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]

                accept = False
                if delta_J < 0:
                    accept = True
                else:
                    if delta_J / T > 80.0:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_J / T)

                if accept:
                    signs[k] = -signs[k]
                    sums[pos_to_aps_arr[k]] += delta
                    G += delta * C[:, k]
                    steps += 1

            # Periodically check hard max
            if steps % 30 == 0:
                current_hard_max = get_hard_max(sums)
                if current_hard_max < best_hard_max:
                    best_hard_max = current_hard_max
                    best_signs = signs.copy()
                    no_improve = 0
                    last_improve_hard = current_hard_max
                else:
                    no_improve += 1

                # If no improvement for a while, perturb
                if no_improve > 150:
                    for _ in range(rng.randint(3, 6)):
                        kk = rng.randint(0, N - 1)
                        dd = -2.0 * signs[kk]
                        signs[kk] = -signs[kk]
                        sums[pos_to_aps_arr[kk]] += dd
                        G += dd * C[:, kk]
                    no_improve = 0

            # Annealing
            if steps % 600 == 0:
                T *= T_decay
                if T < T_min:
                    T = T_min

            # Early exit
            if best_hard_max <= 1:
                break

    # Final polish: local search with 1-flip and 2-flip
    if best_signs is not None and time.time() < deadline - 2:
        signs = best_signs.copy()
        sums = M @ signs
        G = C @ signs
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False

            # 1-flip pass
            for k in range(N):
                delta = -2.0 * signs[k]
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                if delta_J < -1e-9:
                    signs[k] = -signs[k]
                    sums[pos_to_aps_arr[k]] += delta
                    G += delta * C[:, k]
                    improved = True

            # 2-flip pass (limited for speed)
            for k1 in range(0, N, 2):  # Sample pairs
                for k2 in range(k1 + 1, min(k1 + 10, N)):
                    delta1 = -2.0 * signs[k1]
                    delta2 = -2.0 * signs[k2]
                    cross = C[k1, k2]
                    delta_J = (2.0 * delta1 * G[k1] + 2.0 * delta2 * G[k2] +
                               delta1 * delta1 * K[k1] + delta2 * delta2 * K[k2] +
                               2.0 * delta1 * delta2 * cross)
                    if delta_J < -1e-9:
                        signs[k1] = -signs[k1]
                        signs[k2] = -signs[k2]
                        sums[pos_to_aps_arr[k1]] += delta1
                        sums[pos_to_aps_arr[k2]] += delta2
                        G += delta1 * C[:, k1] + delta2 * C[:, k2]
                        improved = True

            # Check hard max
            current_hm = get_hard_max(sums)
            if current_hm < best_hard_max:
                best_hard_max = current_hm
                best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
