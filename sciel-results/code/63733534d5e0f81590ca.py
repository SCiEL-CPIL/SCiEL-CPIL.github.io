import numpy as np
import time

def construct():
    start = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    limits = [10**7, 2*10**7]

    for N in limits:
        if time.time() - start > deadline - 25:
            break

        try:
            # SPF (smallest prime factor) sieve
            spf = np.arange(N + 1, dtype=np.int64)
            sqrt_N = int(np.sqrt(N))
            for i in range(2, sqrt_N + 1):
                if spf[i] == i:
                    for j in range(i * i, N + 1, i):
                        if spf[j] == j:
                            spf[j] = i

            # Compute sigma via SPF: for n = p^a * m with p not dividing m,
            # sigma(n) = sigma(m) * (1 + p + ... + p^a)
            rem = np.zeros(N + 1, dtype=np.int64)
            cnt = np.zeros(N + 1, dtype=np.int64)
            sigma = np.ones(N + 1, dtype=np.int64)

            for n in range(2, N + 1):
                p = spf[n]
                m = n // p
                if m >= 2 and spf[m] == p:
                    rem[n] = rem[m]
                    cnt[n] = cnt[m] + 1
                else:
                    rem[n] = m
                    cnt[n] = 1

                # Compute 1 + p + p^2 + ... + p^a
                a = cnt[n]
                geo = 1
                for _ in range(a):
                    geo = geo * p + 1
                sigma[n] = sigma[rem[n]] * geo

            # Compute M = k * sigma(k)
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]

            # Filter M <= 10^18
            mask = M_vals <= 10**18
            M_valid = M_vals[mask]
            k_valid = k_vals[mask]

            # Group by M, find max frequency
            unique_M, inverse_indices, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
            max_count = np.max(counts)
            if max_count > best_count:
                idx = np.argmax(counts)
                target_M = unique_M[idx]
                k_solutions = k_valid[inverse_indices == idx]

                best_M = int(target_M)
                best_k = [int(x) for x in k_solutions]
                best_count = int(max_count)

        except MemoryError:
            break
        except Exception:
            break

    return {'M': best_M, 'k': best_k}
