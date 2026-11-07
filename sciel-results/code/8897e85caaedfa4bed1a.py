import time
import bisect
import numpy as np

MAXV_LIMIT = 10000000
WALL_LIMIT = 90.0
SAFETY = 0.85

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    try:
        N = MAXV_LIMIT

        # ---------- Linear Sieve for Sigma (O(N)) ----------
        spf = [0] * (N + 1)
        sigma = [0] * (N + 1)
        cnt = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        p_power = [0] * (N + 1)

        sigma[1] = 1
        sigma_pow[1] = 1
        cnt[1] = 0
        p_power[1] = 1

        primes = []

        # Optimize local variables
        spf_l = spf
        sigma_l = sigma
        cnt_l = cnt
        sigma_pow_l = sigma_pow
        p_power_l = p_power
        primes_l = primes

        for i in range(2, N + 1):
            if spf_l[i] == 0:
                spf_l[i] = i
                primes_l.append(i)
                sigma_l[i] = 1 + i
                sigma_pow_l[i] = 1 + i
                cnt_l[i] = 1
                p_power_l[i] = i

            for p in primes_l:
                ip = i * p
                if ip > N or p > spf_l[i]:
                    break
                spf_l[ip] = p
                if p == spf_l[i]:
                    cnt_l[ip] = cnt_l[i] + 1
                    p_power_l[ip] = p_power_l[i] * p
                    sigma_pow_l[ip] = sigma_pow_l[i] + p_power_l[ip]
                    sigma_l[ip] = sigma_l[i] // sigma_pow_l[i] * sigma_pow_l[ip]
                else:
                    cnt_l[ip] = 1
                    p_power_l[ip] = p
                    sigma_pow_l[ip] = 1 + p
                    sigma_l[ip] = sigma_l[i] * (1 + p)

        # ---------- Convert to numpy array for fast sorting ----------
        sigma_arr = np.array(sigma[1:], dtype=np.int64)
        indices = np.arange(1, N + 1, dtype=np.int64)

        # Sort by (sigma, index) so groups are contiguous AND sorted by index
        order = np.lexsort((indices, sigma_arr))
        sorted_sigma = sigma_arr[order]
        sorted_indices = indices[order]

        # Find group boundaries
        changes = np.empty(len(sorted_sigma), dtype=bool)
        changes[0] = True
        changes[1:] = sorted_sigma[1:] != sorted_sigma[:-1]
        starts = np.nonzero(changes)[0]
        ends = np.append(starts[1:], len(sorted_sigma))

        # Create list of groups
        groups = []
        for gi in range(len(starts)):
            s0 = starts[gi]
            e0 = ends[gi]
            if e0 - s0 >= 2:
                gn = sorted_indices[s0:e0]
                # Store max element to sort groups later
                groups.append((int(gn[-1]), gn))

        # Sort groups by max element descending
        # This ensures we check large b first
        groups.sort(key=lambda x: x[0], reverse=True)

        best_val = float('inf')
        best_pair = None

        for max_val, gn in groups:
            if time_left() < 5.0:
                break

            L = len(gn)
            # gn is sorted ascending. We want to iterate b descending.
            # But bisect requires sorted array.
            # We can just iterate indices in reverse.
            
            # Optimization: If the minimum possible error for this group is > best_val, skip?
            # Min error is roughly 1 / (2 * max_b).
            # If 1 / (2 * max_val) > best_val, then even the best possible pair in this group is worse than current best.
            # So we can break entirely if we process groups in descending order of max_val.
            if 1.0 / (2.0 * max_val) > best_val:
                break

            for bi in range(L - 1, -1, -1):
                b = int(gn[bi])
                target = 1.5 * b
                j = bisect.bisect_left(gn, target)

                if j < L:
                    a = int(gn[j])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)

                if j > 0:
                    a = int(gn[j - 1])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Early exit if we hit a very small error?
                if best_val < 1e-8:
                    break

        if best_pair is None:
            return [11, 6]

        a, b = best_pair
        if not (1 <= a <= N and 1 <= b <= N):
            return [11, 6]
        if sigma[a] != sigma[b]:
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]

        return [int(a), int(b)]

    except Exception:
        return [11, 6]
