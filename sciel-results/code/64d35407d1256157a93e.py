import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    best_M = 333312
    best_ks = [336, 372, 434]

    def update_best(M, ks):
        nonlocal best_M, best_ks
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list

    def search_with_numpy(K):
        """Sieve sigma up to K, compute k*sigma(k), find best collision group."""
        local_start = time.time()
        try:
            sigma = np.zeros(K + 1, dtype=np.int64)

            # Sieve: for each divisor i, add i to multiples
            for i in range(1, K + 1):
                if i % 50000 == 0 and time.time() > deadline - 15:
                    return 1  # timeout
                sigma[i::i] += i

            if time.time() > deadline - 15:
                return 1

            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]

            # Filter: k <= isqrt(M) means k^2 <= M = k*sigma(k) => k <= sigma(k)
            # Actually the constraint is k_i <= isqrt(M) where M = k*sigma(k)
            # So k <= isqrt(k*sigma(k)) => k^2 <= k*sigma(k) => k <= sigma(k)
            # This is always true for k >= 1 since sigma(k) >= 1+k > k for k>1, and sigma(1)=1.
            # Wait: sigma(1) = 1, so k=1: 1 <= isqrt(1) = 1. OK.
            # For k=2: sigma(2)=3, M=6, isqrt(6)=2, k=2<=2. OK.
            # For k=3: sigma(3)=4, M=12, isqrt(12)=3, k=3<=3. OK.
            # For k=4: sigma(4)=7, M=28, isqrt(28)=5, k=4<=5. OK.
            # Actually k <= sigma(k) is always true for k>=1 (sigma(k) >= k+1 for k>1, sigma(1)=1).
            # So the constraint k <= isqrt(M) = isqrt(k*sigma(k)) means k^2 <= k*sigma(k), i.e., k <= sigma(k).
            # This is always satisfied. So we just need M <= 10^18.

            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1

            vals = vals[valid_mask]
            ks = ks[valid_mask]

            # Also need k <= isqrt(M) = isqrt(k*sigma(k)), which as shown is always true.
            # But let's double-check with explicit computation for safety
            # Actually, let's just verify: for each valid k, check k <= isqrt(k*sigma(k))
            # isqrt(k*sigma(k)) >= k iff k^2 <= k*sigma(k) iff k <= sigma(k)
            # sigma(k) >= 1 + k for all k (since 1 and k are both divisors, and for k>1 they're distinct)
            # For k=1: sigma(1)=1, k=1, 1<=1 OK.
            # So always valid. No extra filtering needed.

            if time.time() > deadline - 10:
                return 1

            # Sort and group by value
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]

            # Find groups of equal values
            if len(sorted_vals) == 0:
                return 1

            diffs = np.diff(sorted_vals)
            change_points = np.where(diffs > 0)[0] + 1

            start_points = np.concatenate(([0], change_points))
            end_points = np.concatenate((change_points, [len(sorted_vals)]))

            counts = end_points - start_points

            if len(counts) == 0:
                return 1

            # Find the group with maximum count
            max_idx = np.argmax(counts)
            max_count = int(counts[max_idx])

            if max_count > 1:
                s = int(start_points[max_idx])
                e = int(end_points[max_idx])
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                # Verify all k_i satisfy k_i <= isqrt(M)
                isqrt_M = math.isqrt(M)
                valid_ks = [int(x) for x in ks_group if int(x) <= isqrt_M]
                if len(valid_ks) > len(best_ks):
                    update_best(M, valid_ks)

            return 0  # Success

        except MemoryError:
            print(f"MemoryError for K={K}")
            return 1
        except Exception as e:
            print(f"Error in search for K={K}: {e}")
            return 1

    # Try multiple K values from largest to smallest
    # Memory estimate: sigma is (K+1)*8 bytes, vals is K*8 bytes, sorting adds more
    # For K=20M: ~160MB sigma + 160MB vals + sorting overhead ~500MB total
    # For K=30M: ~240MB + 240MB + overhead ~700MB
    # Be conservative: start with 20M, then try larger if time permits
    target_Ks = [20000000, 30000000, 15000000]

    for K in target_Ks:
        if time.time() > deadline - 30:
            break
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K}: status={status}, elapsed={elapsed:.1f}s, best_count={len(best_ks)}")
        if status == 0:
            # Success - if we found a good answer, try a larger K if time permits
            if time.time() > deadline - 40:
                break

    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(x) for x in best_ks]

    return {'M': best_M, 'k': best_ks}
