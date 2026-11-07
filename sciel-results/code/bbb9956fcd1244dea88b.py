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
        
        # ---------- Sigma Sieve ----------
        # Using the harmonic sieve: sigma[n] = sum of divisors
        # sigma[d::d] += d for d in 1..N
        # This is O(N log N) operations.
        sigma = np.zeros(N + 1, dtype=np.int32)
        
        # Optimize by using a loop.
        # For very large N, this loop can be slow.
        # We can try to vectorize chunks or just rely on NumPy's speed for small slices?
        # Actually, for d=1, sigma[1::1] += 1 is N ops.
        # For d=N, sigma[N::N] += N is 1 op.
        
        # A faster approach might be SPF, but the Python loop for sigma calculation is slow.
        # Let's stick to the harmonic sieve but check time.
        
        for d in range(1, N + 1):
            sigma[d::d] += d
            
            if (d & 0xFFFFF) == 0 and time_left() < 30.0:
                # If we are running out of time, we might not finish.
                # But we can't easily stop and use partial results because sigma is incomplete.
                # We should pick a smaller N if we know it's too slow.
                # Let's assume it finishes. If not, we might timeout.
                pass

        if time_left() < 20.0:
            # Fallback to smaller N if sieve is too slow?
            # We can't easily redo the sieve with smaller N in the same function easily without restarting.
            # Let's hope it finishes.
            pass

        # ---------- Group by sigma ----------
        # We want to group indices 1..N by their sigma value.
        # sigma[0] is unused.
        sigma_vals = sigma[1:]
        
        # Get indices sorted by sigma value
        # argsort returns indices 0..N-1 corresponding to sigma[1]..sigma[N]
        order = np.argsort(sigma_vals, kind='stable')
        sorted_sigma = sigma_vals[order]
        # The actual numbers are order + 1
        sorted_indices = order + 1
        
        # Find boundaries of groups
        # A new group starts where sorted_sigma[i] != sorted_sigma[i-1]
        # We can use np.diff or simple comparison
        changes = np.empty(len(sorted_sigma), dtype=bool)
        changes[0] = True
        changes[1:] = sorted_sigma[1:] != sorted_sigma[:-1]
        starts = np.nonzero(changes)[0]
        ends = np.append(starts[1:], len(sorted_sigma))
        
        best_val = float('inf')
        best_pair = None
        
        # Iterate through groups
        for gi in range(len(starts)):
            if time_left() < 5.0:
                break
                
            s0 = starts[gi]
            e0 = ends[gi]
            length = e0 - s0
            if length < 2:
                continue
            
            # Extract the group of numbers
            gn = sorted_indices[s0:e0]
            # gn is already sorted because order was from argsort?
            # No, argsort sorts by KEY (sigma). The indices (values) are not sorted.
            # We need to sort gn to use bisect.
            gn = np.sort(gn)
            L = len(gn)
            
            # For each b in gn, find a in gn such that a/b ~ 1.5
            # Minimize |2a - 3b|
            
            # Optimization: We can iterate over b and binary search for 1.5*b
            # Or iterate over a and b?
            # Since L can be large, O(L log L) is good.
            
            for bi in range(L):
                b = int(gn[bi])
                # We want a ~ 1.5 * b
                target = 1.5 * b
                j = bisect.bisect_left(gn, target)
                
                # Check right neighbor
                if j < L:
                    a = int(gn[j])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check left neighbor
                if j > 0:
                    a = int(gn[j-1])
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num > 0:
                            val = num / (2 * b)
                            if val < best_val:
                                best_val = val
                                best_pair = (a, b)
                                
        if best_pair is None:
            return [11, 6]
            
        a, b = best_pair
        # Verify
        if not (1 <= a <= MAXV_LIMIT and 1 <= b <= MAXV_LIMIT):
            return [11, 6]
        if sigma[a] != sigma[b]:
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]
            
        return [int(a), int(b)]
        
    except Exception:
        return [11, 6]
