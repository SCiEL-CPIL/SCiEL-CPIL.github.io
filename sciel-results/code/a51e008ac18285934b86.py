import time
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
        p_power = [0] * (N + 1)  # p^(cnt[i]) where p = spf[i]

        sigma[1] = 1
        sigma_pow[1] = 1
        cnt[1] = 0
        p_power[1] = 1

        primes = []

        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                sigma[i] = 1 + i
                sigma_pow[i] = 1 + i
                cnt[i] = 1
                p_power[i] = i

            for p in primes:
                ip = i * p
                if ip > N or p > spf[i]:
                    break
                spf[ip] = p
                if p == spf[i]:
                    cnt[ip] = cnt[i] + 1
                    p_power[ip] = p_power[i] * p
                    sigma_pow[ip] = sigma_pow[i] + p_power[ip]
                    sigma[ip] = sigma[i] // sigma_pow[i] * sigma_pow[ip]
                else:
                    cnt[ip] = 1
                    p_power[ip] = p
                    sigma_pow[ip] = 1 + p
                    sigma[ip] = sigma[i] * (1 + p)

        # Convert to numpy array for faster access?
        # Python list access is fast enough for random access in a tight loop?
        # Numpy array access is also fast.
        # Let's stick to list for simplicity, or convert if needed.
        # List of ints is fine.
        
        best_val = float('inf')
        best_pair = (11, 6) # Fallback: sigma(11)=12, sigma(6)=12, |11/6 - 1.5| = 1/3

        # Iterate b from N down to 1
        # We prioritize large b because it allows for smaller error |a/b - 1.5|
        
        for b in range(N, 0, -1):
            if time_left() < 2.0:
                break
            
            # Calculate target a range
            # We want |a/b - 1.5| < best_val
            # |2a - 3b| < 2 * b * best_val
            
            if best_val == float('inf'):
                # Initial broad search
                # Check a close to 1.5 * b
                # Use integer arithmetic for center
                c1 = (3 * b) // 2
                c2 = c1 + 1
                
                # Check +/- 500 around center to find first pair
                found = False
                for offset in range(500):
                    for a in [c1 + offset, c1 - offset, c2 + offset, c2 - offset]:
                        if 1 <= a <= N and a != b:
                            if sigma[a] == sigma[b]:
                                num = abs(2 * a - 3 * b)
                                if num > 0:
                                    val = num / (2 * b)
                                    best_val = val
                                    best_pair = (a, b)
                                    found = True
                                    break
                    if found:
                        break
                
                # If no pair found in this broad search, we might need to widen or continue
                # But for N=10^7, we should find one quickly.
                # If we don't, we continue to next b.
                
            else:
                # Narrow search
                # max_diff = 2 * b * best_val
                # We check a such that |2a - 3b| < max_diff
                # Let k = 2a - 3b. We want |k| < max_diff.
                # k must have same parity as 3b (since 2a is even, 3b parity depends on b).
                # Actually, we just iterate a.
                
                # Range of a:
                # 2a - 3b < 2*b*best_val  =>  a < 1.5b + b*best_val
                # 2a - 3b > -2*b*best_val =>  a > 1.5b - b*best_val
                
                max_offset = int(b * best_val) + 1
                
                c1 = (3 * b) // 2
                c2 = c1 + 1
                
                # Check from center outwards
                # We stop at the first valid pair found for this b,
                # because it will be the best for this b.
                found_for_b = False
                for offset in range(max_offset + 1):
                    for a in [c1 + offset, c1 - offset, c2 + offset, c2 - offset]:
                        if 1 <= a <= N and a != b:
                            if sigma[a] == sigma[b]:
                                num = abs(2 * a - 3 * b)
                                if num > 0:
                                    val = num / (2 * b)
                                    if val < best_val:
                                        best_val = val
                                        best_pair = (a, b)
                                found_for_b = True
                                break
                    if found_for_b:
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
