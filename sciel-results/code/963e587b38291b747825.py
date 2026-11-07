import time
import math
import bisect
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 80.0
    
    N = 10_000_000
    
    # Sieve for smallest prime factor
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    if time.time() - start_time > deadline - 30:
        return [int(11), int(6)]
    
    # Compute sigma(n) for all n using multiplicative property with spf
    sigma = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    # Track the p-power part: for n = p^k * r where p ∤ r, 
    # sigma(n) = sigma(r) * (p^(k+1)-1)/(p-1)
    # We track: spow[n] = p^k (the full p-part of n)
    # and ssigma[n] = (p^(k+1)-1)/(p-1) (the sigma of the p-part)
    spow = np.zeros(N + 1, dtype=np.int64)
    ssigma = np.zeros(N + 1, dtype=np.int64)
    spow[1] = 1
    ssigma[1] = 1
    
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 20:
            break
        p = spf[n]
        m = n // p
        if spf[m] != p:
            # p does not divide m, so n = p * m where p ∤ m
            # sigma(n) = sigma(m) * (1 + p)
            # spow(n) = p, ssigma(n) = 1 + p
            sigma[n] = sigma[m] * (1 + p)
            spow[n] = p
            ssigma[n] = 1 + p
        else:
            # p divides m, so m = p^k * r with k >= 1
            # n = p^(k+1) * r
            # sigma(n) = sigma(m) * (1 + p + ... + p^(k+1)) / (1 + ... + p^k)
            # = sigma(m) * (p^(k+2) - 1) / (p^(k+1) - 1)
            # But we can compute: sigma(n) = sigma(r) * (p^(k+2)-1)/(p-1)
            # And sigma(m) = sigma(r) * (p^(k+1)-1)/(p-1)
            # So sigma(n) = sigma(m) * (p^(k+2)-1) / (p^(k+1)-1)
            # Alternatively: sigma(n) = sigma(m) - sigma(m // p^k) * p^k + sigma(m // p^k) * p^(k+1)
            # Simpler: sigma(n) = sigma(m) + sigma(m // p^k) * p^(k+1) - sigma(m // p^k) * p^k... 
            # Let's use: if m = p^k * r, then sigma(m) = sigma(r) * S_k where S_k = (p^(k+1)-1)/(p-1)
            # sigma(n) = sigma(r) * S_{k+1} = sigma(m) * S_{k+1}/S_k
            # S_{k+1} = S_k + p^(k+1)
            # So sigma(n) = sigma(m) + sigma(m) * p^(k+1) / S_k
            # But sigma(m) / S_k = sigma(r), so sigma(n) = sigma(m) + sigma(r) * p^(k+1)
            # And sigma(r) = sigma(m) / S_k
            # We have spow[m] = p^k and ssigma[m] = S_k
            # sigma(r) = sigma[m] / ssigma[m]
            # sigma(n) = sigma[m] + (sigma[m] // ssigma[m]) * (spow[m] * p)
            # But division may not be exact in integer arithmetic... let's check
            # Actually sigma[m] = sigma[r] * ssigma[m], so sigma[m] / ssigma[m] = sigma[r] exactly
            r_sigma = sigma[m] // ssigma[m]
            sigma[n] = sigma[m] + r_sigma * (spow[m] * p)
            spow[n] = spow[m] * p
            ssigma[n] = ssigma[m] + spow[m] * p
    
    if time.time() - start_time > deadline - 10:
        return [int(11), int(6)]
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() - start_time > deadline - 5:
        return [int(11), int(6)]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Sort each group for binary search
    for s_val in groups:
        groups[s_val].sort()
    
    for s_val, nums in groups.items():
        if time.time() - start_time > deadline:
            break
        if len(nums) < 2:
            continue
        # For each b, find a closest to 1.5*b
        for i in range(len(nums)):
            b = nums[i]
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx - 1])
            for a in candidates:
                if a == b:
                    continue
                # Check exact: 2*a == 3*b means error is 0, which is invalid
                if 2 * a == 3 * b:
                    continue
                # err = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = num / den
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
    
    if best_error == float('inf'):
        return [int(11), int(6)]
    return [int(best_a), int(best_b)]
