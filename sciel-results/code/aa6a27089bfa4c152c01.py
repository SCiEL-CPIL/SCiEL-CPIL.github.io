"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_N = 10000000

    # Sieve for sigma(n) = sum of divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i

    if time.time() > deadline - 5.0:
        return [11, 6]

    best_num = 1
    best_den = 6
    best_a, best_b = 11, 6  # sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = 1/6

    # For small k, vectorize the search
    # 2a - 3b = k => a = (3b+k)//2, need (3b+k) even
    # 2a - 3b = -k => a = (3b-k)//2, need (3b-k) even
    
    for k in range(1, 50):
        if time.time() > deadline - 3.0:
            break
        
        # Case 1: 2a - 3b = k => a = (3b + k)//2
        # (3b + k) must be even => b and k same parity
        # 1 <= a <= MAX_N => 2 <= 3b + k <= 2*MAX_N => (2-k)//3 <= b <= (2*MAX_N - k)//3
        # 1 <= b <= MAX_N
        
        if k % 2 == 0:
            # b must be even
            b_min = max(2, (2 - k + 1) // 2)  # (3b+k)>=2 => b>=(2-k)/3, and b even
            b_max = min(MAX_N, (2 * MAX_N - k) // 3)
            if b_max % 2 != 0:
                b_max -= 1
            if b_min % 2 != 0:
                b_min += 1
            if b_min <= b_max:
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
                a_vals = (3 * b_vals + k) // 2
                mask = (a_vals >= 1) & (a_vals <= MAX_N) & (a_vals != b_vals)
                b_valid = b_vals[mask]
                a_valid = a_vals[mask]
                if len(b_valid) > 0:
                    sigma_b = sigma[b_valid]
                    sigma_a = sigma[a_valid]
                    match = sigma_b == sigma_a
                    if np.any(match):
                        # Find the one with largest b for smallest error
                        b_matches = b_valid[match]
                        a_matches = a_valid[match]
                        idx = np.argmax(b_matches)
                        b_found = int(b_matches[idx])
                        a_found = int(a_matches[idx])
                        # Error = k/(2*b_found)
                        if k * best_den < best_num * 2 * b_found:
                            best_num = k
                            best_den = 2 * b_found
                            best_a, best_b = a_found, b_found
                            if k == 1:
                                return [int(best_a), int(best_b)]
        
        if k % 2 == 1:
            # b must be odd
            b_min = max(1, (2 - k + 1) // 2)
            if b_min < 1:
                b_min = 1
            b_max = min(MAX_N, (2 * MAX_N - k) // 3)
            if b_max % 2 == 0:
                b_max -= 1
            if b_min % 2 == 0:
                b_min += 1
            if b_min <= b_max:
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
                a_vals = (3 * b_vals + k) // 2
                mask = (a_vals >= 1) & (a_vals <= MAX_N) & (a_vals != b_vals)
                b_valid = b_vals[mask]
                a_valid = a_vals[mask]
                if len(b_valid) > 0:
                    sigma_b = sigma[b_valid]
                    sigma_a = sigma[a_valid]
                    match = sigma_b == sigma_a
                    if np.any(match):
                        b_matches = b_valid[match]
                        a_matches = a_valid[match]
                        idx = np.argmax(b_matches)
                        b_found = int(b_matches[idx])
                        a_found = int(a_matches[idx])
                        if k * best_den < best_num * 2 * b_found:
                            best_num = k
                            best_den = 2 * b_found
                            best_a, best_b = a_found, b_found
                            if k == 1:
                                return [int(best_a), int(best_b)]
        
        # Case 2: 2a - 3b = -k => a = (3b - k)//2
        # (3b - k) must be even => b and k same parity
        # 1 <= a <= MAX_N => 2 <= 3b - k <= 2*MAX_N => (k+2)//3 <= b <= (2*MAX_N + k)//3
        # 1 <= b <= MAX_N
        
        if k % 2 == 0:
            b_min = max(2, (k + 2 + 1) // 2)
            b_max = min(MAX_N, (2 * MAX_N + k) // 3)
            if b_max % 2 != 0:
                b_max -= 1
            if b_min % 2 != 0:
                b_min += 1
            if b_min <= b_max:
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
                a_vals = (3 * b_vals - k) // 2
                mask = (a_vals >= 1) & (a_vals <= MAX_N) & (a_vals != b_vals)
                b_valid = b_vals[mask]
                a_valid = a_vals[mask]
                if len(b_valid) > 0:
                    sigma_b = sigma[b_valid]
                    sigma_a = sigma[a_valid]
                    match = sigma_b == sigma_a
                    if np.any(match):
                        b_matches = b_valid[match]
                        a_matches = a_valid[match]
                        idx = np.argmax(b_matches)
                        b_found = int(b_matches[idx])
                        a_found = int(a_matches[idx])
                        if k * best_den < best_num * 2 * b_found:
                            best_num = k
                            best_den = 2 * b_found
                            best_a, best_b = a_found, b_found
                            if k == 1:
                                return [int(best_a), int(best_b)]
        
        if k % 2 == 1:
            b_min = max(1, (k + 2 + 1) // 2)
            if b_min < 1:
                b_min = 1
            b_max = min(MAX_N, (2 * MAX_N + k) // 3)
            if b_max % 2 == 0:
                b_max -= 1
            if b_min % 2 == 0:
                b_min += 1
            if b_min <= b_max:
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
                a_vals = (3 * b_vals - k) // 2
                mask = (a_vals >= 1) & (a_vals <= MAX_N) & (a_vals != b_vals)
                b_valid = b_vals[mask]
                a_valid = a_vals[mask]
                if len(b_valid) > 0:
                    sigma_b = sigma[b_valid]
                    sigma_a = sigma[a_valid]
                    match = sigma_b == sigma_a
                    if np.any(match):
                        b_matches = b_valid[match]
                        a_matches = a_valid[match]
                        idx = np.argmax(b_matches)
                        b_found = int(b_matches[idx])
                        a_found = int(a_matches[idx])
                        if k * best_den < best_num * 2 * b_found:
                            best_num = k
                            best_den = 2 * b_found
                            best_a, best_b = a_found, b_found
                            if k == 1:
                                return [int(best_a), int(best_b)]
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
