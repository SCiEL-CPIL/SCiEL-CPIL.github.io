import numpy as np
import time
from math import gcd

def construct():
            start_time = time.time()
            deadline = start_time + 80  # 80s limit, 10s margin
            
            # 1. Compute LCM of 1..30
            L = 1
            for i in range(1, 31):
                L = L * i // gcd(L, i)
            
            # Weights
            w = [L // i for i in range(1, 31)]
            
            # Split
            w1 = w[:15]
            w2 = w[15:]
            
            # Generate sums for first half
            sums1 = np.array([0], dtype=np.int64)
            for weight in w1:
                # sums1 contains all current sums
                # New sums: s - w, s, s + w
                # We need to be careful with memory. 3^15 is ~14M. int64 is 8 bytes. 14M * 8 = 112 MB.
                # Concatenating 3 arrays of size N creates 3N.
                # This is fine.
                sums1 = np.concatenate([sums1 - weight, sums1, sums1 + weight])
            
            # Generate sums for second half
            sums2 = np.array([0], dtype=np.int64)
            for weight in w2:
                sums2 = np.concatenate([sums2 - weight, sums2, sums2 + weight])
            
            # Sort sums2
            # We need to keep track of indices to recover coefficients later
            # But for finding the *value*, we just need the sorted values.
            # To recover coefficients, we need the index in the unsorted array.
            idx2 = np.argsort(sums2)
            sorted_sums2 = sums2[idx2]
            
            # For each sum in sums1, find best match in sorted_sums2
            # Target: -sums1
            targets = -sums1
            # searchsorted returns the index where the target would be inserted
            pos = np.searchsorted(sorted_sums2, targets)
            
            # We need to check pos and pos-1 (and maybe pos+1 for safety? No, pos is insertion point, so pos-1 is <= target, pos is >= target)
            # Actually, if target exists, searchsorted returns the first occurrence.
            # The closest values are at pos-1 and pos.
            
            best_diff = np.iinfo(np.int64).max
            best_s1_idx = 0
            best_s2_idx = 0
            
            # Vectorized check is hard because we need to update global best based on min.
            # Looping over 14M items in Python is slow (~1-2s? maybe 10s?).
            # 14M iterations in Python is risky for 60s limit if not optimized.
            # Let's try to vectorize the difference calculation.
            
            # Get candidates at pos-1 and pos
            # Handle boundaries
            pos_minus = np.maximum(0, pos - 1)
            pos_plus = np.minimum(len(sorted_sums2) - 1, pos)
            
            # Candidate 1: sums2 at pos-1
            c1 = sorted_sums2[pos_minus]
            diff1 = sums1 + c1
            
            # Candidate 2: sums2 at pos
            c2 = sorted_sums2[pos_plus]
            diff2 = sums1 + c2
            
            # We want min |diff| != 0
            abs_diff1 = np.abs(diff1)
            abs_diff2 = np.abs(diff2)
            
            # Mask out zeros
            mask1 = abs_diff1 > 0
            mask2 = abs_diff2 > 0
            
            # If diff is 0, it's invalid.
            # We can set abs_diff to infinity where mask is false.
            abs_diff1[~mask1] = np.iinfo(np.int64).max
            abs_diff2[~mask2] = np.iinfo(np.int64).max
            
            # Find min
            min_diff1 = np.min(abs_diff1)
            min_diff2 = np.min(abs_diff2)
            
            if min_diff1 < min_diff2:
                best_diff = min_diff1
                # Find index
                best_s1_idx = np.argmin(abs_diff1)
                best_s2_val_idx = pos_minus[best_s1_idx]
                best_s2_idx = idx2[best_s2_val_idx]
            else:
                best_diff = min_diff2
                best_s1_idx = np.argmin(abs_diff2)
                best_s2_val_idx = pos_plus[best_s1_idx]
                best_s2_idx = idx2[best_s2_val_idx]
            
            # Decode coefficients
            # best_s1_idx is index in the concatenated array.
            # The generation order was:
            # Start: [0]
            # Step 1: [-w1, 0, w1] -> indices 0,1,2 correspond to coeff -1, 0, 1
            # Step 2: 
            #   From -w1: -w1-w2, -w1, -w1+w2
            #   From 0: -w2, 0, w2
            #   From w1: w1-w2, w1, w1+w2
            # This corresponds to base-3 representation where:
            # Digit 0 -> -1
            # Digit 1 -> 0
            # Digit 2 -> 1
            # And the most significant digit is the last weight added?
            # Let's trace:
            # w1 added first.
            # Index 0: coeff -1 for w1.
            # Index 1: coeff 0 for w1.
            # Index 2: coeff 1 for w1.
            # Then w2 added.
            # The array is concatenated: [S-1, S, S+1] where S is previous array.
            # So index `i` in the new array:
            # If i < N: it came from S-1. Coeff for w2 is -1. Index in S is i.
            # If N <= i < 2N: it came from S. Coeff for w2 is 0. Index in S is i-N.
            # If 2N <= i < 3N: it came from S+1. Coeff for w2 is 1. Index in S is i-2N.
            # This is base-3, but the mapping is:
            # Remainder 0 -> -1
            # Remainder 1 -> 0
            # Remainder 2 -> 1
            # And we divide by 3 and continue.
            # The first weight added (w1) corresponds to the *least* significant digit in this recursive construction?
            # Let's check:
            # After 1 step (w1):
            # idx 0: rem 0 -> -1. Coeffs: [-1]
            # idx 1: rem 1 -> 0. Coeffs: [0]
            # idx 2: rem 2 -> 1. Coeffs: [1]
            # After 2 steps (w2):
            # idx 0: came from S-1 (idx 0 of S). Coeff w2 is -1. Coeff w1 is -1. Vector: [-1, -1] (w1, w2)
            # idx 3: came from S (idx 0 of S). Coeff w2 is 0. Coeff w1 is -1. Vector: [-1, 0]
            # idx 6: came from S+1 (idx 0 of S). Coeff w2 is 1. Coeff w1 is -1. Vector: [-1, 1]
            # So the decoding gives coefficients in the order they were added.
            # w1 is the first added, so it's the "last" to be decoded?
            # No, the recursion goes:
            # `sums_new = [sums_old - w, sums_old, sums_old + w]`
            # The index in `sums_new` determines the coefficient for the *current* weight `w`.
            # The quotient `index // 3` determines the index in `sums_old`, which corresponds to the *previous* weights.
            # So if we decode `idx` by taking `idx % 3` first, we get the coefficient for the *last* weight added (w15).
            # Then `idx // 3 % 3` gives coefficient for w14.
            # ...
            # `idx // 3^14 % 3` gives coefficient for w1.
            # So we need to reverse the coefficients to get them in order 1..15.
            
            def decode_idx(idx, n_weights):
                coeffs = []
                for _ in range(n_weights):
                    r = idx % 3
                    # Mapping: 0->-1, 1->0, 2->1
                    if r == 0:
                        coeffs.append(-1)
                    elif r == 1:
                        coeffs.append(0)
                    else:
                        coeffs.append(1)
                    idx //= 3
                coeffs.reverse()
                return coeffs
            
            c1_coeffs = decode_idx(best_s1_idx, 15)
            c2_coeffs = decode_idx(best_s2_idx, 15)
            
            result = c1_coeffs + c2_coeffs
            
            # Verify
            s = sum(result[i-1] / i for i in range(1, 31))
            if s == 0:
                # This shouldn't happen if we skipped zeros, but just in case
                # Fallback to seed
                return [0] * 28 + [1, -1]
            
            return result
