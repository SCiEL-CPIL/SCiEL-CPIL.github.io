    import numpy as np
    import time

    def construct():
        start_time = time.time()
        deadline = start_time + 55
        
        min_n = 1000
        max_n = 1000000
        max_val = max_n + 24
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        sf = np.ones(max_val + 1, dtype=np.int64)
        
        for p in primes:
            pk = p
            while pk <= max_val:
                indices = np.arange(pk, max_val + 1, pk)
                sf[indices] *= p
                pk *= p
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        
        # Extract the relevant slice
        # n in [1000, 1000000]
        # Window for n is sf[n+1 : n+25]
        # Start index in sf: 1001
        # End index in sf: 1000000 + 24 = 1000024
        sf_slice = sf[1001 : 1000025]
        
        # Get unique values and inverse indices
        # unique_vals will be the list of distinct factors
        # inverse_indices maps each position in sf_slice to an index in unique_vals
        unique_vals, inverse_indices = np.unique(sf_slice, return_inverse=True)
        
        num_windows = max_n - min_n + 1 # 999001
        distinct_counts = np.zeros(num_windows, dtype=np.int32)
        
        # Iterate over each unique value
        # K = len(unique_vals)
        for v_idx in range(len(unique_vals)):
            if time.time() > deadline - 2:
                break
            
            # Boolean mask for this value
            mask = (inverse_indices == v_idx)
            
            # Sliding window sum
            # We want to know if ANY element in the window is True
            # Using convolution with a window of ones
            # np.convolve requires 1D arrays
            # mask is bool, convert to int
            # 'valid' mode returns length len(mask) - 24 + 1
            # len(mask) is 999024
            # len(result) is 999001. Matches num_windows.
            
            # Optimization: if the mask has very few True values, we can just check those indices?
            # But np.convolve is C-speed.
            
            # Alternative: np.bincount on shifted indices?
            # Let's stick to convolve or cumsum.
            # cumsum is often faster than convolve for this.
            
            # cumsum approach:
            # pad mask with zeros?
            # cumsum = np.cumsum(mask)
            # window_sum = np.zeros(num_windows, dtype=np.int32)
            # For i in range(num_windows):
            #   start = i
            #   end = i + 24
            #   count = cumsum[end] - cumsum[start]
            # This is a Python loop. NO.
            
            # Vectorized cumsum difference:
            # We need cumsum of length len(mask) + 1 to handle boundaries easily?
            # Or just:
            # cumsum = np.cumsum(mask, dtype=np.int32)
            # # cumsum[i] is sum of mask[0..i-1]
            # # Sum of mask[i : i+24] is cumsum[i+24] - cumsum[i]
            # # We need cumsum up to index len(mask).
            # # So we need an extra 0 at the end?
            # # np.cumsum(mask) gives length L.
            # # We want indices 0..L-1.
            # # diff = cumsum[24:] - cumsum[:-24]
            # # This works if we handle the last part carefully.
            # # Actually, np.cumsum(mask) computes prefix sums.
            # # P[i] = sum(mask[0..i-1]) is not standard np.cumsum.
            # # np.cumsum(mask)[i] = sum(mask[0..i]).
            # # So sum(mask[i : i+24]) = sum(mask[0..i+23]) - sum(mask[0..i-1])
            # # = C[i+23] - (C[i-1] if i>0 else 0)
            
            # Let's use np.convolve, it's simpler and likely optimized.
            # However, np.convolve on large arrays might be slower than cumsum trick if not careful.
            # Let's try the cumsum trick.
            
            # To make it easy, pad mask with 23 zeros at the end?
            # No, just:
            # csum = np.cumsum(mask, dtype=np.int32)
            # # csum[i] = sum(mask[0..i])
            # # We want S[i] = sum(mask[i : i+24])
            # # S[i] = csum[i+23] - csum[i-1]
            # # For i=0, csum[-1] is 0.
            
            # We can create an array `csum_padded` of length L+1 where csum_padded[0]=0 and csum_padded[1:]=csum.
            # Then S[i] = csum_padded[i+24] - csum_padded[i].
            
            csum_padded = np.zeros(len(mask) + 1, dtype=np.int32)
            csum_padded[1:] = np.cumsum(mask, dtype=np.int32)
            
            # S[i] = csum_padded[i+24] - csum_padded[i]
            # i ranges from 0 to num_windows - 1
            # i+24 ranges from 24 to num_windows - 1 + 24 = len(mask) - 1 + 24?
            # len(mask) = 999024.
            # num_windows = 999001.
            # Max i = 999000.
            # Max i+24 = 999024.
            # csum_padded has length 999025. Index 999024 is valid.
            
            window_sums = csum_padded[24:] - csum_padded[:-24]
            
            # Check if any element is True
            distinct_counts += (window_sums > 0).astype(np.int32)
        
        min_count = np.min(distinct_counts)
        best_offset = np.argmin(distinct_counts)
        best_n = min_n + best_offset
        
        return best_n
    