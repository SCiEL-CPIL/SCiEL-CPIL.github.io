        # Update F using geometric series sum
        # Sum_{j=0}^3 exp(-i 2 pi m (i_0 + j) / N_fft)
        # = exp(-i 2 pi m i_0 / N_fft) * (1 - exp(-i 2 pi m 4 / N_fft)) / (1 - exp(-i 2 pi m / N_fft))
        
        # Handle m=0 case separately to avoid div by 0
        # At m=0, sum = 4
        
        m = freqs
        i_0 = start_idx
        
        # Vectorized geometric sum
        # numerator = 1 - exp(-i 2 pi m 4 / fft_size)
        # denominator = 1 - exp(-i 2 pi m / fft_size)
        
        # To be safe and fast, just use the loop-free outer product version 
        # unless performance is critical. 3000k steps is enough.
        # But let's try the geometric series for elegance and speed.
        
        # Actually, np.exp is the cost. 
        # Let's stick to the outer product for simplicity and correctness, 
        # 3M steps is plenty for SA.
