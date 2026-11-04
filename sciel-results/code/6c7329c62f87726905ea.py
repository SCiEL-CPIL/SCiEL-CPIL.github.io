import numpy as np

N = 400

def construct():
    # Target support size. 
    # Larger M reduces the 1/M factor, but increases collision complexity.
    # M=80 is a reasonable balance.
    M = 80
    
    best_peak = float('inf')
    best_x = None
    
    # Number of random restarts for the support selection
    num_restarts = 20
    
    for _ in range(num_restarts):
        # 1. Initialize subset using a random permutation of indices
        indices = np.random.permutation(N)
        support = indices[:M]
        
        # 2. Initialize signs randomly
        signs = np.random.choice([1, -1], size=M)
        
        # 3. Optimize signs to minimize autoconvolution peak (excluding lag 0)
        # We use a simple local search / greedy flip
        # To speed up, we can precompute the difference map or just compute conv each time.
        # With M=80, N=400, conv is cheap.
        
        current_peak = float('inf')
        x_temp = np.zeros(N)
        x_temp[support] = signs / np.sqrt(M)
        
        # Initial peak calculation
        conv = np.convolve(x_temp, x_temp, mode='full')
        # conv length is 2N-1 = 799. Center is at index N-1 = 399.
        # We exclude the center (lag 0).
        # The lags are from -(N-1) to N-1.
        # Indices 0 to 398 correspond to lags 398 down to 0? 
        # np.convolve full: index k corresponds to lag k - (N-1).
        # Lag 0 is at index N-1 = 399.
        # We want max abs of conv excluding index 399.
        
        def get_peak(conv_arr):
            # Exclude center
            if len(conv_arr) > 1:
                # Create a copy with center zeroed out
                c = conv_arr.copy()
                c[N-1] = 0
                return np.max(np.abs(c))
            return 0.0

        current_peak = get_peak(conv)
        
        # Local search on signs
        improved = True
        max_iters = 50 # Limit iterations per restart
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try flipping each sign
            for i in range(M):
                # Flip sign i
                old_sign = signs[i]
                signs[i] = -old_sign
                
                # Update x_temp
                x_temp[support[i]] = signs[i] / np.sqrt(M)
                
                # Recompute conv (efficient enough for M=80, N=400)
                conv_new = np.convolve(x_temp, x_temp, mode='full')
                new_peak = get_peak(conv_new)
                
                if new_peak < current_peak - 1e-9:
                    current_peak = new_peak
                    conv = conv_new
                    improved = True
                else:
                    # Revert
                    signs[i] = old_sign
                    x_temp[support[i]] = old_sign / np.sqrt(M)
                    
            # If we didn't improve in a full pass, we could try a random flip or stop
            # To escape local minima, let's try a few random flips if stuck?
            # For now, simple greedy is fine.
            
        if current_peak < best_peak:
            best_peak = current_peak
            best_x = x_temp.copy()
            best_support = support.copy()
            best_signs = signs.copy()
            
    if best_x is None:
        # Fallback
        best_x = np.zeros(N)
        best_x[0] = 1.0
        
    return best_x
