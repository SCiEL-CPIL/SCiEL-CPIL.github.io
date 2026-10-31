import numpy as np

N = 100

def construct():
    # Generate Thue-Morse sequence
    tm = np.zeros(128, dtype=float)
    for i in range(1, 128):
        tm[i] = tm[i >> 1] ^ (i & 1)
    h = tm[:N].copy()
    
    # Balance to 50 ones
    num_ones = np.sum(h)
    if num_ones > 50:
        ones_idx = np.where(h == 1.0)[0]
        # Flip the last ones
        to_flip = ones_idx[int(num_ones - 50):]
        h[to_flip] = 0.0
    elif num_ones < 50:
        zeros_idx = np.where(h == 0.0)[0]
        to_flip = zeros_idx[:int(50 - num_ones)]
        h[to_flip] = 1.0
        
    def get_min_ac(h_vec):
        # Circular autocorrelation
        # h_vec is 0/1
        H = np.fft.fft(h_vec)
        ac = np.real(np.fft.ifft(H * np.conj(H)))
        # ac[0] is sum(h^2) = 50
        # ac[k] is sum(h[i] h[i+k])
        # We care about k=1..N-1
        return np.min(ac[1:])

    current_min = get_min_ac(h)
    
    # Local Search
    # We will try all swaps if it's fast enough with FFT.
    # One FFT is ~ N log N = 700 ops.
    # 2500 swaps * 700 = 1.75M ops. Very fast.
    
    improved = True
    while improved:
        improved = False
        ones_idx = np.where(h == 1.0)[0]
        zeros_idx = np.where(h == 0.0)[0]
        
        best_gain = 0
        best_swap = None
        best_new_min = current_min
        
        for i in ones_idx:
            for j in zeros_idx:
                # Swap
                h[i] = 0.0
                h[j] = 1.0
                
                new_min = get_min_ac(h)
                
                gain = new_min - current_min
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (i, j)
                    best_new_min = new_min
                
                # Revert
                h[i] = 1.0
                h[j] = 0.0
                
        if best_swap:
            i, j = best_swap
            h[i] = 0.0
            h[j] = 1.0
            current_min = best_new_min
            improved = True
        else:
            break
            
    return h
