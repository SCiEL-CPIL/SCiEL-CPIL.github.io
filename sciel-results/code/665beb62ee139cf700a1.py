def calculate_score_fast(U):
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=int)
    max_u = U_arr.max()
    if max_u == 0:
        return 1.0
        
    N = 2 * max_u + 1
    N_fft = 1
    while N_fft < N:
        N_fft *= 2
        
    A = np.zeros(N_fft)
    A[U_arr] = 1.0
    
    # B for difference
    B = np.zeros(N_fft)
    # B[i] = A[max_u - i]
    # Indices i from 0 to max_u
    B[:max_u+1] = A[max_u::-1]
    
    F_A = np.fft.fft(A)
    F_B = np.fft.fft(B)
    
    S = np.fft.ifft(F_A * F_A).real
    D = np.fft.ifft(F_A * F_B).real
    
    eps = 1e-9
    count_sum = np.count_nonzero(S > eps)
    count_diff = np.count_nonzero(D > eps)
    
    if count_sum == 0 or count_diff == 0:
        return 0.0
        
    ratio = count_diff / count_sum
    if ratio <= 0:
        return 0.0
        
    score = 1.0 + np.log(ratio) / np.log(2 * max_u + 1)
    return score
