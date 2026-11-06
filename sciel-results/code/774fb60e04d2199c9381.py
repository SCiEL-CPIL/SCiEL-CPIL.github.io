import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We want to maximize L to maximize the search space.
    # Memory constraint: FFT array of size 2*L+1 of float64.
    # 8 bytes * 2*L.
    # If L=10^8, size=2*10^8, mem=1.6GB.
    # Let's try L=10^8. If it fails, fallback.
    
    L = 10**8
    
    def generate_powerful(L):
        powerful = set()
        # y is square-free, x >= 1
        # val = x^2 * y^3 <= L
        # y^3 <= L => y <= L^(1/3)
        max_y = int(L ** (1/3)) + 1
        
        # Precompute square-free numbers up to max_y
        sq_free = []
        for y in range(1, max_y + 1):
            is_sq_free = True
            temp_y = y
            d = 2
            while d * d <= temp_y:
                if temp_y % (d * d) == 0:
                    is_sq_free = False
                    break
                d += 1
            if is_sq_free:
                sq_free.append(y)
                
        for y in sq_free:
            y_cubed = y ** 3
            if y_cubed > L:
                continue
            # x^2 <= L / y^3
            max_x = int(math.isqrt(L // y_cubed))
            for x in range(1, max_x + 1):
                val = x * x * y_cubed
                if val <= L:
                    powerful.add(val)
                    
        return sorted(powerful)

    try:
        P = generate_powerful(L)
    except MemoryError:
        # Fallback to smaller L if generation or FFT fails
        L = 5 * 10**7
        P = generate_powerful(L)
        
    K = len(P)
    
    # Check time
    if time.time() > deadline - 20:
        # If generation took too long, we might not have time for FFT
        # But generation is fast.
        pass

    # Prepare array for FFT
    # Size must be at least 2*L to avoid circular convolution artifacts for sums up to 2*L
    # Actually, max sum is 2*L.
    # We need indices 0 to 2*L.
    # FFT size M should be > 2*L.
    # Next power of 2 for efficiency.
    M = 1
    while M <= 2 * L:
        M *= 2
        
    # Check memory for M
    # M * 8 bytes
    if M * 8 > 2 * 10**9: # 2 GB limit
         # Try smaller L? Or just proceed and hope?
         # If it crashes, we score 0.
         # Let's be safer.
         if L > 5 * 10**7:
             L = 5 * 10**7
             P = generate_powerful(L)
             M = 1
             while M <= 2 * L:
                 M *= 2

    A = np.zeros(M, dtype=np.float64)
    for p in P:
        A[p] = 1.0
        
    # Convolution
    # C = A * A
    # R[k] = sum A[i] A[k-i]
    # We want R[N] for N in P+P
    
    fft_A = np.fft.fft(A)
    fft_A2 = fft_A * fft_A
    R = np.fft.ifft(fft_A2).real
    
    # R[k] is the count of pairs (a,b) such that a+b=k
    # Note: (a,b) and (b,a) are distinct in convolution if a!=b.
    # The task asks for distinct pairs a <= b.
    # If a != b, the convolution counts both (a,b) and (b,a).
    # If a == b, it counts (a,a) once.
    # We want the number of pairs {a,b} with a <= b.
    # So for a != b, count = R[k] / 2.
    # For a == b, count = R[k] (if k is even and k/2 is powerful).
    # Actually, R[k] = sum_{a+b=k} 1.
    # If a != b, (a,b) and (b,a) are two terms.
    # If a == b, (a,a) is one term.
    # So number of unordered pairs = (R[k] + count_of_diagonal) / 2 ?
    # No.
    # Let N_pairs be the number of unordered pairs {a,b} with a,b in P, a+b=k.
    # R[k] = sum_{a in P, b in P, a+b=k} 1.
    # If a != b, it contributes 2 (once for a, once for b in the sum? No, sum is over all a,b).
    # Yes, sum is over all ordered pairs.
    # So if a != b, the pair {a,b} contributes 2 to R[k].
    # If a == b, the pair {a,a} contributes 1 to R[k].
    # Let D be 1 if k/2 is in P and k is even, else 0.
    # Then R[k] = 2 * N_pairs_off_diag + 1 * D.
    # N_pairs = N_pairs_off_diag + D.
    # R[k] = 2 * (N_pairs - D) + D = 2 * N_pairs - D.
    # N_pairs = (R[k] + D) / 2.
    
    max_count = 0
    best_N = 0
    
    # We only care about N up to 2*L
    # R is an array of size M.
    # We can just find the max in R[2:2*L+1]
    # But we need to adjust for D.
    # D depends on k.
    
    # To be precise, we can iterate over candidates?
    # Or just take max R[k] and then verify/adjust.
    # The difference between R[k] and 2*N_pairs is at most 1.
    # So the N with max R[k] is very likely the one with max N_pairs.
    
    # Find top few candidates from R
    # R is float, so we round to int.
    R_int = np.rint(R).astype(np.int64)
    
    # Get indices of top 100 values
    # We can use np.argsort or just find max.
    # Let's find the max index in the valid range.
    # Valid range for N is 2 to 2*L.
    
    # To be safe, let's check the top 1000 sums.
    # But argsort on 10^8 elements is slow.
    # We can just find the global max.
    # If there are ties, it doesn't matter much.
    
    # However, if the max R is at k, and D=1, N_pairs = (R+1)/2.
    # If another k' has R' = R-1 and D'=0, N_pairs' = (R-1)/2.
    # So the one with max R is indeed the best or tied best.
    
    idx_max = np.argmax(R_int[2:2*L+1]) + 2
    best_N = int(idx_max)
    
    # Now retrieve pairs for best_N
    pairs = []
    powerful_set = set(P)
    
    for a in P:
        if a > best_N:
            break
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set:
            pairs.append([a, b])
            
    # The score is the number of pairs
    # We should return the count if we want to be sure, but the task asks for pairs.
    # The evaluator counts the pairs.
    
    return {'N': best_N, 'pairs': pairs}
