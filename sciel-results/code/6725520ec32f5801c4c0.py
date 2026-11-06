import math
import time
import numpy as np

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to 'limit'.
    A powerful number is of the form x^2 * y^3 where y is square-free.
    """
    powerful = set()
    # Max y such that y^3 <= limit
    max_y = int(math.ceil(limit ** (1.0/3.0))) + 2
    
    # Sieve for square-free numbers up to max_y
    # sq_free[i] is True if i is square-free
    sq_free = [True] * (max_y + 1)
    sq_free[0] = False
    p = 2
    while p * p <= max_y:
        if sq_free[p]:
            sq = p * p
            for j in range(sq, max_y + 1, sq):
                sq_free[j] = False
        p += 1
    
    for y in range(1, max_y + 1):
        if not sq_free[y]:
            continue
        y3 = y * y * y
        if y3 > limit:
            break
        # x^2 * y3 <= limit => x^2 <= limit / y3
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful.add(n)
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_n = 0
    best_pairs = []
    
    # We will try increasing limits
    # L=10^6 is very fast, L=10^7 is standard, L=10^8 is aggressive
    limits = [10**6, 10**7, 2*10**7]
    
    for L in limits:
        if time.time() > deadline - 30:
            break
            
        try:
            P = generate_powerful_numbers(L)
            if not P:
                continue
                
            P_arr = np.array(P, dtype=np.int64)
            P_set = set(P)
            
            # FFT Convolution
            # Target size for sum: 2*L
            target_size = 2 * L + 1
            n_fft = 1
            while n_fft < target_size:
                n_fft *= 2
                
            # Memory check: n_fft * 8 bytes (float64) * 2 (real/imag) * 2 (forward/back)
            # rfft is more efficient.
            if n_fft > 2**27: # ~134M points
                continue
                
            A = np.zeros(n_fft, dtype=np.float64)
            # Mark powerful numbers
            # P_arr values are indices
            A[P_arr[P_arr < n_fft]] = 1.0
            
            F = np.fft.rfft(A)
            F2 = F * F
            result = np.fft.irfft(F2, n=n_fft)
            
            # The result array contains counts of pairs (a,b) such that a+b = index
            # Note: FFT convolution counts ordered pairs (a,b) and (b,a) if a!=b, and (a,a) once?
            # Actually, convolution C[k] = sum A[i] A[k-i].
            # If A[i]=1, A[k-i]=1, it contributes 1.
            # For a != b, (a,b) and (b,a) are distinct terms in the sum, so count is 2.
            # For a == b, (a,a) is one term, count is 1.
            # We want distinct pairs a <= b.
            # If count is C, then:
            # If N is even, a=N/2 is a powerful number, it contributes 1 to C.
            # Other pairs a!=b contribute 2 to C.
            # So number of pairs = (C - d) / 2 + d = (C + d) / 2, where d is 1 if N/2 is powerful else 0.
            
            # Extract scores
            # We only care about N <= 2*L
            scores = result[:2*L+1]
            
            # Calculate d (1 if N/2 is powerful and N is even)
            # d[N] = 1 if N is even and N/2 in P
            d = np.zeros(2*L+1, dtype=np.int32)
            even_indices = np.arange(2, 2*L+1, 2)
            half_indices = even_indices // 2
            # Check which half_indices are in P
            # P_bool array
            P_bool = np.zeros(L+1, dtype=np.int8)
            P_bool[P_arr[P_arr <= L]] = 1
            d[even_indices] = P_bool[half_indices]
            
            # Total pairs count = (scores + d) / 2
            # We want to maximize this.
            # Since scores and d are integers (mostly), we can just look at scores + d.
            # Note: scores from FFT might have small floating point errors.
            # We should round them.
            scores_int = np.rint(scores).astype(np.int64)
            
            total_pairs = (scores_int + d) // 2
            
            # Find top candidates
            # We want the max total_pairs
            # Get top 100 indices
            if len(total_pairs) > 100:
                top_indices = np.argpartition(total_pairs, -100)[-100:]
            else:
                top_indices = np.arange(len(total_pairs))
                
            top_indices = top_indices[np.argsort(total_pairs[top_indices])[::-1]]
            
            current_best_n = 0
            current_best_pairs = []
            
            for idx in top_indices:
                N = int(idx)
                if N < 2:
                    continue
                
                # Verify and extract pairs
                pairs = []
                # Iterate a in P, a <= N/2
                # P is sorted
                # Use binary search or simple loop
                # Since P is sorted, we can break early
                for a in P:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in P_set:
                        pairs.append([a, b])
                        
                if len(pairs) > len(current_best_pairs):
                    current_best_pairs = pairs
                    current_best_n = N
                    
            if len(current_best_pairs) > len(best_pairs):
                best_pairs = current_best_pairs
                best_n = current_best_n
                
        except Exception as e:
            continue
            
    # Fallback
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
