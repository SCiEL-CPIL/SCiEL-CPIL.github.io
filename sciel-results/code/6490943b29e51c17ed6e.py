import numpy as np
import bisect
import time

def construct():
    N = 10**7
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    # 1. Compute sigma(n) for all n <= N
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Sieve of Eratosthenes style for sum of divisors
    # sigma[n] = sum of divisors of n
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # Check time
    if time.time() > deadline - 5:
        # Fallback to a simple valid pair if sieve is too slow (unlikely)
        return [11, 6] # sigma(11)=12, sigma(6)=12. 11/6 = 1.833, 3/2=1.5. Error=0.333
        
    # 2. Group numbers by their sigma value
    groups = {}
    # Using a list of lists might be faster if we pre-allocate, but dict of lists is fine
    # To speed up grouping, we can iterate through the numpy array
    # However, iterating 10^7 times in Python to build dict is ~1s
    
    # Convert sigma to list for faster access in the loop? 
    # sigma is a numpy array, accessing sigma[i] is fast but has overhead.
    # Let's stick to numpy array access.
    
    for n in range(1, N + 1):
        s_val = int(sigma[n])
        if s_val in groups:
            groups[s_val].append(n)
        else:
            groups[s_val] = [n]
            
    # 3. Sort each group
    for key in groups:
        if len(groups[key]) > 1:
            groups[key].sort()
            
    # 4. Find best pair
    best_error = float('inf')
    best_pair = (11, 6) # Default valid pair
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a fixed a, we want b close to 2a/3.
    
    for a in range(1, N + 1):
        if time.time() > deadline:
            break
            
        s_val = int(sigma[a])
        G = groups.get(s_val)
        if G is None or len(G) < 2:
            continue
            
        # Target b is 2a/3
        target = (2.0 * a) / 3.0
        idx = bisect.bisect_left(G, target)
        
        candidates = []
        if idx < len(G):
            candidates.append(G[idx])
        if idx > 0:
            candidates.append(G[idx - 1])
            
        for b in candidates:
            if b == a:
                continue
                
            # Calculate exact error |a/b - 3/2| = |2a - 3b| / (2b)
            # We want to minimize this.
            # Compare using cross multiplication to avoid float issues?
            # Error = |2a - 3b| / (2b)
            # Since we are comparing different pairs, we can just compute the float.
            # The values are small enough that float64 is precise enough for comparison?
            # a, b <= 10^7. 2a, 3b <= 3*10^7.
            # |2a - 3b| is an integer.
            # error_num = abs(2*a - 3*b)
            # error_den = 2*b
            # error = error_num / error_den
            
            # To find the global minimum, we can track the best rational.
            # But float comparison is likely sufficient given the magnitude.
            
            if error_num := abs(2 * a - 3 * b):
                 pass # Just checking syntax, will compute below
            
            err_num = abs(2 * a - 3 * b)
            err_den = 2 * b
            
            if err_num == 0:
                # Invalid if exactly 3/2. 
                # We should skip this specific pair, but maybe another b in G works?
                # Or just skip this b.
                continue
                
            # Compare err_num/err_den with best_error
            # best_error is a float.
            # It's better to compare cross products if we store best as num/den?
            # But best_error changes.
            # Let's just use float for simplicity, as precision loss is unlikely to flip the order of distinct rational errors of this magnitude unless they are extremely close.
            # However, "score is the binary64 serialization of that rational".
            # So we just need the pair that yields the smallest float value.
            
            current_err = err_num / err_den
            
            if current_err < best_error:
                best_error = current_err
                best_pair = (a, b)
                
    # Ensure we return a valid pair if best_pair wasn't updated (shouldn't happen)
    if best_pair == (11, 6) and best_error == float('inf'):
         # Fallback
         return [11, 6]
         
    return [best_pair[0], best_pair[1]]
