import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Search over geometric progressions U = {0} U {B^k | k=0..n-1}
    # B >= 2
    # n >= 1 (so at least {0, 1})
    
    for B in range(2, 20):
        U = [0]
        # Add powers B^0, B^1, ...
        # B^0 = 1
        # Check if 1 is already in U? No, U starts with 0.
        # If B=1, U={0,1}.
        # We loop B from 2.
        
        current_pow = 1
        for n in range(1, 25):
            if time.time() > deadline:
                break
                
            # Add B^(n-1) to U
            # U currently has 0 and B^0..B^(n-2)
            # We want to add B^(n-1)
            if n == 1:
                U.append(1) # B^0
            else:
                U.append(current_pow * B)
                current_pow *= B
            
            # Calculate score
            if len(U) < 2:
                continue
                
            max_u = max(U)
            # If max_u is too large, log is large, score drops.
            # But we need to check.
            
            # Compute sumset and diffset
            # For small n, this is fast.
            sumset = set()
            diffset = set()
            
            # Optimization: if n is large, sets grow quadratically.
            # n=20 -> 400 elements. 16000 ops. Fast.
            # n=50 -> 2500 elements. 625000 ops. Still fast in Python? Maybe.
            # But geometric growth makes max_u huge.
            
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            
            len_sum = len(sumset)
            len_diff = len(diffset)
            
            if len_sum == 0 or len_diff == 0:
                continue
                
            ratio = len_diff / len_sum
            if ratio <= 0:
                continue
            
            log_ratio = math.log(ratio)
            log_max = math.log(2 * max_u + 1)
            
            if log_max == 0:
                continue
                
            score = 1 + log_ratio / log_max
            
            if score > best_score:
                best_score = score
                best_set = U[:]
                
        # Reset U for next B
        # Note: The loop structure above appends to U. We need to reset U for each B.
        # My code above appends to U inside the n loop, but U is defined outside.
        # I need to fix the loop structure.
        
    # Let's rewrite the loop properly to avoid state issues.
    
    best_score = 0
    best_set = [0, 1, 3]
    
    for B in range(2, 30):
        U = [0]
        current_pow = 1
        for n in range(1, 30):
            if time.time() > deadline:
                break
            
            if n == 1:
                U.append(1)
            else:
                U.append(current_pow * B)
                current_pow *= B
            
            # Safety: if max U is too big, log_max grows, score likely drops.
            # But let's just compute.
            if len(U) > 50:
                break # Heuristic: geometric growth explodes max_u fast.
            
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            
            len_sum = len(sumset)
            len_diff = len(diffset)
            
            if len_sum == 0 or len_diff == 0:
                continue
                
            ratio = len_diff / len_sum
            if ratio <= 0:
                continue
            
            log_ratio = math.log(ratio)
            log_max = math.log(2 * max(U) + 1)
            
            if log_max == 0:
                continue
                
            score = 1 + log_ratio / log_max
            
            if score > best_score:
                best_score = score
                best_set = U[:]
                
    return best_set
