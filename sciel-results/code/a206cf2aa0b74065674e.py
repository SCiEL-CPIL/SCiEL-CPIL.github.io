import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    while time.time() < deadline:
        S = []
        S_np = np.array([], dtype=np.int32)
        counts = np.zeros(2001, dtype=np.int8)
        available = list(range(1, 1001))
        
        while available:
            best_x = -1
            best_score = float('inf')
            candidates = []
            
            for x in available:
                # Calculate sums
                if len(S_np) > 0:
                    sums = x + S_np
                    sums = np.append(sums, 2 * x)
                else:
                    sums = np.array([2 * x], dtype=np.int32)
                
                c = counts[sums]
                
                # Check validity
                if np.any(c == 2):
                    continue
                
                # Score
                score = np.sum(c == 1)
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    candidates = [x]
                elif score == best_score:
                    candidates.append(x)
            
            if best_x == -1:
                break
            
            # Random tie-breaking
            if len(candidates) > 1:
                best_x = candidates[random.randint(0, len(candidates) - 1)]
            
            # Update state
            S.append(best_x)
            S_np = np.append(S_np, best_x)
            if len(S_np) > 1:
                sums = best_x + S_np[:-1]
                sums = np.append(sums, 2 * best_x)
            else:
                sums = np.array([2 * best_x], dtype=np.int32)
            counts[sums] += 1
            
            # Remove from available
            # available is a list, removal is O(N). 
            # For 1000 items, this is fine.
            available.remove(best_x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
