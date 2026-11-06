import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We know K=4 is impossible (W(4,3)=76).
    # So we start with K=5.
    for K in range(5, 10):
        # Try multiple random restarts
        max_restarts = 2000
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            colors = [-1] * 129  # 1-indexed, index 0 unused
            success = True
            
            for i in range(1, 129):
                valid = set(range(K))
                
                # Check all 4-APs ending at i
                # i, i-d, i-2d, i-3d
                max_d = (i - 1) // 3
                for d in range(1, max_d + 1):
                    i1 = i - 3 * d
                    i2 = i - 2 * d
                    i3 = i - d
                    
                    c1 = colors[i1]
                    c2 = colors[i2]
                    c3 = colors[i3]
                    
                    # The set of colors in the first 3 terms
                    S = {c1, c2, c3}
                    
                    # If the first 3 terms use <= 2 distinct colors,
                    # the 4th term must be a color NOT in S to ensure
                    # the total distinct colors is >= 3.
                    # If |S| == 1, S is {A}. c_i must not be A.
                    # If |S| == 2, S is {A, B}. c_i must not be A or B.
                    # If |S| == 3, S is {A, B, C}. c_i can be anything.
                    
                    if len(S) <= 2:
                        valid -= S
                        
                if not valid:
                    success = False
                    break
                    
                # Choose a color randomly from valid
                # Using random.choice is fine for now
                c = random.choice(list(valid))
                colors[i] = c
                
            if success:
                # Return the list of colors for positions 1 to 128
                return [colors[i] for i in range(1, 129)]
                
        # If K=5 didn't work in time, try K=6, etc.
        
    # Fallback: A simple pattern that is likely valid with many colors
    # Using x % 128 ensures all distinct, so valid.
    return [i % 128 for i in range(1, 129)]
