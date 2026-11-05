import random
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    # We can run as many trials as possible
    # Each trial is O(N * K)
    # N=1000, K~80. 80k ops per trial.
    # 1000 trials = 80M ops. Python ~ 10M-50M ops/s.
    # So 1000-8000 trials should fit in 115s.
    
    for _ in range(10000):
        if time.time() - start_time > timeout:
            break
            
        current_set = []
        counts = {}
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        # Local variables for speed
        c_set = current_set
        c_counts = counts
        
        for x in candidates:
            valid = True
            # Check sums x + y
            for y in c_set:
                s = x + y
                if c_counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                s = 2 * x
                if c_counts.get(s, 0) >= 2:
                    valid = False
            
            if valid:
                c_set.append(x)
                for y in c_set[:-1]: # y is the new x? No, y iterates old set
                    # Wait, I need to update counts for x+y for all y in OLD set
                    # and 2*x
                    pass
                
                # Update counts properly
                # The loop above checked validity. Now update.
                # We need to iterate c_set again?
                # To save time, we can store the sums that were checked?
                # Or just iterate again. It's small.
                
                for y in c_set[:-1]:
                    s = x + y
                    c_counts[s] = c_counts.get(s, 0) + 1
                s = 2 * x
                c_counts[s] = c_counts.get(s, 0) + 1
                
                if len(c_set) > best_size:
                    best_size = len(c_set)
                    best_set = c_set.copy()

    return best_set
