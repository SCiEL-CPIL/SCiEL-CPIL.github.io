import numpy as np
import random
import time

def construct():
    n = 128
    targets = list(range(128, 257))
    target_set = set(targets)
    
    # Initialize A with all elements
    A = set(range(0, 129))
    
    def compute_counts(A_set):
        counts = np.zeros(257, dtype=int)
        A_list = list(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if 128 <= s <= 256:
                    counts[s] += 1
        return counts

    counts = compute_counts(A)
    current_max = int(np.max(counts[128:257]))
    
    # Phase 1: Greedy Deletion
    improved = True
    while improved:
        improved = False
        best_x = None
        best_new_max = current_max
        best_sum_sq = int(np.sum(counts[128:257]**2))
        
        for x in list(A):
            # Check if removing x breaks coverage
            can_remove = True
            for a in A:
                s = x + a
                if 128 <= s <= 256:
                    if counts[s] == 1:
                        can_remove = False
                        break
            
            if not can_remove:
                continue
            
            # Incremental update
            old_counts = counts.copy()
            for a in A:
                s = x + a
                if 128 <= s <= 256:
                    counts[s] -= 1
            
            new_max = int(np.max(counts[128:257]))
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_x = x
                best_sum_sq = int(np.sum(counts[128:257]**2))
                improved = True
            elif new_max == best_new_max:
                new_sum_sq = int(np.sum(counts[128:257]**2))
                if new_sum_sq < best_sum_sq:
                    best_x = x
                    best_sum_sq = new_sum_sq
                    improved = True
            
            counts = old_counts
            
        if best_x is not None:
            A.remove(best_x)
            for a in A:
                s = best_x + a
                if 128 <= s <= 256:
                    counts[s] -= 1
            current_max = best_new_max
            improved = True

    # Phase 2: Randomized Swaps
    start_time = time.time()
    time_limit = 55
    
    A_list = list(A)
    
    while time.time() - start_time < time_limit:
        if not A:
            break
        x = random.choice(A_list)
        y = random.randint(0, 128)
        if y in A:
            continue
        
        # Check validity of removing x
        can_remove = True
        for a in A:
            s = x + a
            if 128 <= s <= 256:
                if counts[s] == 1:
                    can_remove = False
                    break
        
        if not can_remove:
            continue
        
        # Simulate Swap: Remove x, Add y
        # 1. Remove x
        for a in A:
            s = x + a
            if 128 <= s <= 256:
                counts[s] -= 1
        
        # 2. Add y
        for a in A:
            if a == x:
                continue
            s = y + a
            if 128 <= s <= 256:
                counts[s] += 1
        
        s_yy = y + y
        if 128 <= s_yy <= 256:
            counts[s_yy] += 1
            
        new_max = int(np.max(counts[128:257]))
        
        if new_max < current_max:
            A.remove(x)
            A.add(y)
            current_max = new_max
            A_list = list(A)
        else:
            # Revert
            for a in A:
                if a == x:
                    continue
                s = y + a
                if 128 <= s <= 256:
                    counts[s] -= 1
            
            s_yy = y + y
            if 128 <= s_yy <= 256:
                counts[s_yy] -= 1
                
            for a in A:
                s = x + a
                if 128 <= s <= 256:
                    counts[s] += 1
                    
    return list(A)
