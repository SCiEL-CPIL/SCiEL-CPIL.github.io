import time
import random
import math

def score_set(U):
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate U+U and U-U
    sum_set = set()
    diff_set = set()
    
    for u1 in U:
        for u2 in U:
            sum_set.add(u1 + u2)
            diff_set.add(u1 - u2)
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        return 1.0
        
    log_ratio = math.log2(ratio)
    log_denom = math.log2(2 * max_u + 1)
    
    return 1.0 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Initialize with some good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 4, 10, 28],
        [0, 1, 5, 15, 45],
        [0, 1, 2, 4, 8, 16],
    ]
    
    for c in candidates:
        if time.time() > deadline:
            break
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local search / Hill climbing
    # We will try to improve the best set found so far
    current_set = best_set[:]
    current_score = best_score
    
    # Random restarts if time permits
    while time.time() < deadline:
        # Pick a random set size
        k = random.randint(3, 15)
        # Generate a random set in [0, N]
        # N should be related to k
        N = k * k * 2
        # Ensure 0 is in the set
        current_set = [0] + random.sample(range(1, N + 1), k - 1)
        current_set.sort()
        
        s = score_set(current_set)
        if s > best_score:
            best_score = s
            best_set = current_set[:]
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing an element
            for i in range(1, len(current_set)):
                new_set = current_set[:i] + current_set[i+1:]
                if len(new_set) < 2: continue
                s_new = score_set(new_set)
                if s_new > s:
                    s = s_new
                    current_set = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = current_set[:]
                    break
            
            # Try adding an element
            if not improved:
                max_u = max(current_set)
                # Try adding numbers slightly larger or filling gaps
                candidates_to_add = []
                for x in range(1, max_u + 10):
                    if x not in current_set:
                        candidates_to_add.append(x)
                
                if candidates_to_add:
                    # Pick a few random candidates to test
                    random.shuffle(candidates_to_add)
                    for x in candidates_to_add[:10]:
                        new_set = current_set + [x]
                        new_set.sort()
                        s_new = score_set(new_set)
                        if s_new > s:
                            s = s_new
                            current_set = new_set
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_set = current_set[:]
                            break
            
            # Try swapping an element
            if not improved and len(current_set) > 2:
                i = random.randint(1, len(current_set) - 1)
                old_val = current_set[i]
                # Try replacing with a random value
                for _ in range(20):
                    new_val = random.randint(1, max(current_set) + 10)
                    if new_val in current_set:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_val
                    new_set.sort()
                    s_new = score_set(new_set)
                    if s_new > s:
                        s = s_new
                        current_set = new_set
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = current_set[:]
                        break
                        
    return best_set
