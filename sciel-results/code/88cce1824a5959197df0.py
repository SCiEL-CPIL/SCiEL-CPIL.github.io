import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    def count_total_3ap(A):
        count = 0
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    def has_4ap(A):
        A_set = set(A)
        for a in A_set:
            for d in range(1, (N - a) // 3 + 1):
                if a + d in A_set and a + 2 * d in A_set and a + 3 * d in A_set:
                    return True
        return False

    def is_valid_add(x, A):
        # Check if adding x creates a 4-AP
        # x is 1st
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    best_A = set()
    best_score = 0
    
    # Run multiple greedy iterations with randomization
    for _ in range(20):
        if time.time() > deadline - 10:
            break
            
        A = set()
        remaining = set(range(1, N + 1))
        
        # Shuffle remaining to break symmetry
        rem_list = list(remaining)
        random.shuffle(rem_list)
        
        # Greedy Phase
        while remaining and time.time() < deadline - 5:
            best_x = -1
            best_score_local = -1
            
            # We can optimize this by pre-filtering or just iterating
            for x in remaining:
                if is_valid_add(x, A):
                    score = count_3ap_gain(x, A)
                    if score > best_score_local:
                        best_score_local = score
                        best_x = x
            
            if best_x == -1:
                break
            A.add(best_x)
            remaining.remove(best_x)
            
        current_score = count_total_3ap(A)
        if current_score > best_score:
            best_score = current_score
            best_A = set(A)

    # Local Search on the best found so far
    if time.time() < deadline - 2:
        A = set(best_A)
        current_score = best_score
        
        # 1-swap local search
        # For each a in A, try to remove it and add some b not in A
        # We need to check if A - {a} + {b} is 4-AP free and has more 3-APs
        # This is expensive. Let's try a simpler 1-add/1-remove if possible, or just 1-remove if it doesn't break validity? No, we want to maximize.
        
        # A faster heuristic: try to remove an element that is "low value" and see if we can add a "high value" element elsewhere.
        # But checking 4-AP validity for every pair is O(N^3).
        
        # Let's stick to the greedy restarts which are safer and might find better global structures due to randomness.
        # If we have time, do a few targeted swaps.
        
        # Calculate contribution of each element to 3-APs
        # This is tricky because 3-APs involve 3 elements.
        
        # Let's try: for each a in A, remove it. Check if we can add any b from remaining that was previously blocked by a?
        # Actually, just restarting is often effective for small N like 256.
        
        # Let's do a few more restarts if time permits
        while time.time() < deadline - 2:
            A = set()
            remaining = set(range(1, N + 1))
            rem_list = list(remaining)
            random.shuffle(rem_list)
            
            while remaining and time.time() < deadline - 2:
                best_x = -1
                best_score_local = -1
                for x in remaining:
                    if is_valid_add(x, A):
                        score = count_3ap_gain(x, A)
                        if score > best_score_local:
                            best_score_local = score
                            best_x = x
                if best_x == -1:
                    break
                A.add(best_x)
                remaining.remove(best_x)
                
            current_score = count_total_3ap(A)
            if current_score > best_score:
                best_score = current_score
                best_A = set(A)

    return sorted(best_A)
