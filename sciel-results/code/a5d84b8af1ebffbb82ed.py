# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    target_size = 80
    
    # Precompute: for each x in 1..n, the set of triple sums it creates
    # when added to a set S. But that depends on S, so we need a different approach.
    
    # Key insight: A set S is valid iff all unordered triples (with repetition) 
    # have distinct sums. This is a B_3[1] sequence.
    
    # Greedy approach with bitset optimization:
    # Maintain a set of all triple sums seen so far.
    # When considering adding x, check if any triple involving x collides.
    
    # Triples involving x with existing elements a, b in S:
    # (x,x,x), (x,x,a), (x,a,a), (x,a,b) where a<=b in S
    
    # For efficiency, we can use a numpy boolean array for sums.
    # Max sum = 3*4096 = 12288
    
    max_sum = 3 * n
    sums_seen = np.zeros(max_sum + 1, dtype=bool)
    
    selected = []
    
    def is_valid_addition(x, S):
        """Check if adding x to S keeps all triple sums unique."""
        # Generate all triples involving x with elements from S
        # Triples: (x,x,x), (x,x,a), (x,a,a), (x,a,b) for a<=b in S
        # Also need to check against existing sums in sums_seen
        
        # (x,x,x)
        s = 3 * x
        if s <= max_sum and sums_seen[s]:
            return False
        new_sums = [s]
        
        # (x,x,a) for a in S
        for a in S:
            s = 2 * x + a
            if s <= max_sum and (sums_seen[s] or s in new_sums):
                return False
            new_sums.append(s)
        
        # (x,a,a) for a in S
        for a in S:
            s = x + 2 * a
            if s <= max_sum and (sums_seen[s] or s in new_sums):
                return False
            new_sums.append(s)
        
        # (x,a,b) for a<=b in S, a!=b (already covered a==b above)
        # Actually, we need all pairs (a,b) with a<=b from S
        # But (x,a,a) is already done. So now (x,a,b) with a<b
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                s = x + a + b
                if s <= max_sum and (sums_seen[s] or s in new_sums):
                    return False
                new_sums.append(s)
        
        return True
    
    def commit(x, S):
        """Mark all triple sums involving x as seen."""
        s = 3 * x
        sums_seen[s] = True
        for a in S:
            sums_seen[2 * x + a] = True
            sums_seen[x + 2 * a] = True
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                sums_seen[x + a + b] = True
    
    # Greedy: try to add elements in increasing order
    # But this may not maximize cardinality. Let's try multiple strategies.
    
    best_result = []
    
    # Strategy 1: Simple greedy from 1 to n
    selected1 = []
    sums_seen1 = np.zeros(max_sum + 1, dtype=bool)
    
    def check_and_add(x, S, sums_arr):
        """Check if x can be added, return True if yes and update."""
        new_sums = []
        # (x,x,x)
        s = 3 * x
        if s <= max_sum and sums_arr[s]:
            return False
        new_sums.append(s)
        
        # (x,x,a)
        for a in S:
            s = 2 * x + a
            if s <= max_sum and (sums_arr[s] or s in new_sums):
                return False
            new_sums.append(s)
        
        # (x,a,a)
        for a in S:
            s = x + 2 * a
            if s <= max_sum and (sums_arr[s] or s in new_sums):
                return False
            new_sums.append(s)
        
        # (x,a,b) a<b
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                b = S[j]
                s = x + a + b
                if s <= max_sum and (sums_arr[s] or s in new_sums):
                    return False
                new_sums.append(s)
        
        # Commit
        for s in new_sums:
            sums_arr[s] = True
        S.append(x)
        return True
    
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if len(selected1) >= target_size:
            break
        if check_and_add(x, selected1, sums_seen1):
            pass
        # If we've found enough, break
        if len(selected1) >= target_size:
            break
    
    if len(selected1) > len(best_result):
        best_result = list(selected1)
    
    if time.time() > deadline - 5:
        return best_result[:target_size]
    
    # Strategy 2: Greedy with random restarts, trying different orderings
    best_result = list(best_result)
    
    for restart in range(50):
        if time.time() > deadline - 2:
            break
        
        # Random permutation of candidates, biased towards smaller numbers
        # but with some randomness
        candidates = list(range(1, n + 1))
        # Weight towards smaller values but allow some larger ones
        weights = [1.0 / (c + 1) ** 0.5 for c in candidates]
        # Shuffle with bias
        random.shuffle(candidates)
        
        selected2 = []
        sums_seen2 = np.zeros(max_sum + 1, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if len(selected2) >= target_size:
                break
            check_and_add(x, selected2, sums_seen2)
        
        if len(selected2) > len(best_result):
            best_result = list(selected2)
            if len(best_result) >= target_size:
                break
    
    if time.time() > deadline - 2:
        return best_result[:target_size]
    
    # Strategy 3: Local search / hill climbing
    # Try to swap elements to improve
    if len(best_result) < target_size:
        current = list(best_result)
        current.sort()
        
        for iteration in range(200):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try removing an element and adding a different one
            for i in range(len(current)):
                if time.time() > deadline - 2:
                    break
                
                removed = current[i]
                # Rebuild sums without removed
                sums_tmp = np.zeros(max_sum + 1, dtype=bool)
                S_tmp = [x for x in current if x != removed]
                
                # Rebuild sums for S_tmp
                for a_idx, a in enumerate(S_tmp):
                    sums_tmp[3 * a] = True
                    for b_idx in range(a_idx + 1, len(S_tmp)):
                        b = S_tmp[b_idx]
                        sums_tmp[2 * a + b] = True
                        sums_tmp[a + 2 * b] = True
                        for c_idx in range(b_idx + 1, len(S_tmp)):
                            c = S_tmp[c_idx]
                            sums_tmp[a + b + c] = True
                
                # Try adding a new element
                for new_x in range(1, n + 1):
                    if new_x == removed:
                        continue
                    if new_x in S_tmp:
                        continue
                    
                    # Check validity
                    valid = True
                    new_sums_list = []
                    
                    s = 3 * new_x
                    if s <= max_sum and sums_tmp[s]:
                        valid = False
                    else:
                        new_sums_list.append(s)
                    
                    if valid:
                        for a in S_tmp:
                            s = 2 * new_x + a
                            if s <= max_sum and (sums_tmp[s] or s in new_sums_list):
                                valid = False
                                break
                            new_sums_list.append(s)
                    
                    if valid:
                        for a in S_tmp:
                            s = new_x + 2 * a
                            if s <= max_sum and (sums_tmp[s] or s in new_sums_list):
                                valid = False
                                break
                            new_sums_list.append(s)
                    
                    if valid:
                        for i2 in range(len(S_tmp)):
                            a = S_tmp[i2]
                            for j2 in range(i2 + 1, len(S_tmp)):
                                b = S_tmp[j2]
                                s = new_x + a + b
                                if s <= max_sum and (sums_tmp[s] or s in new_sums_list):
                                    valid = False
                                    break
                                new_sums_list.append(s)
                            if not valid:
                                break
                    
                    if valid:
                        # Accept the swap
                        S_tmp.append(new_x)
                        S_tmp.sort()
                        current = S_tmp
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_result[:target_size]

# EVOLVE-BLOCK-END
