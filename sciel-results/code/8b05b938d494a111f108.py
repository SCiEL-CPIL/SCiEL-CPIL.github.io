import random
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    
    def get_sums_set(S):
        sums_set = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums_set.add(S[i] + S[j])
        return sums_set

    def count_collisions(sums_set, S):
        # We need to know if there are duplicates.
        # The set of sums loses info about duplicates.
        # So we still need the list or a dict.
        # Let's stick to list for collision count, but use set for validity check in polish.
        # Actually, for score, we need collision count.
        # For polish, we just need to know if add is valid.
        pass

    def score(S):
        sums = []
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(S[i] + S[j])
        c = len(sums) - len(set(sums))
        if c > 0:
            return len(S) - 100 * c
        return len(S)

    def polish(S, sums_set):
        S = list(S)
        S_set = set(S)
        # Recalculate sums_set if needed? 
        # If we pass sums_set, it must be current.
        # Let's assume we update sums_set incrementally.
        
        changed = True
        while changed:
            changed = False
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            random.shuffle(candidates)
            for x in candidates:
                # Check validity using sums_set
                valid = True
                for s in S:
                    if (s + x) in sums_set:
                        valid = False
                        break
                if valid and (x + x) in sums_set:
                    valid = False
                
                if valid:
                    S.append(x)
                    S_set.add(x)
                    # Update sums_set
                    for s in S[:-1]: # s is all previous elements
                        sums_set.add(s + x)
                    sums_set.add(x + x)
                    changed = True
        return S, sums_set

    # Init
    best_S = []
    best_score = -1
    
    for _ in range(10):
        S = list(PRESCRIBED)
        S_set = set(S)
        sums_set = get_sums_set(S)
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if x in S_set:
                continue
            valid = True
            for s in S:
                if (s + x) in sums_set:
                    valid = False
                    break
            if valid and (x + x) in sums_set:
                valid = False
            
            if valid:
                S.append(x)
                S_set.add(x)
                for s in S[:-1]:
                    sums_set.add(s + x)
                sums_set.add(x + x)
        
        S, sums_set = polish(S, sums_set)
        s = score(S)
        if s > best_score:
            best_score = s
            best_S = list(S)
            best_sums_set = set(sums_set) # Copy
            
    current_S = list(best_S)
    current_sums_set = set(best_sums_set)
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    while time.time() - start_time < DEADLINE:
        c = score(current_S) # This is expensive. 
        # We can track collisions count incrementally? Hard.
        # But score is only needed for acceptance.
        # We can just use len(S) if we assume we stay valid?
        # No, we might accept invalid moves.
        
        # Heuristic: If we are at a local max (valid), we want to swap.
        # Let's just calculate score.
        
        new_S = list(current_S)
        new_S_set = set(new_S)
        new_sums_set = set(current_sums_set) # Copy
        
        # Determine move
        # Check if current is valid
        # We can check if len(current_sums_set) == len(current_S)*(len(current_S)+1)//2
        num_sums_expected = len(current_S) * (len(current_S) + 1) // 2
        is_valid = (len(current_sums_set) == num_sums_expected)
        
        if not is_valid:
             # Remove an element involved in collision
             # Find collisions
             sums_list = []
             for i in range(len(new_S)):
                 for j in range(i, len(new_S)):
                     sums_list.append(new_S[i] + new_S[j])
             sum_counts = {}
             for s in sums_list:
                 sum_counts[s] = sum_counts.get(s, 0) + 1
             colliding_sums = [s for s, cnt in sum_counts.items() if cnt > 1]
             
             involved = set()
             for i in range(len(new_S)):
                 for j in range(i, len(new_S)):
                     if new_S[i] + new_S[j] in colliding_sums:
                         involved.add(new_S[i])
                         involved.add(new_S[j])
                         
             removable = [y for y in involved if y not in PRESCRIBED]
             if removable:
                 y = random.choice(removable)
                 # Remove y
                 new_S.remove(y)
                 new_S_set.discard(y)
                 # Update sums_set
                 # Remove sums involving y
                 for s in new_S:
                     new_sums_set.discard(s + y)
                 new_sums_set.discard(y + y)
             else:
                 # Stuck
                 pass
        else:
            # Valid
            move_type = random.random()
            if move_type < 0.5:
                # Add
                x = random.randint(1, N)
                if x not in new_S_set:
                    # Check validity
                    valid = True
                    for s in new_S:
                        if (s + x) in new_sums_set:
                            valid = False
                            break
                    if valid and (x + x) in new_sums_set:
                        valid = False
                    
                    if valid:
                        new_S.append(x)
                        new_S_set.add(x)
                        for s in new_S[:-1]:
                            new_sums_set.add(s + x)
                        new_sums_set.add(x + x)
                    else:
                        # Reject? Or accept with low prob?
                        # If we reject, we don't change state.
                        # Let's just not apply the move if invalid and we wanted to add.
                        # But SA should explore invalid states?
                        # For this task, valid sets are much better.
                        # Let's skip this move if invalid.
                        continue 
            else:
                # Swap
                non_prescribed = [y for y in new_S if y not in PRESCRIBED]
                if non_prescribed:
                    y = random.choice(non_prescribed)
                    # Remove y
                    new_S.remove(y)
                    new_S_set.discard(y)
                    for s in new_S:
                        new_sums_set.discard(s + y)
                    new_sums_set.discard(y + y)
                    
                    # Add x
                    x = random.randint(1, N)
                    while x in new_S_set:
                        x = random.randint(1, N)
                    
                    # Check validity of adding x
                    valid = True
                    for s in new_S:
                        if (s + x) in new_sums_set:
                            valid = False
                            break
                    if valid and (x + x) in new_sums_set:
                        valid = False
                    
                    if valid:
                        new_S.append(x)
                        new_S_set.add(x)
                        for s in new_S[:-1]:
                            new_sums_set.add(s + x)
                        new_sums_set.add(x + x)
                    else:
                        # Invalid swap. Revert?
                        # Just skip this move.
                        continue
                else:
                    # Only prescribed
                    x = random.randint(1, N)
                    if x not in new_S_set:
                        valid = True
                        for s in new_S:
                            if (s + x) in new_sums_set:
                                valid = False
                                break
                        if valid and (x + x) in new_sums_set:
                            valid = False
                        if valid:
                            new_S.append(x)
                            new_S_set.add(x)
                            for s in new_S[:-1]:
                                new_sums_set.add(s + x)
                            new_sums_set.add(x + x)
                        else:
                            continue

        # Calculate new score
        new_score = score(new_S)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_S = new_S
            current_sums_set = new_sums_set
            current_score = new_score
            
            # Polish if size increased or if we want to be sure
            # Polish is fast now?
            # It tries to add more.
            # Only polish if we added something or removed something?
            # If we swapped, size is same.
            # Let's polish if len(current_S) > len(best_S) or random() < 0.1
            if len(current_S) >= len(best_S) or random.random() < 0.2:
                current_S, current_sums_set = polish(current_S, current_sums_set)
                current_score = score(current_S)
                
            if current_score > best_score:
                best_score = current_score
                best_S = list(current_S)
                best_sums_set = set(current_sums_set)
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_S
