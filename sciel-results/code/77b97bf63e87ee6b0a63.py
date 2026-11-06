<<<<<<< SEARCH
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    
    for restart in range(200):
        if time.time() > deadline - 15:
            break
        
        # Greedy coverage initialization
        S, S_bits = _greedy_init()
        cur_score = _score_set_bits(S_bits)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)
        
        # Targeted swap local search with simulated annealing
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        no_improve = 0
        max_no_improve = 15000
        
        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break
            
            # Get missing differences to target
            missing_ds = _missing_diffs(S_bits)
            
            # Try targeted swaps: pick x from S, y not in S, prioritize if y helps cover missing d
            x = random.choice(list(S))
            
            # With 70% probability, try a targeted swap for a missing d
            if missing_ds and random.random() < 0.7:
                d_target = random.choice(missing_ds)
                # Find y such that adding y could help cover d_target
                # Need a, a+d, a+2d all in S after swap
                # If we add y, we need two elements in S that are y-d and y-2d, or y+d and y+2d, or one of each
                candidates = []
                for a in range(1, 513 - 2*d_target + 1):
                    a1, a2, a3 = a, a + d_target, a + 2*d_target
                    # Check how many are already in S
                    in_s = 0
                    missing_pos = []
                    for pos, val in enumerate([a1, a2, a3]):
                        if S_bits & (1 << val):
                            in_s += 1
                        else:
                            missing_pos.append((pos, val))
                    
                    if in_s == 2 and len(missing_pos) == 1:
                        pos, val = missing_pos[0]
                        if val not in S:
                            candidates.append((val, 1))  # Adding val covers this d
                    elif in_s == 1 and len(missing_pos) == 2:
                        # Would need to add two elements, but we're only swapping one
                        # Skip for now
                        pass
                
                if candidates:
                    # Pick a random candidate
                    y, _ = random.choice(candidates)
                else:
                    # Fall back to random y
                    while True:
                        y = random.choice(all_nums)
                        if y not in S:
                            break
            else:
                # Random swap
                while True:
                    y = random.choice(all_nums)
                    if y not in S:
                        break
            
            # Try swap
            S_bits ^= (1 << x)  # Remove x
            S_bits ^= (1 << y)  # Add y
            
            new_score = _score_set_bits(S_bits)
            
            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                S.discard(x)
                S.add(y)
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                # Reject, revert bits
                S_bits ^= (1 << x)
                S_bits ^= (1 << y)
                no_improve += 1
            
            T = max(T_min, T * cooling)
        
        # Reset temperature for next restart
        T = 5.0
    
    if best_set is None:
        return list(range(1, 33))
    
    return list(best_set)
=======
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_score = 0
    best_set = None
    all_nums = list(range(1, 513))
    not_in_S_mask = np.ones(513, dtype=bool)

    # Precompute for fast scoring: for each d, the valid positions where a, a+d, a+2d all in [1,512]
    # We'll use numpy for scoring.
    # Precompute shift masks for numpy-based scoring
    # Actually, let's use a different approach: represent set as a numpy boolean array of size 513
    # and use vectorized numpy ops for scoring.

    for restart in range(200):
        if time.time() > deadline - 20:
            break

        # Greedy coverage initialization
        S, S_arr = _greedy_init()
        cur_score = _score_arr(S_arr)

        if cur_score > best_score:
            best_score = cur_score
            best_set = set(S)

        # Targeted swap local search with simulated annealing
        T = 5.0
        T_min = 0.05
        cooling = 0.998
        no_improve = 0
        max_no_improve = 20000

        while time.time() < deadline - 5:
            if no_improve > max_no_improve:
                break

            # Get missing differences to target
            missing_ds = _missing_diffs_arr(S_arr)

            # Pick x from S
            x = random.choice(list(S))

            # With 70% probability, try a targeted swap for a missing d
            if missing_ds and random.random() < 0.7:
                d_target = random.choice(missing_ds)
                # Find y such that adding y (and removing x) could help cover d_target
                # We want: after removing x and adding y, there exists a with a, a+d, a+2d all in S
                # This means: in the set S' = (S - {x}) ∪ {y}, some 3-AP with difference d_target exists
                # Cases:
                # 1. y is part of a new 3-AP with two elements from S - {x}
                # 2. x was breaking an existing near-3-AP that y completes
                candidates = []
                # Case 1: y completes a 3-AP with two elements from S
                for a in range(1, 513 - 2 * d_target + 1):
                    a1, a2, a3 = a, a + d_target, a + 2 * d_target
                    # Count how many of a1, a2, a3 are in S (excluding x)
                    in_s_count = 0
                    missing_val = None
                    for val in (a1, a2, a3):
                        if val == x:
                            continue  # x is being removed, so treat as not in S
                        if val in S:
                            in_s_count += 1
                        else:
                            missing_val = val
                    if in_s_count == 2 and missing_val is not None and missing_val != x:
                        candidates.append(missing_val)

                if candidates:
                    y = random.choice(candidates)
                else:
                    # Fall back: pick y that maximizes potential
                    y = _random_y_not_in_S(S, all_nums)
            else:
                # Random swap
                y = _random_y_not_in_S(S, all_nums)

            if y == x:
                continue

            # Try swap: remove x, add y
            S.discard(x)
            S.add(y)
            S_arr[x] = False
            S_arr[y] = True

            new_score = _score_arr(S_arr)

            delta = new_score - cur_score
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(S)
            else:
                # Reject, revert
                S.discard(y)
                S.add(x)
                S_arr[y] = False
                S_arr[x] = True
                no_improve += 1

            T = max(T_min, T * cooling)

    # Final polish: try all single swaps from best_set
    if best_set is not None and time.time() < deadline - 10:
        best_set, best_score = _polish(best_set, best_score, all_nums)

    if best_set is None:
        return list(range(1, 33))

    return list(best_set)
>>>>>>> REPLACE

