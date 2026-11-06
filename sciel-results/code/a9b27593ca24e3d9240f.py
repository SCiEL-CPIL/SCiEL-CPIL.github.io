import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for 90s limit
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    ap3_indices = {}
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3s.append((a, b, c))
            ap3_indices[(a, b, c)] = len(ap3s) - 1
            
    ap4s = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Map element to APs it is part of
    elem_to_ap3 = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(ap3s):
        elem_to_ap3[a].append(i)
        elem_to_ap3[b].append(i)
        elem_to_ap3[c].append(i)
        
    elem_to_ap4 = [[] for _ in range(N + 1)]
    for i, (a, b, c, d) in enumerate(ap4s):
        elem_to_ap4[a].append(i)
        elem_to_ap4[b].append(i)
        elem_to_ap4[c].append(i)
        elem_to_ap4[d].append(i)
        
    # Bitmask representation for fast operations
    # We'll use a numpy boolean array for S
    
    def count_3aps(S_mask):
        """Count 3-APs using vectorized numpy operations"""
        count = 0
        for a, b, c in ap3s:
            if S_mask[a - 1] and S_mask[b - 1] and S_mask[c - 1]:
                count += 1
        return count
    
    def is_valid_4ap_free(S_mask):
        """Check if S has no 4-APs"""
        for a, b, c, d in ap4s:
            if S_mask[a - 1] and S_mask[b - 1] and S_mask[c - 1] and S_mask[d - 1]:
                return False
        return True
    
    def get_gain(x, S_mask):
        """Calculate gain of adding x to S"""
        gain = 0
        for idx in elem_to_ap3[x]:
            a, b, c = ap3s[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_mask[o - 1] for o in others):
                gain += 1
        return gain
    
    def get_cost(x, S_mask):
        """Calculate cost (loss) of removing x from S"""
        cost = 0
        for idx in elem_to_ap3[x]:
            a, b, c = ap3s[idx]
            others = [a, b, c]
            others.remove(x)
            if all(S_mask[o - 1] for o in others):
                cost += 1
        return cost
    
    def check_4ap_validity(x, S_mask):
        """Check if adding x to S creates a 4-AP"""
        for idx in elem_to_ap4[x]:
            a, b, c, d = ap4s[idx]
            others = [a, b, c, d]
            others.remove(x)
            if all(S_mask[o - 1] for o in others):
                return False
        return True
    
    # Greedy construction
    S = np.zeros(N, dtype=bool)
    S_set = set()
    
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    while True:
        if time.time() - start_time > timeout - 15:
            break
            
        best_x = -1
        best_gain = -1
        valid_candidates = []
        
        for x in candidates:
            if x in S_set:
                continue
                
            # Check 4-AP constraint
            if not check_4ap_validity(x, S):
                continue
                
            # Calculate Gain
            gain = get_gain(x, S)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                valid_candidates = [x]
            elif gain == best_gain:
                valid_candidates.append(x)
                
        if best_x == -1:
            break
            
        if best_gain > 0 and len(valid_candidates) > 1:
            best_x = random.choice(valid_candidates)
            
        S[best_x - 1] = True
        S_set.add(best_x)
        candidates.remove(best_x)
        
    current_score = count_3aps(S)
    
    # Local search with swaps
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and time.time() - start_time < timeout - 5 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try random swaps
        for _ in range(200):
            if time.time() - start_time > timeout - 5:
                break
                
            if len(S_set) == 0:
                break
                
            x = random.choice(list(S_set))
            y_candidates = [y for y in range(1, N + 1) if y not in S_set]
            if not y_candidates:
                break
            y = random.choice(y_candidates)
            
            # Check 4-AP validity for S' = S - {x} + {y}
            valid = True
            for idx in elem_to_ap4[y]:
                a, b, c, d = ap4s[idx]
                others = [a, b, c, d]
                others.remove(y)
                all_in_sp = True
                for o in others:
                    if o == x:
                        continue
                    if not S[o - 1]:
                        all_in_sp = False
                        break
                if all_in_sp:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Calculate Delta
            cost_x = get_cost(x, S)
            benefit_y = 0
            for idx in elem_to_ap3[y]:
                a, b, c = ap3s[idx]
                others = [a, b, c]
                others.remove(y)
                all_in_sp = True
                for o in others:
                    if o == x:
                        continue
                    if not S[o - 1]:
                        all_in_sp = False
                        break
                if all_in_sp:
                    benefit_y += 1
            
            delta = benefit_y - cost_x
            
            if delta > 0:
                S[x - 1] = False
                S[y - 1] = True
                S_set.remove(x)
                S_set.add(y)
                current_score += delta
                improved = True
                break  # Restart local search loop to recalculate
    
    # Try adding more elements if possible
    for x in range(1, N + 1):
        if x in S_set:
            continue
        if time.time() - start_time > timeout - 3:
            break
        if check_4ap_validity(x, S):
            S[x - 1] = True
            S_set.add(x)
    
    return sorted(S_set)

construct()
