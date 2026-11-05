import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 192
    
    def can_add(x, S):
        """S is a list of selected elements. Check if x can be added."""
        m = len(S)
        if m < 3:
            if m >= 2:
                for a in S:
                    others = [v for v in S if v != a]
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            if (x + others[i] + others[j]) % a == 0:
                                return False
            return True
        
        S_arr = np.array(S)
        
        # Check 1: x does not divide any triple sum in S
        S_mod_x = S_arr % x
        for i in range(m):
            for j in range(i+1, m):
                target = (-(S[i] + S[j])) % x
                mask = S_mod_x == target
                mask[i] = False
                mask[j] = False
                if mask.any():
                    return False
        
        # Check 2: for each a in S, a does not divide x + b + c for any pair (b,c) in S\{a}
        for idx_a, a in enumerate(S):
            others_idx = [i for i in range(m) if i != idx_a]
            if len(others_idx) < 2:
                continue
            others_arr = S_arr[others_idx]
            target = (-x) % a
            others_mod_a = others_arr % a
            for i in range(len(others_idx)):
                for j in range(i+1, len(others_idx)):
                    if (others_mod_a[i] + others_mod_a[j]) % a == target:
                        return False
        
        return True
    
    def is_valid(S):
        """Check if S is a valid set."""
        m = len(S)
        for i in range(m):
            x = S[i]
            # x should not divide sum of any 3 distinct other elements
            others = [S[j] for j in range(m) if j != i]
            for a_idx in range(len(others)):
                a = others[a_idx]
                # Check if a divides x + b + c for any pair (b,c) in others\{a}
                rest = [v for idx, v in enumerate(others) if idx != a_idx]
                for b_idx in range(len(rest)):
                    b = rest[b_idx]
                    for c_idx in range(b_idx+1, len(rest)):
                        c = rest[c_idx]
                        if (a + b + c) % x == 0:
                            return False
        return True
    
    def remove_and_check(S, idx_remove):
        """Remove element at idx_remove from S, return new list."""
        return S[:idx_remove] + S[idx_remove+1:]
    
    def try_add(S, x):
        """Try to add x to S, return new list if valid, else None."""
        if x in S:
            return None
        if can_add(x, S):
            return S + [x]
        return None
    
    # Greedy initialization with multiple orders
    best_size = 0
    best_set = []
    
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
        return S
    
    # Try several greedy orders
    orders = [list(range(n, 0, -1))]
    for _ in range(5):
        o = list(range(1, n+1))
        random.shuffle(o)
        orders.append(o)
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Simulated annealing with remove-1/add-2 moves
    if best_set and time.time() < deadline - 5:
        S = list(best_set)
        S_set = set(S)
        all_nums = set(range(1, n+1))
        
        current_size = len(S)
        best_sa_size = current_size
        best_sa_set = list(S)
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        total_steps = 5000
        T = T_start
        
        for step in range(total_steps):
            if time.time() > deadline - 3:
                break
            
            # Annealing schedule
            T = T_start * (T_end / T_start) ** (step / total_steps)
            
            # Choose move type: 70% remove-1/add-2, 30% single add
            move_type = random.random()
            
            if move_type < 0.7 and len(S) >= 3:
                # Remove one element, try to add two others
                idx_remove = random.randint(0, len(S)-1)
                removed = S[idx_remove]
                S_new = remove_and_check(S, idx_remove)
                
                # Try to add two elements from remaining pool
                candidates = list(all_nums - set(S_new))
                random.shuffle(candidates)
                
                added_count = 0
                for x in candidates:
                    if added_count >= 2:
                        break
                    if can_add(x, S_new):
                        S_new.append(x)
                        added_count += 1
                
                new_size = len(S_new)
                delta = new_size - current_size
                
                if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                    S = S_new
                    S_set = set(S)
                    current_size = new_size
                    if current_size > best_sa_size:
                        best_sa_size = current_size
                        best_sa_set = list(S)
            else:
                # Single add
                candidates = list(all_nums - S_set)
                if candidates:
                    x = random.choice(candidates)
                    if can_add(x, S):
                        S_new = S + [x]
                        new_size = len(S_new)
                        delta = new_size - current_size
                        if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                            S = S_new
                            S_set = set(S)
                            current_size = new_size
                            if current_size > best_sa_size:
                                best_sa_size = current_size
                                best_sa_set = list(S)
        
        if best_sa_size > best_size:
            best_size = best_sa_size
            best_set = best_sa_set
    
    # Final greedy fill
    if best_set and time.time() < deadline - 2:
        S = list(best_set)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in range(1, n+1):
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return best_set
