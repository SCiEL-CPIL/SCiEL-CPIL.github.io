import time
import random

def construct():
    N = 256
    deadline = time.time() + 85.0  # Safety margin
    
    def is_valid_addition(x, S):
        for a in S:
            c = 2 * a - x
            if c in S:
                return False
            c2 = 2 * x - a
            if c2 in S:
                return False
        return True

    def get_weight(S):
        return sum(1.0 / x for x in S)

    def greedy_fill(S):
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S and is_valid_addition(x, S):
                    S.add(x)
                    changed = True
        return S

    # Initial Greedy
    S = set()
    greedy_fill(S)
    
    best_S = set(S)
    best_weight = get_weight(S)
    
    no_improve_count = 0
    max_no_improve = 50  # Trigger kick after 50 failed swaps/adds
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding any valid x
        for x in range(1, N + 1):
            if x not in S and is_valid_addition(x, S):
                S.add(x)
                improved = True
                if get_weight(S) > best_weight:
                    best_weight = get_weight(S)
                    best_S = set(S)
                no_improve_count = 0
                break # Restart loop to check for more additions
        
        if improved:
            continue
            
        # 2. Try swaps
        S_list = list(S)
        swap_performed = False
        # Prioritize removing larger elements to allow adding smaller ones (higher weight)
        for y in sorted(S_list, reverse=True):
            S_without_y = S - {y}
            for x in range(1, y):
                if x in S:
                    continue
                if is_valid_addition(x, S_without_y):
                    S.remove(y)
                    S.add(x)
                    swap_performed = True
                    improved = True
                    break
            if swap_performed:
                break
        
        if swap_performed:
            if get_weight(S) > best_weight:
                best_weight = get_weight(S)
                best_S = set(S)
            no_improve_count = 0
            continue
            
        # 3. If stuck, perform a kick: remove random element and re-greedy
        if no_improve_count > max_no_improve:
            if len(S) > 0:
                y = random.choice(list(S))
                S.remove(y)
                greedy_fill(S)
                
                current_weight = get_weight(S)
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_S = set(S)
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert to best_S to avoid drifting to worse states?
                    # Or keep current S to explore?
                    # If we keep current S, and it's worse, we might get stuck in a worse basin.
                    # Reverting to best_S is safer for "hill climbing".
                    # But then we are back to the same state, and will kick again.
                    # Let's revert to best_S if kick fails.
                    S = set(best_S)
            else:
                break
        else:
            no_improve_count += 1

    return list(best_S)
