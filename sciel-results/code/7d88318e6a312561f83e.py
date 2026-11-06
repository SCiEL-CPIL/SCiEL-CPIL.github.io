import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    initial = [1, 2, 5, 11]
    base_diffs = set()
    for i in range(len(initial)):
        for j in range(i + 1, len(initial)):
            base_diffs.add(abs(initial[i] - initial[j]))
            
    best_solution = initial
    best_size = len(initial)
    
    candidates = list(range(1, 285))
    
    def backtrack(current_list, current_set, used_diffs_set, current_size, order):
        nonlocal best_solution, best_size, deadline
        
        if time.time() > deadline:
            return False # Signal timeout
        
        if current_size > best_size:
            best_size = current_size
            best_solution = list(current_list)
            
        if current_size == 20:
            return True # Found optimal
            
        # Try candidates in the specified order
        for x in order:
            if x in current_set:
                continue
                
            valid = True
            new_diffs = []
            for y in current_list:
                d = abs(x - y)
                if d in used_diffs_set:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            current_list.append(x)
            current_set.add(x)
            used_diffs_set.update(new_diffs)
            
            if backtrack(current_list, current_set, used_diffs_set, current_size + 1, order):
                return True
                
            current_list.pop()
            current_set.remove(x)
            used_diffs_set.difference_update(new_diffs)
            
            if time.time() > deadline:
                return False
                
        return False

    # Attempt 1: Natural order
    if not backtrack(initial, set(initial), base_diffs.copy(), len(initial), candidates):
        # If not found 20 or timed out, try random orders
        attempts = 0
        while time.time() < deadline - 5 and attempts < 50:
            random.shuffle(candidates)
            if backtrack(initial, set(initial), base_diffs.copy(), len(initial), candidates):
                break
            attempts += 1
            
    return best_solution
