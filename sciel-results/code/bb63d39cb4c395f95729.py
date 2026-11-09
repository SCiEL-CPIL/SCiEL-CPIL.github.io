import time

def construct():
    t_start = time.time()
    DEADLINE = t_start + 55.0
    N = 4096
    MAX_SIZE = 80
    
    def try_greedy(start_val=1):
        selected = []
        sums = set()
        
        for x in range(start_val, N + 1):
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE:
                break
            
            # Generate new triple sums involving x
            # 1. x, x, x
            new_sums = [3 * x]
            
            # 2. x, x, y
            for y in selected:
                new_sums.append(2 * x + y)
                
            # 3. x, y, z where y <= z from selected
            n_sel = len(selected)
            for i in range(n_sel):
                y = selected[i]
                for j in range(i, n_sel):
                    z = selected[j]
                    new_sums.append(x + y + z)
            
            # Check for collisions
            valid = True
            for s in new_sums:
                if s in sums:
                    valid = False
                    break
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
                
        return selected

    # Standard greedy from 1
    result = try_greedy(1)
    
    return result
