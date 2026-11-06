import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    selected = []
    sums = set()
    
    # Greedy selection from 1 to 4096
    for x in range(1, 4097):
        if time.time() > deadline:
            break
            
        # Generate all new sums involving x
        # 1. (x, x, x)
        # 2. (x, x, y) for y in selected
        # 3. (x, y, z) for y <= z in selected
        
        new_sums = []
        new_sums.append(3 * x)
        
        for y in selected:
            new_sums.append(2 * x + y)
            
        for i in range(len(selected)):
            y = selected[i]
            for j in range(i, len(selected)):
                z = selected[j]
                new_sums.append(x + y + z)
                
        # Check for internal collisions in new_sums
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check for collisions with existing sums
        conflict = False
        for s in new_sums:
            if s in sums:
                conflict = True
                break
                
        if not conflict:
            selected.append(x)
            sums.update(new_sums)
            
    return selected
