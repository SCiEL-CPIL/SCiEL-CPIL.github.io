import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 256
    
    # 1. Greedy Construction
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        ok = True
        for b in selected:
            a = 2 * b - x
            if a in selected_set and a != b:
                ok = False
                break
        if ok:
            selected.append(x)
            selected_set.add(x)
            
    # 2. Local Search (Swap)
    rejected = [x for x in range(1, n + 1) if x not in selected_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort selected in descending order to try removing largest first
        selected.sort(reverse=True)
        
        for i in range(len(selected)):
            x = selected[i]
            
            # Try to find smallest y in rejected to swap with x
            for y in rejected:
                if y >= x:
                    break
                
                # Check compatibility of y with selected - {x}
                compatible = True
                for a in selected:
                    if a == x:
                        continue
                    
                    # Case 1: y is middle (a, y, b) => b = 2y - a
                    b = 2 * y - a
                    if b in selected_set and b != x and b != a:
                        compatible = False
                        break
                    
                    # Case 2: y is an end (y, mid, a or a, mid, y) => mid = (a+y)/2
                    if (a + y) % 2 == 0:
                        mid = (a + y) // 2
                        if mid in selected_set and mid != x and mid != a and mid != y:
                            compatible = False
                            break
                
                if compatible:
                    # Perform swap
                    selected[i] = y
                    selected_set.remove(x)
                    selected_set.add(y)
                    
                    # Update rejected
                    rejected.remove(y)
                    rejected.append(x)
                    rejected.sort()
                    
                    improved = True
                    break # Restart outer loop
            
            if improved:
                break

    return selected
