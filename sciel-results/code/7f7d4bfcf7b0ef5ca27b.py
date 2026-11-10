"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute: for each x and d, store the 4-AP containing x at each position
    # Actually, let's use bitsets for efficiency
    
    N = 256
    # selected as a set for membership tests
    selected = set()
    
    # Greedy: add elements that don't create 4-APs
    # To be smart, process in an order that maximizes 3-AP density
    # Start with all numbers, then remove those in 4-APs? No, let's do forward greedy.
    
    # Better greedy: try all elements, add those that create most 3-APs
    # But first get a valid base set
    
    # Phase 1: Forward greedy (simple)
    for x in range(1, N+1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # x as 1st: x, x+d, x+2d, x+3d
        for d in range(1, N - x + 1):
            if d*3 + x <= N and (x+d) in selected and (x+2*d) in selected and (x+3*d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, x):
            if x+2*d <= N and (x-d) in selected and (x+d) in selected and (x+2*d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, (x+1)//2):
            if x+d <= N and (x-2*d) in selected and (x-d) in selected and (x+d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, (x+2)//3):
            if (x-3*d) in selected and (x-2*d) in selected and (x-d) in selected:
                creates_4ap = True
                break
        if creates_4ap:
            continue
        selected.add(x)
    
    # Count current 3-APs
    def count_3aps(sel):
        cnt = 0
        for a in sel:
            for d in range(1, N - a + 1):
                if a + 2*d <= N and (a+d) in sel and (a+2*d) in sel:
                    cnt += 1
        return cnt
    
    current_3aps = count_3aps(selected)
    
    # Phase 2: Local search with time budget
    def check_4ap(sel, x):
        """Check if x creates a 4-AP with sel."""
        for d in range(1, N - x + 1):
            if x+3*d <= N and (x+d) in sel and (x+2*d) in sel and (x+3*d) in sel:
                return True
        for d in range(1, x):
            if x+2*d <= N and (x-d) in sel and (x+d) in sel and (x+2*d) in sel:
                return True
        for d in range(1, (x+1)//2):
            if x+d <= N and (x-2*d) in sel and (x-d) in sel and (x+d) in sel:
                return True
        for d in range(1, (x+2)//3):
            if (x-3*d) in sel and (x-2*d) in sel and (x-d) in sel:
                return True
        return False
    
    def gain_3aps(sel, x):
        """New 3-APs if x added to sel."""
        cnt = 0
        for d in range(1, N - x + 1):
            if x+2*d <= N and (x+d) in sel and (x+2*d) in sel:
                cnt += 1
        for d in range(1, x):
            if (x-d) in sel and (x+d) in sel:
                cnt += 1
        for d in range(1, (x+1)//2):
            if (x-2*d) in sel and (x-d) in sel:
                cnt += 1
        return cnt
    
    def loss_3aps(sel, x):
        """3-APs lost if x removed from sel."""
        cnt = 0
        for d in range(1, N - x + 1):
            if x+2*d <= N and (x+d) in sel and (x+2*d) in sel:
                cnt += 1
        for d in range(1, x):
            if (x-d) in sel and (x+d) in sel:
                cnt += 1
        for d in range(1, (x+1)//2):
            if (x-2*d) in sel and (x-d) in sel:
                cnt += 1
        return cnt
    
    # Local search iterations
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Try adding
        best_gain = 0
        best_add = None
        for x in range(1, N+1):
            if x in selected:
                continue
            if check_4ap(selected, x):
                continue
            g = gain_3aps(selected, x)
            if g > best_gain:
                best_gain = g
                best_add = x
        if best_add is not None and best_gain > 0:
            selected.add(best_add)
            current_3aps += best_gain
            improved = True
            continue
        
        # Try swap: remove one, add one
        best_net = 0
        best_swap = None
        items = list(selected)
        random.shuffle(items)
        for rem in items:
            if (time.time() - start_time) > deadline:
                break
            lost = loss_3aps(selected, rem)
            # Try adding candidates
            for add in range(1, N+1):
                if add == rem or add in selected:
                    continue
                # Check 4-AP with selected minus rem
                # We need to verify no 4-AP in (selected - {rem}) ∪ {add}
                # A 4-AP involving add must have at least one other element from selected-{rem}
                # A 4-AP not involving add must be in selected-{rem}, which is fine since selected has no 4-APs
                # So we only need to check 4-APs that include add
                ok = True
                for d in range(1, N - add + 1):
                    if add+3*d <= N and (add+d) in selected and (add+d) != rem and (add+2*d) in selected and (add+2*d) != rem and (add+3*d) in selected and (add+3*d) != rem:
                        ok = False
                        break
                if ok:
                    for d in range(1, add):
                        if add+2*d <= N and (add-d) in selected and (add-d) != rem and (add+d) in selected and (add+d) != rem and (add+2*d) in selected and (add+2*d) != rem:
                            ok = False
                            break
                if ok:
                    for d in range(1, (add+1)//2):
                        if add+d <= N and (add-2*d) in selected and (add-2*d) != rem and (add-d) in selected and (add-d) != rem and (add+d) in selected and (add+d) != rem:
                            ok = False
                            break
                if ok:
                    for d in range(1, (add+2)//3):
                        if (add-3*d) in selected and (add-3*d) != rem and (add-2*d) in selected and (add-2*d) != rem and (add-d) in selected and (add-d) != rem:
                            ok = False
                            break
                if not ok:
                    continue
                g = gain_3aps(selected - {rem}, add)
                net = g - lost
                if net > best_net:
                    best_net = net
                    best_swap = (rem, add)
        
        if best_swap is not None and best_net > 0:
            rem, add = best_swap
            selected.discard(rem)
            selected.add(add)
            current_3aps += best_net
            improved = True
    
    return sorted(selected)

# EVOLVE-BLOCK-END
