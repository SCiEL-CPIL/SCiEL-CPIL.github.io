import time

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin before 60s kill (or 90s if evaluator kills later, prompt says 60s finish, 90s kill)
    
    N = 256
    selected = []
    selected_set = set()
    
    # Greedy Initialization
    # We only need to check if x forms a 3-AP as the LARGEST term
    # because we are adding in increasing order.
    for x in range(1, N + 1):
        valid = True
        for i in range(len(selected)):
            a = selected[i]
            for j in range(i + 1, len(selected)):
                b = selected[j]
                # x is largest: x = 2b - a
                if 2 * b - a == x:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
            
    current_score = sum(1.0 / x for x in selected)
    
    def is_valid(s_set):
        # Full check
        for a in s_set:
            for b in s_set:
                if a >= b: continue
                c = 2 * b - a
                if c > b and c in s_set:
                    return False
        return True

    improved = True
    while improved and time.time() - start_time < limit:
        improved = False
        # Try 1-swaps
        # Candidates are numbers not in set, sorted by value (ascending) to prioritize high weight
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        candidates.sort()
        
        for x in candidates:
            if time.time() - start_time > limit: break
            if x in selected_set: continue
            
            # Find conflicts
            # x conflicts with pair (a,b) if {x,a,b} is 3-AP
            # We need to remove elements to break all such pairs.
            # Strategy: Try removing each element involved in conflicts.
            
            conflict_elems = set()
            s_list = list(selected_set)
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    # Check if {x, a, b} is 3-AP
                    # x = 2b - a
                    # x = 2a - b
                    # 2x = a + b
                    if 2 * b - a == x or 2 * a - b == x or 2 * x == a + b:
                        conflict_elems.add(a)
                        conflict_elems.add(b)
            
            if not conflict_elems:
                # No conflicts
                new_set = set(selected_set)
                new_set.add(x)
                if is_valid(new_set):
                    selected_set = new_set
                    selected = list(new_set)
                    current_score += 1.0 / x
                    improved = True
                    break
            
            # Try removing one element from conflict_elems
            # Sort by value descending (remove large, low weight first)
            conflict_elems_list = sorted(conflict_elems, reverse=True)
            for r in conflict_elems_list:
                if time.time() - start_time > limit: break
                new_set = set(selected_set)
                new_set.remove(r)
                new_set.add(x)
                if is_valid(new_set):
                    gain = 1.0 / x - 1.0 / r
                    if gain > 0:
                        selected_set = new_set
                        selected = list(new_set)
                        current_score += gain
                        improved = True
                        break
            if improved: break

    return selected
