import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 192
    max_size = 48
    
    selected = []
    selected_set = set()
    
    # Iterate from largest to smallest
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if x can be added to selected
        valid = True
        
        s_len = len(selected)
        
        # Build res_map if needed for Check 1 or Check 2a
        res_map = None
        if s_len >= 2:
            res_map = {}
            for v in selected:
                r = v % x
                if r not in res_map:
                    res_map[r] = []
                res_map[r].append(v)
        
        # Check 1: x does not divide sum of any 3 distinct elements in selected
        # i.e., no a,b,c in selected distinct such that (a+b+c) % x == 0
        if s_len >= 3 and valid:
            for i in range(s_len):
                a = selected[i]
                for j in range(i+1, s_len):
                    b = selected[j]
                    target_res = (- (a + b)) % x
                    if target_res in res_map:
                        # Check if there is a c in res_map[target_res] such that c != a and c != b
                        # Since a,b are in selected, they might be in this list
                        for c in res_map[target_res]:
                            if c != a and c != b:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
        
        # Check 2a: x does not divide (a+b) for any distinct a,b in selected
        # i.e., no a,b in selected distinct such that (a+b) % x == 0
        if s_len >= 2 and valid:
            for i in range(s_len):
                a = selected[i]
                a_res = a % x
                target_res = (-a_res) % x
                if target_res in res_map:
                     # Check if there is a b in res_map[target_res] such that b != a
                     # Note: if a_res == target_res, then 2*a_res == 0 mod x
                     # We need b != a
                     for b in res_map[target_res]:
                         if b != a:
                             valid = False
                             break
                if not valid:
                    break
        
        if valid:
            # Check 2b: For any distinct a,b in selected, and any y in selected:
            # y does not divide (x + a + b)
            if s_len >= 2:
                for i in range(s_len):
                    a = selected[i]
                    for j in range(i+1, s_len):
                        b = selected[j]
                        s_ab = x + a + b
                        # Check if any y in selected divides s_ab
                        for y in selected:
                            if s_ab % y == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
            
            if len(selected) >= max_size:
                break

    return selected
