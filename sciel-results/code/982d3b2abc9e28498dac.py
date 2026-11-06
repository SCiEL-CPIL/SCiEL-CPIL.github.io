import time

def construct():
    start_time = time.time()
    timeout = 110.0
    
    a = [0]
    
    def get_diffs(s):
        return set(y - x for i, x in enumerate(s) for y in s[i + 1:])

    def is_valid_addition(a, new_elements):
        # new_elements is a list of ints to add
        trial = sorted(a + new_elements)
        # Check size
        if len(trial) > 32:
            return False
        # Check Sidon
        diffs = get_diffs(trial)
        if len(diffs) != len(trial) * (len(trial) - 1) // 2:
            return False
        return True

    current_diffs = get_diffs(a)
    
    for d in range(1, 31):
        if time.time() - start_time > timeout:
            break
            
        if d in current_diffs:
            continue
            
        found = False
        candidates = set()
        for v in a:
            candidates.add(v + d)
            if v - d >= 0:
                candidates.add(v - d)
        
        # Try adding single element
        for x in sorted(candidates):
            if is_valid_addition(a, [x]):
                a = sorted(a + [x])
                current_diffs = get_diffs(a)
                found = True
                break
        
        if not found:
            # Try adding pair y, y+d
            # Start y from max(a) + 1
            y = max(a) + 1
            while y < 10**6:
                if is_valid_addition(a, [y, y + d]):
                    a = sorted(a + [y, y + d])
                    current_diffs = get_diffs(a)
                    found = True
                    break
                y += 1
                if time.time() - start_time > timeout:
                    break
            if not found:
                # Failed to cover d
                break

    final_diffs = get_diffs(a)
    if all(d in final_diffs for d in range(1, 31)):
        return a
    else:
        return a

