import time
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256
    
    def is_valid_addition_fast(a, S_set):
        """Check if a can be added. All elements in S are > a."""
        k_min = 2
        while k_min * a < 2 * (a + 1):
            k_min += 1
        k_max = 2 * n // a
        
        for k in range(k_min, k_max + 1):
            M = k * a
            b_start = max(a + 1, M - n)
            b_end = M // 2
            for b in range(b_start, b_end + 1):
                if b in S_set and (M - b) in S_set:
                    return False
        return True

    def full_validate(S_list):
        n2 = len(S_list)
        S_set = set(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def try_start_from(m_start):
        """Try building a set starting from [m_start, n] and adding below."""
        if time.time() > deadline - 10:
            return None
        
        # Check if base interval [m_start, n] is valid
        base = list(range(m_start, n+1))
        valid = True
        S_set_base = set(base)
        for i in range(len(base)):
            a = base[i]
            for j in range(i+1, len(base)):
                b = base[j]
                for k in range(j, len(base)):
                    c = base[k]
                    if (b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            return None
        
        S_list = base[:]
        S_set = set(S_list)
        
        # Add elements below m_start in descending order
        for a in range(m_start - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            if a in S_set:
                continue
            if is_valid_addition_fast(a, S_set):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
        
        return S_list

    # Try multiple starting thresholds
    best_S_list = None
    best_size = 0
    
    # Start with the known good [171, 256]
    S_list = list(range(171, n+1))
    S_set = set(S_list)
    best_S_list = S_list[:]
    best_size = len(S_list)
    
    # Add elements below 171 in descending order
    for a in range(170, 0, -1):
        if time.time() > deadline - 15:
            break
        if a in S_set:
            continue
        if is_valid_addition_fast(a, S_set):
            pos = bisect.bisect_left(S_list, a)
            S_list.insert(pos, a)
            S_set.add(a)
            if len(S_list) > best_size:
                best_S_list = S_list[:]
                best_size = len(S_list)
    
    # Try other starting thresholds
    for m_start in [165, 160, 155, 150, 145, 140, 135, 130, 128, 125, 120]:
        if time.time() > deadline - 10:
            break
        
        result = try_start_from(m_start)
        if result is not None and len(result) > best_size:
            best_S_list = result
            best_size = len(result)
    
    # Final polish: try to add any remaining elements to best_S_list
    if time.time() < deadline - 8:
        S_list = best_S_list[:]
        S_set = set(S_list)
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            # Try ascending first (smaller elements might fit)
            for a in range(1, n+1):
                if time.time() > deadline - 3:
                    break
                if a in S_set:
                    continue
                if is_valid_addition_fast(a, S_set):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    changed = True
                    if len(S_list) > best_size:
                        best_S_list = S_list[:]
                        best_size = len(S_list)
            
            # Try descending (larger elements might fit)
            for a in range(n, 0, -1):
                if time.time() > deadline - 3:
                    break
                if a in S_set:
                    continue
                if is_valid_addition_fast(a, S_set):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    changed = True
                    if len(S_list) > best_size:
                        best_S_list = S_list[:]
                        best_size = len(S_list)
    
    # Final validation
    if time.time() < deadline:
        S_final = best_S_list
        if len(S_final) > 5 and not full_validate(S_final):
            # Fallback to known valid set
            S_final = list(range(171, n+1))
        return S_final
    
    return best_S_list
