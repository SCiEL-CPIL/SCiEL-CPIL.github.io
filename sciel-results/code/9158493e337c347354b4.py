import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    best_set = list(range(192, 257))
    best_len = len(best_set)
    
    def can_add(candidate, s_list):
        # Condition 1: candidate as 'a' - check pairs (b,c) with candidate < b <= c
        gt_start = bisect.bisect_right(s_list, candidate)
        gt = s_list[gt_start:]
        for i in range(len(gt)):
            b = gt[i]
            for j in range(i, len(gt)):
                c = gt[j]
                if (b + c) % candidate == 0:
                    return False
        
        # Condition 2: candidate as 'b' - check pairs (a,c) with a < candidate <= c
        lt_end = bisect.bisect_left(s_list, candidate)
        ge_start = bisect.bisect_left(s_list, candidate)
        lt = s_list[:lt_end]
        ge = s_list[ge_start:]
        for a in lt:
            for c in ge:
                if (candidate + c) % a == 0:
                    return False
        
        # Condition 3: candidate as 'c' - check pairs (a,b) with a < b <= candidate
        le_end = bisect.bisect_right(s_list, candidate)
        le = s_list[:le_end]
        for i in range(len(le)):
            a = le[i]
            for j in range(i + 1, len(le)):
                b = le[j]
                if (b + candidate) % a == 0:
                    return False
        
        return True
    
    # Multi-start greedy from different upper-tail starting points
    for start in range(256, 100, -1):
        if time.time() > deadline:
            break
        
        current_set = set(range(start, 257))
        current_list = list(range(start, 257))
        
        for candidate in range(start - 1, 0, -1):
            if time.time() > deadline:
                break
            if candidate in current_set:
                continue
            if can_add(candidate, current_list):
                current_set.add(candidate)
                bisect.insort(current_list, candidate)
        
        if len(current_set) > best_len:
            best_set = list(current_set)
            best_len = len(current_set)
    
    # Local improvement pass
    if time.time() < deadline - 3.0:
        best_set_sorted = sorted(best_set)
        best_set_set = set(best_set_sorted)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for candidate in range(1, 257):
                if time.time() > deadline:
                    break
                if candidate in best_set_set:
                    continue
                if can_add(candidate, best_set_sorted):
                    best_set_set.add(candidate)
                    bisect.insort(best_set_sorted, candidate)
                    improved = True
                    break
        best_set = list(best_set_set)
    
    return best_set
