import random
import time

def construct():
    n = 256
    best_size = 0
    best_sets = ([], [])
    
    def try_add(set_list, sum_mask, diff_mask_self, diff_mask_other, x):
        new_sums = 1 << (2 * x)
        new_diffs = 0
        
        for s in set_list:
            new_sums |= 1 << (x + s)
            d1 = x - s
            d2 = s - x
            new_diffs |= 1 << (d1 + 256)
            new_diffs |= 1 << (d2 + 256)
        
        if new_sums & sum_mask:
            return None
        if new_diffs & diff_mask_self:
            return None
        if new_diffs & diff_mask_other:
            return None
            
        new_sum_mask = sum_mask | new_sums
        new_diff_mask = diff_mask_self | new_diffs
        new_set = set_list + [x]
        
        return new_set, new_sum_mask, new_diff_mask

    def local_search(A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B, time_limit):
        A = list(A)
        B = list(B)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            for x in range(1, n + 1):
                if x in A:
                    continue
                res = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x)
                if res:
                    A, sum_mask_A, diff_mask_A = res
                    improved = True
                    break
            
            for x in range(1, n + 1):
                if x in B:
                    continue
                res = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x)
                if res:
                    B, sum_mask_B, diff_mask_B = res
                    improved = True
                    break
            
            if time.time() >= time_limit:
                break
            for i, a in enumerate(A):
                if time.time() >= time_limit:
                    break
                sum_mask_A_no_a = sum_mask_A
                diff_mask_A_no_a = diff_mask_A
                for s in A:
                    if s == a:
                        continue
                    sum_mask_A_no_a &= ~(1 << (a + s))
                    d = a - s
                    diff_mask_A_no_a &= ~(1 << (d + 256))
                    diff_mask_A_no_a &= ~(1 << (-d + 256))
                sum_mask_A_no_a &= ~(1 << (2 * a))
                
                for x in range(1, n + 1):
                    if x in A:
                        continue
                    new_sums = 1 << (2 * x)
                    new_diffs = 0
                    for s in A:
                        if s == a:
                            continue
                        new_sums |= 1 << (x + s)
                        d1 = x - s
                        d2 = s - x
                        new_diffs |= 1 << (d1 + 256)
                        new_diffs |= 1 << (d2 + 256)
                    
                    if new_sums & sum_mask_A_no_a:
                        continue
                    if new_diffs & diff_mask_A_no_a:
                        continue
                    if new_diffs & diff_mask_B:
                        continue
                    
                    A[i] = x
                    sum_mask_A = sum_mask_A_no_a | new_sums
                    diff_mask_A = diff_mask_A_no_a | new_diffs
                    improved = True
                    break
                if improved:
                    break
            
            if time.time() >= time_limit:
                break
            for i, b in enumerate(B):
                if time.time() >= time_limit:
                    break
                sum_mask_B_no_b = sum_mask_B
                diff_mask_B_no_b = diff_mask_B
                for s in B:
                    if s == b:
                        continue
                    sum_mask_B_no_b &= ~(1 << (b + s))
                    d = b - s
                    diff_mask_B_no_b &= ~(1 << (d + 256))
                    diff_mask_B_no_b &= ~(1 << (-d + 256))
                sum_mask_B_no_b &= ~(1 << (2 * b))
                
                for x in range(1, n + 1):
                    if x in B:
                        continue
                    new_sums = 1 << (2 * x)
                    new_diffs = 0
                    for s in B:
                        if s == b:
                            continue
                        new_sums |= 1 << (x + s)
                        d1 = x - s
                        d2 = s - x
                        new_diffs |= 1 << (d1 + 256)
                        new_diffs |= 1 << (d2 + 256)
                    
                    if new_sums & sum_mask_B_no_b:
                        continue
                    if new_diffs & diff_mask_B_no_b:
                        continue
                    if new_diffs & diff_mask_A:
                        continue
                    
                    B[i] = x
                    sum_mask_B = sum_mask_B_no_b | new_sums
                    diff_mask_B = diff_mask_B_no_b | new_diffs
                    improved = True
                    break
                if improved:
                    break
        
        return A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B

    start_time = time.time()
    max_time = 55.0
    
    candidates = list(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        random.shuffle(candidates)
        
        A = []
        B = []
        sum_mask_A = 0
        diff_mask_A = 0
        sum_mask_B = 0
        diff_mask_B = 0
        
        for x in candidates:
            res_A = try_add(A, sum_mask_A, diff_mask_A, diff_mask_B, x)
            if res_A:
                A, sum_mask_A, diff_mask_A = res_A
            
            res_B = try_add(B, sum_mask_B, diff_mask_B, diff_mask_A, x)
            if res_B:
                B, sum_mask_B, diff_mask_B = res_B
        
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time > 1.0:
            A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B = local_search(
                A, B, sum_mask_A, diff_mask_A, sum_mask_B, diff_mask_B,
                start_time + max_time
            )
        
        total_size = len(A) + len(B)
        if total_size > best_size:
            best_size = total_size
            best_sets = (list(A), list(B))
            
    return [best_sets[0], best_sets[1]]
