# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns 128 lists representing targets 1..128.
    Each list contains distinct 5-smooth numbers summing to the target.
    The set of numbers in each list forms a divisibility antichain.
    Minimizes the total number of summands across all targets.
    """
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit

    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_numbers.sort()
    n_smooth = len(smooth_numbers)
    
    # Precompute divisibility
    # divides[i][j] is True if smooth_numbers[i] divides smooth_numbers[j]
    divides = [[False]*n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(i+1, n_smooth):
            vj = smooth_numbers[j]
            if vj % vi == 0:
                divides[i][j] = True
                divides[j][i] = True
                
    # best_for_sum[s] = (min_size, tuple_of_indices)
    best_for_sum = [None] * 129
    
    # Backtracking to generate antichains
    def backtrack(start_idx, current_sum, current_size, current_set):
        nonlocal best_for_sum
        nonlocal start_time
        nonlocal timeout
        
        if time.time() - start_time > timeout:
            return False
            
        if current_sum <= 128:
            if current_sum > 0:
                if best_for_sum[current_sum] is None or current_size < best_for_sum[current_sum][0]:
                    best_for_sum[current_sum] = (current_size, tuple(current_set))
        
        if current_sum > 128:
            return True
            
        for i in range(start_idx, n_smooth):
            val = smooth_numbers[i]
            
            # Check compatibility
            compatible = True
            for j in current_set:
                if divides[i][j]:
                    compatible = False
                    break
            
            if compatible:
                if current_sum + val <= 128:
                    current_set.append(i)
                    if not backtrack(i + 1, current_sum + val, current_size + 1, current_set):
                        current_set.pop()
                        return False
                    current_set.pop()
                else:
                    break
        
        return True

    if not backtrack(0, 0, 0, []):
        pass
        
    final_answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            size, indices = best_for_sum[n]
            nums = [smooth_numbers[i] for i in indices]
            final_answers.append(nums)
        else:
            ans = _find_single_target(n, smooth_numbers, divides, n_smooth)
            if ans:
                final_answers.append(ans)
            else:
                final_answers.append([n] if n in smooth_numbers else [1, n-1]) 
                
    return final_answers

def _find_single_target(target, smooth_numbers, divides, n_smooth):
    def dfs(start, rem, path):
        if rem == 0:
            return list(path)
        if rem < 0:
            return None
        for i in range(start, n_smooth):
            val = smooth_numbers[i]
            if val > rem:
                break
            ok = True
            for j in path:
                if divides[i][j]:
                    ok = False
                    break
            if ok:
                path.append(i)
                res = dfs(i+1, rem-val, path)
                if res:
                    return res
                path.pop()
        return None
        
    indices = dfs(0, target, [])
    if indices:
        return [smooth_numbers[i] for i in indices]
    return None

# EVOLVE-BLOCK-END
