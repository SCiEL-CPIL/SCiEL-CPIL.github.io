import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    n = 256
    
    # 1. Greedy Initialization
    def greedy():
        selected = []
        in_set = [False] * (n + 1)
        for x in range(1, n + 1):
            can_add = True
            for a in selected:
                b1 = 2 * a - x
                if b1 > 0 and b1 <= n and in_set[b1]:
                    can_add = False
                    break
                b2 = 2 * x - a
                if b2 > 0 and b2 <= n and in_set[b2]:
                    can_add = False
                    break
                if (a + x) % 2 == 0:
                    b3 = (a + x) // 2
                    if b3 > 0 and b3 <= n and in_set[b3]:
                        can_add = False
                        break
            if can_add:
                selected.append(x)
                in_set[x] = True
        return selected, sum(1.0 / x for x in selected)

    best_set, best_sum = greedy()
    
    # 2. Backtracking Improvement
    # Precompute suffix sums of reciprocals for pruning
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i+1] + 1.0/i
        
    in_set = [False] * (n + 1)
    for x in best_set:
        in_set[x] = True
    current_list = list(best_set)
    
    # We need to backtrack from a state.
    # Since we start with a full greedy set, we can try to "improve" it by swapping or adding.
    # But standard backtracking builds from scratch.
    # Let's restart backtracking from scratch but with a good initial best_sum.
    # This is safer.
    
    in_set = [False] * (n + 1)
    current_list = []
    
    def backtrack(i, current_sum, in_set, current_list):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if i > n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_list)
            return
            
        # Pruning
        if current_sum + suffix_sum[i] <= best_sum:
            return
            
        # Option 1: Exclude i
        backtrack(i + 1, current_sum, in_set, current_list)
        
        # Option 2: Include i
        x = i
        can_add = True
        for a in current_list:
            b1 = 2 * a - x
            if b1 > 0 and b1 <= n and in_set[b1]:
                can_add = False
                break
            b2 = 2 * x - a
            if b2 > 0 and b2 <= n and in_set[b2]:
                can_add = False
                break
            if (a + x) % 2 == 0:
                b3 = (a + x) // 2
                if b3 > 0 and b3 <= n and in_set[b3]:
                    can_add = False
                    break
            
        if can_add:
            in_set[x] = True
            current_list.append(x)
            backtrack(i + 1, current_sum + 1.0 / x, in_set, current_list)
            current_list.pop()
            in_set[x] = False
            
    backtrack(1, 0.0, in_set, current_list)
    
    return best_set
