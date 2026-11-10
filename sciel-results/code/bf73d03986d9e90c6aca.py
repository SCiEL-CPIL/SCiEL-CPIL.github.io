def construct():
    import time
    from math import gcd
    
    deadline = time.time() + 85  # 85s out of 90s budget
    
    N = 256
    # Numbers are 2..256, index 0..253
    nums = list(range(2, 257))
    n = len(nums)  # 255
    
    # Precompute conflict: conflict[i][j] = True if lcm(nums[i], nums[j]) <= 256
    # lcm(a,b) = a*b//gcd(a,b)
    conflict = [bytearray(n) for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i+1, n):
            b = nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i][j] = 1
                conflict[j][i] = 1
    
    # Weight: 1/nums[i]
    weights = [1.0 / nums[i] for i in range(n)]
    
    def total_weight(selected):
        return sum(weights[i] for i in selected)
    
    def is_valid_add(selected_set, idx):
        """Check if idx can be added to selected_set"""
        for j in selected_set:
            if conflict[idx][j]:
                return False
        return True
    
    # Greedy initialization: try multiple orderings
    best_set = set()
    best_score = 0.0
    
    # Ordering 1: by weight descending (smallest nums first)
    order1 = sorted(range(n), key=lambda i: -weights[i])
    # Ordering 2: by number of conflicts ascending (least constrained first)
    conflict_counts = [sum(conflict[i]) for i in range(n)]
    order2 = sorted(range(n), key=lambda i: (conflict_counts[i], -weights[i]))
    # Ordering 3: by weight/conflict_count ratio
    order3 = sorted(range(n), key=lambda i: -(weights[i] / (conflict_counts[i] + 1)))
    
    for order in [order1, order2, order3]:
        selected = set()
        for i in order:
            if is_valid_add(selected, i):
                selected.add(i)
        score = total_weight(selected)
        if score > best_score:
            best_score = score
            best_set = set(selected)
    
    # Local search: try swaps
    selected = set(best_set)
    current_score = best_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single swaps: remove i from selected, add j not in selected
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                # Check if j can replace i
                # Remove i, then check if j conflicts with remaining
                can_add = True
                for k in selected:
                    if k != i and conflict[j][k]:
                        can_add = False
                        break
                if can_add:
                    new_score = current_score - weights[i] + weights[j]
                    if new_score > current_score + 1e-15:
                        selected.remove(i)
                        selected.add(j)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Try adding elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for j in range(n):
            if j in selected:
                continue
            if is_valid_add(selected, j):
                selected.add(j)
                current_score += weights[j]
                improved = True
                break
    
    # Try removing elements that allow adding better ones (2-swap)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in list(selected):
            for j in range(n):
                if j in selected:
                    continue
                # Remove i, see if j can be added
                can_add = True
                for k in selected:
                    if k != i and conflict[j][k]:
                        can_add = False
                        break
                if can_add:
                    new_score = current_score - weights[i] + weights[j]
                    if new_score > current_score + 1e-15:
                        selected.remove(i)
                        selected.add(j)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    return [nums[i] for i in selected]
