# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    
    # Precompute divisibility relations for pairs in smooth
    # divs[a][b] = True if a divides b (a < b)
    n_smooth = len(smooth)
    # Use a 2D list for quick lookup
    # divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            if sj % si == 0:
                divs[i][j] = True
    
    def is_antichain_indices(indices):
        """Check if the set of indices forms an antichain."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if divs[indices[i]][indices[j]]:
                    return False
        return True
    
    def find_rep(n):
        # k=1
        if n in smooth_set:
            return [n]
        
        # k=2
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                # Find index of b
                # Since smooth is sorted, we can binary search or just check
                # For speed, let's use a dict
                if b % a != 0:
                    return [a, b]
        
        # k=3
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    # a < b < c
                    if b % a != 0 and c % a != 0 and c % b != 0:
                        return [a, b, c]
        
        # k=4
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d in smooth_set:
                        # a < b < c < d
                        if is_antichain([a, b, c, d]):
                            return [a, b, c, d]
        
        # k=5
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    for l in range(k + 1, n_smooth):
                        d = smooth[l]
                        if a + b + c + d >= n:
                            break
                        e = n - a - b - c - d
                        if e <= d:
                            continue
                        if e in smooth_set:
                            # a < b < c < d < e
                            if is_antichain([a, b, c, d, e]):
                                return [a, b, c, d, e]
        
        # Generic backtracking for k > 5
        # This should rarely be needed for n <= 128
        def backtrack(target, start_idx, current_sum, current_list, current_indices):
            if current_sum == target:
                return current_list[:]
            if current_sum > target:
                return None
            
            # Pruning: if we have enough terms or start_idx out of bounds
            # Also, we can prune if even taking the largest possible remaining terms
            # cannot reach the target? No, we are summing up to target.
            # We can prune if the minimum sum of remaining terms (smallest ones)
            # plus current_sum > target? No.
            
            # Heuristic: if current_sum + sum of largest possible terms < target?
            # We just iterate.
            
            for idx in range(start_idx, n_smooth):
                term = smooth[idx]
                if current_sum + term > target:
                    break
                
                # Check antichain property with new term
                valid = True
                for existing_idx in current_indices:
                    if divs[existing_idx][idx]:
                        valid = False
                        break
                    # Note: divs[idx][existing_idx] is always False because idx > existing_idx
                    # and divs is only populated for i<j.
                    # Wait, divs[i][j] is True if smooth[i] divides smooth[j].
                    # If existing_idx < idx, we check divs[existing_idx][idx].
                    # If smooth[existing_idx] divides smooth[idx], it's invalid.
                
                if valid:
                    current_list.append(term)
                    current_indices.append(idx)
                    res = backtrack(target, idx + 1, current_sum + term, current_list, current_indices)
                    if res:
                        return res
                    current_list.pop()
                    current_indices.pop()
            return None

        res = backtrack(n, 0, 0, [], [])
        if res:
            return res
        
        # Absolute fallback: should not happen
        return [1]

    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 5:
            # If time is tight, we might need to break, but n<=128 is small.
            # The backtracking is the bottleneck.
            pass
        
        found = find_rep(n)
        if found is None:
            # This should not happen, but handle it
            found = [n] if n in smooth_set else [1]
        
        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
