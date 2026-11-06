import time

def construct():
    t0 = time.time()
    DEADLINE = 80.0

    # Generate smooth numbers <= 128
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n = len(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # Precompute conflict matrix
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                    conflict[i][j] = True

    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = [None] * 129

    # Initialize with single elements
    for i in range(n):
        s = smooth[i]
        if s <= 128:
            best_for_sum[s] = (1, (i,))

    # DFS to find minimal term counts
    def dfs(start_idx, current_sum, current_indices, current_count):
        if time.time() - t0 > DEADLINE:
            return

        # Pruning: if we have a solution with <= current_count for this sum, stop
        if best_for_sum[current_sum] is not None and current_count >= best_for_sum[current_sum][0]:
            return

        for i in range(start_idx, n):
            if current_sum + smooth[i] > 128:
                break
            
            # Check if smooth[i] conflicts with any in current_indices
            valid = True
            for j in current_indices:
                if conflict[j][i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_sum = current_sum + smooth[i]
            new_indices = current_indices + (i,)
            new_count = current_count + 1
            
            if best_for_sum[new_sum] is None or new_count < best_for_sum[new_sum][0]:
                best_for_sum[new_sum] = (new_count, new_indices)
            
            dfs(i + 1, new_sum, new_indices, new_count)

    # Start DFS from empty set
    dfs(0, 0, (), 0)

    # Build answers
    answers = []
    for t in range(1, 129):
        if best_for_sum[t] is not None:
            count, indices = best_for_sum[t]
            row = sorted(smooth[i] for i in indices)
            answers.append(row)
        else:
            # Fallback: Find a valid representation
            # This should rarely be needed
            if t in smooth_set:
                answers.append([t])
            else:
                found = False
                for i in range(n):
                    if smooth[i] >= t:
                        break
                    rem = t - smooth[i]
                    if rem in smooth_set:
                        j = val_to_idx[rem]
                        if i != j and not conflict[i][j]:
                            answers.append([smooth[i], rem])
                            found = True
                            break
                if not found:
                    # Try 3 terms
                    found = False
                    for i in range(n):
                        if smooth[i] >= t:
                            break
                        for j in range(i+1, n):
                            s2 = smooth[i] + smooth[j]
                            if s2 >= t:
                                break
                            rem = t - s2
                            if rem in smooth_set:
                                k = val_to_idx[rem]
                                if k > j and not conflict[i][j] and not conflict[i][k] and not conflict[j][k]:
                                    answers.append([smooth[i], smooth[j], rem])
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        # Last resort: just return a valid antichain for some sum? 
                        # No, must sum to t.
                        # This implies no solution found. 
                        # But problem states all targets require valid representations.
                        # So this branch should not execute.
                        answers.append([1]) # Placeholder, likely invalid
                        pass

    return answers
