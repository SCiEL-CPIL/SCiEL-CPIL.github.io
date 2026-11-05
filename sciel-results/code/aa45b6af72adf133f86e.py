        # Heuristic ordering
        if curr_sum > 0:
            order = [-1, 0, 1]
        elif curr_sum < 0:
            order = [1, 0, -1]
        else:
            order = [0, -1, 1]
            
        for val in order:
             if time.time() > deadline: return
             coeffs[idx] = val
             new_sum = curr_sum + val * inv[idx]
             dfs(idx + 1, new_sum, coeffs)
