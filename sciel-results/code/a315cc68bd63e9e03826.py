            def dfs(start_idx, current_sum, current_size, mask):
                if current_size == k:
                    return current_sum == n
                if current_sum > n:
                    return None
                if current_size + (N - start_idx) < k:
                    return None
                if current_sum + min_sum_from(start_idx, k - current_size) > n:
                    return None
                    
                for i in range(start_idx, N):
                    if mask & conflict[i]:
                        continue
                    new_sum = current_sum + smooth[i]
                    if new_sum > n:
                        break
                    new_mask = mask | (1 << i)
                    res = dfs(i + 1, new_sum, current_size + 1, new_mask)
                    if res is not None:
                        return res
                return None
