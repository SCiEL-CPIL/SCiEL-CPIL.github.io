    call_count = 0
    def dfs(idx, curr_s, curr_m):
        nonlocal best_sum, best_mask, call_count
        call_count += 1
        if call_count % 1000 == 0 and time.time() > DEADLINE:
            return
        # ... rest of logic
