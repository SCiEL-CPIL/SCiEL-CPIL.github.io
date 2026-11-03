    node_count = 0
    last_time_check = start_time

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask, start_time, time_limit, node_count, last_time_check
        
        node_count += 1
        if node_count % 1000 == 0:
            if time.time() - last_time_check > 5.0:
                last_time_check = time.time()
                if time.time() - start_time > time_limit:
                    return
