    # Try a few random greedy inits
    best_init_cost = float('inf')
    best_init_rows = None
    best_init_diffs = None
    best_init_used = None
    
    for _ in range(5):
        used_tmp = set()
        rows_tmp = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_tmp:
                    # Randomly accept with some probability, or just take smallest?
                    # If we take smallest, it's the same every time.
                    # Let's try: with prob 0.5, take x, with prob 0.5, try x+1, etc?
                    # Simpler: just take the smallest. The SA will do the work.
                    row.append(x)
                    used_tmp |= new_diffs
            rows_tmp.append(row)
            
        cost_tmp = sum(max(r) for r in rows_tmp)
        if cost_tmp < best_init_cost:
            best_init_cost = cost_tmp
            best_init_rows = [r[:] for r in rows_tmp]
            # Recompute diffs and used
            used_tmp = set()
            diffs_tmp = []
            for r in rows_tmp:
                d = set()
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d.add(r[j]-r[i])
                diffs_tmp.append(d)
                used_tmp |= d
            best_init_diffs = diffs_tmp
            best_init_used = used_tmp
            
    if best_init_rows:
        rows = best_init_rows
        row_diffs = best_init_diffs
        used = best_init_used
        current_cost = best_init_cost
        best_rows = [r[:] for r in rows]
        best_cost = current_cost
