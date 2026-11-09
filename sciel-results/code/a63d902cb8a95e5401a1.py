<<<<<<< SEARCH
    # ---- Optimized Simulated Annealing ----
    def local_search(start, rng, max_time=8.0, T0=5.0):
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        t_start = time.time()

        def try_rebuild_row(other_mask, limit, rng):
            """Rebuild a single row avoiding other_mask, with scope limit."""
            def _bt(cur_row, cur_mask, depth):
                if depth == k:
                    return cur_row[:]
                last = cur_row[-1]
                cands = []
                for x in range(last + 1, limit + 1):
                    nm = 0
                    ok = True
                    for a in cur_row:
                        d = x - a
                        bit = 1 << d
                        if (other_mask & bit) or (cur_mask & bit):
                            ok = False
                            break
                        nm |= bit
                    if ok:
                        cands.append((x, nm))
                if not cands:
                    return None
                rng.shuffle(cands)
                for x, nm in cands:
                    res = _bt(cur_row + [x], cur_mask | nm, depth + 1)
                    if res:
                        return res
                return None
            return _bt([0], 0, 0)

        while time.time() - t_start < max_time and remaining() > 5:
            # Select row: bias towards rows with large max
            if rng.random() < 0.6:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)
            
            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])
            
            move_type = rng.random()
            new_row = None
            new_mask = None
            
            if move_type < 0.4:
                # MOVE 1: Targeted Compression of the Max Element
                p = len(row) - 1
                current_max = row[p]
                prev_val = row[p - 1]
                
                for v in range(current_max - 1, prev_val, -1):
                    test_row = row[:]
                    test_row[p] = v
                    nd_test = get_diff_mask(test_row)
                    if nd_test & other_used_mask:
                        continue
                    if bin(nd_test).count("1") != needed:
                        continue
                    new_row = test_row
                    new_mask = nd_test
                    break

            elif move_type < 0.75:
                # MOVE 2: Random Interior Mutation
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 15)
                
                if lo > hi:
                    continue
                
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.3))
                
                if v == row[p]:
                    continue
                
                test_row = row[:]
                test_row[p] = v
                
                if (p > 0 and test_row[p-1] >= v) or (p < len(test_row)-1 and v >= test_row[p+1]):
                    continue
                    
                nd_test = get_diff_mask(test_row)
                if bin(nd_test).count("1") != needed:
                    continue
                if nd_test & other_used_mask:
                    continue
                
                new_row = test_row
                new_mask = nd_test

            else:
                # MOVE 3: Row Reconstruction
                new_row = try_rebuild_row(other_used_mask, cur_scope, rng)
                if new_row is None:
                    new_row = try_rebuild_row(other_used_mask, cur_scope + 8, rng)
                if new_row is None:
                    continue
                new_mask = get_diff_mask(new_row)
                if bin(new_mask).count("1") != needed or (new_mask & other_used_mask):
                    continue

            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = other_used_mask | new_mask
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            
            T = max(0.05, T * 0.999)

        return best
=======
    # ---- Optimized Simulated Annealing ----
    def local_search(start, rng, max_time=8.0, T0=5.0):
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        t_start = time.time()
        total_time = max_time
        elapsed_at_start = time.time() - t0

        def try_rebuild_row(other_mask, limit, rng):
            """Rebuild a single row avoiding other_mask, with scope limit."""
            def _bt(cur_row, cur_mask, depth):
                if depth == k:
                    return cur_row[:]
                last = cur_row[-1]
                remaining_pos = k - depth - 1
                cands = []
                for x in range(last + 1, limit + 1):
                    nm = 0
                    ok = True
                    for a in cur_row:
                        d = x - a
                        bit = 1 << d
                        if (other_mask & bit) or (cur_mask & bit):
                            ok = False
                            break
                        nm |= bit
                    if ok:
                        if x <= limit - remaining_pos:
                            cands.append((x, nm))
                if not cands:
                    return None
                rng.shuffle(cands)
                for x, nm in cands:
                    res = _bt(cur_row + [x], cur_mask | nm, depth + 1)
                    if res:
                        return res
                return None
            return _bt([0], 0, 0)

        def try_compress_row(row, other_mask, rng):
            """Try to compress a row by reducing its max element."""
            p = len(row) - 1
            current_max = row[p]
            prev_val = row[p - 1]
            
            for v in range(current_max - 1, prev_val, -1):
                test_row = row[:]
                test_row[p] = v
                nd_test = get_diff_mask(test_row)
                if nd_test & other_mask:
                    continue
                if mask_popcount(nd_test) != needed:
                    continue
                return test_row, nd_test
            return None, None

        def try_swap_compression(row, other_mask, rng):
            """Try to swap two interior elements to enable compression."""
            if len(row) < 4:
                return None, None
            # Try swapping pairs of interior elements
            for i in range(1, len(row) - 1):
                for j in range(i + 1, len(row) - 1):
                    if i == j:
                        continue
                    test_row = row[:]
                    test_row[i], test_row[j] = test_row[j], test_row[i]
                    # Check strictly increasing
                    if any(test_row[m] >= test_row[m+1] for m in range(len(test_row) - 1)):
                        continue
                    nd_test = get_diff_mask(test_row)
                    if nd_test & other_mask:
                        continue
                    if mask_popcount(nd_test) != needed:
                        continue
                    # Now try to compress the max
                    result, mask = try_compress_row(test_row, other_mask, rng)
                    if result is not None:
                        return result, mask
            return None, None

        while time.time() - t_start < max_time and remaining() > 5:
            # Select row: bias towards rows with large max
            if rng.random() < 0.6:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)
            
            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])
            
            move_type = rng.random()
            new_row = None
            new_mask = None
            
            if move_type < 0.35:
                # MOVE 1: Targeted Compression of the Max Element
                new_row, new_mask = try_compress_row(row, other_used_mask, rng)

            elif move_type < 0.55:
                # MOVE 2: Random Interior Mutation
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 15)
                
                if lo > hi:
                    continue
                
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.3))
                
                if v == row[p]:
                    continue
                
                test_row = row[:]
                test_row[p] = v
                
                if (p > 0 and test_row[p-1] >= v) or (p < len(test_row)-1 and v >= test_row[p+1]):
                    continue
                    
                nd_test = get_diff_mask(test_row)
                if mask_popcount(nd_test) != needed:
                    continue
                if nd_test & other_used_mask:
                    continue
                
                new_row = test_row
                new_mask = nd_test

            elif move_type < 0.75:
                # MOVE 3: Swap + Compression
                new_row, new_mask = try_swap_compression(row, other_used_mask, rng)

            else:
                # MOVE 4: Row Reconstruction
                new_row = try_rebuild_row(other_used_mask, cur_scope, rng)
                if new_row is None:
                    new_row = try_rebuild_row(other_used_mask, cur_scope + 8, rng)
                if new_row is None:
                    continue
                new_mask = get_diff_mask(new_row)
                if mask_popcount(new_mask) != needed or (new_mask & other_used_mask):
                    continue

            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum_row = sum(new_row)
            
            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)
            
            new_sum = cur_sum - sum(row) + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj
            
            # Faster cooling
            progress = (time.time() - t_start) / total_time
            T_current = max(0.01, T0 * (1 - progress) ** 1.5)
            
            if d <= 0 or (T_current > 0.01 and rng.random() < math.exp(-d / T_current)):
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = other_used_mask | new_mask
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]

        return best
>>>>>>> REPLACE

