<<<<<<< SEARCH
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
                
        else:
            # Single-element move with compression bias
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            # Bias toward compression (lower values)
            if rng.random() < 0.6:
                span = old_val - lo
                if span <= 0:
                    continue
                new_val = lo + int(rng.random() * span)
            else:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
            current_rows[i][j] = new_val
            conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
            
            new_scope = _scope(current_rows)
            new_sum = _sum_entries(current_rows)
            new_cost = cost(new_scope, conflict_count, new_sum)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
=======
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
                current_scope = _scope(current_rows)
                current_sum = new_sum - (new_val - old_val)

        else:
            # Single-element move with compression bias
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            
            old_val = current_rows[i][j]
            prev_val = current_rows[i][j-1]
            next_val = current_rows[i][j+1] if j < k else current_scope + 10
            
            lo = prev_val + 1
            hi = next_val - 1
            
            if lo > hi:
                continue
            
            # Bias toward compression (lower values)
            if rng.random() < 0.6:
                span = old_val - lo
                if span <= 0:
                    continue
                new_val = lo + int(rng.random() * span)
            else:
                span = hi - lo
                delta = int(rng.gauss(0, max(1, span * 0.25)))
                new_val = old_val + delta
            
            if new_val == old_val or new_val < lo or new_val > hi:
                continue
            
            conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
            current_rows[i][j] = new_val
            conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
            
            # Optimized scope/sum update
            new_sum = current_sum - old_val + new_val
            if new_val > current_scope:
                new_scope = new_val
            else:
                if current_scope == old_val:
                    new_scope = _scope(current_rows)
                else:
                    new_scope = current_scope
            new_cost = cost(new_scope, conflict_count, new_sum)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                current_scope = new_scope
                current_sum = new_sum
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = new_scope
                    best_conflicts = conflict_count
                    best_sum = new_sum
                    best_rows = [list(r) for r in current_rows]
            else:
                conflict_count = _remove_row(current_rows[i], diff_counts, conflict_count)
                current_rows[i][j] = old_val
                conflict_count = _add_row(current_rows[i], diff_counts, conflict_count)
                current_scope = _scope(current_rows)
                current_sum = new_sum - (new_val - old_val)
>>>>>>> REPLACE

