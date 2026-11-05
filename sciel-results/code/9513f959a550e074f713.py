<<<<<<< SEARCH
    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = set()  # Use a set instead of bitmask for correctness
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = target_scope - 1
            
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                conflict = False
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Heuristic cost: prefer smaller values, penalize large gaps
                cost = 0.0
                for a in current_row:
                    d = x - a
                    cost += 1.0 / d
                
                # Add small noise for diversity
                noise = rng.random() * 0.01
                candidates.append((cost + noise, x, new_diffs))
                
                if len(candidates) > 10:
                    break
            
            if not candidates:
                return False
            
            candidates.sort(key=lambda item: item[0])
            
            # Try top candidates first, then a few random ones
            top_n = min(3, len(candidates))
            top_candidates = candidates[:top_n]
            remaining = candidates[top_n:]
            
            # Shuffle remaining for diversity
            if remaining:
                rng.shuffle(remaining)
                # Add 2 random candidates from remaining
                extra = remaining[:2]
            else:
                extra = []
            
            for cost, x, new_diffs in top_candidates + extra:
                old_used = used_diffs.copy()
                used_diffs |= new_diffs
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0])
=======
    def solve_attempt(target_scope, rng_seed):
        nonlocal best_result, best_scope
        
        rng = random.Random(rng_seed)
        used_diffs = set()
        rows = []
        
        def _backtrack(row_idx, current_row):
            nonlocal used_diffs, rows, best_scope, best_result
            
            if time_up():
                return False
                
            if row_idx == n_rows:
                actual_scope = max(r[-1] for r in rows)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_result = [r[:] for r in rows]
                return True
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                if _backtrack(row_idx + 1, [0]):
                    return True
                rows.pop()
                return False
            
            last_val = current_row[-1]
            upper_bound = target_scope
            
            # Collect candidates efficiently
            candidates = []
            
            for x in range(last_val + 1, upper_bound + 1):
                conflict = False
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Heuristic: prefer smaller values
                candidates.append((x, new_diffs))
                
                if len(candidates) >= 8:
                    break
            
            if not candidates:
                return False
            
            # Try candidates in order (already sorted by value)
            for x, new_diffs in candidates:
                old_used = used_diffs.copy()
                used_diffs |= new_diffs
                
                if _backtrack(row_idx, current_row + [x]):
                    return True
                
                used_diffs = old_used
            
            return False
        
        return _backtrack(0, [0])
>>>>>>> REPLACE

