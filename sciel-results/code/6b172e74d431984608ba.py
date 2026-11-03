import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    k_plus_1 = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    lower_bound = total_diffs_needed - 1

    def greedy_construct(n_rows, k, max_scope=200):
        """Fast greedy baseline."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                if x > max_scope:
                    return None
                conflict = False
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        conflict = True
                        break
                if not conflict:
                    for a in row:
                        used_mask |= (1 << (x - a))
                    row.append(x)
            rows.append(row)
        return rows

    def solve_csp(n_rows, k, max_scope, deadline):
        """
        Backtracking solver to find a solution with max entry <= max_scope.
        Returns list of rows or None.
        """
        rows = []
        used_mask = 0
        
        def build_row(pos, current_last, row_so_far, mask, remaining_rows):
            if time.time() > deadline:
                return None

            if pos == k_plus_1:
                # Row complete
                if remaining_rows == 0:
                    return [list(r) for r in rows] + [list(row_so_far)]
                rows.append(list(row_so_far))
                result = build_full_rows(remaining_rows - 1, mask, deadline)
                if result is not None:
                    rows.pop()
                    return result
                rows.pop()
                return None

            # Heuristic: We need to place (k_plus_1 - pos) more elements.
            # The smallest possible next element is current_last + 1.
            # The largest possible next element is constrained by max_scope and the need to fit remaining elements.
            # If we pick x, the next element must be > x.
            # So if we pick x, we need space for (k_plus_1 - pos - 1) more elements after x.
            # Min sum of increments is 1 * (k_plus_1 - pos - 1).
            # So x + (k_plus_1 - pos - 1) <= max_scope.
            max_candidate = max_scope - (k_plus_1 - pos - 1)
            
            if max_candidate <= current_last:
                return None

            # Generate candidates
            # We try smaller candidates first to keep the row compact.
            # Optimization: Skip candidates that create "gaps" that are hard to fill?
            # For now, linear scan is fast enough for small k.
            
            for x in range(current_last + 1, max_candidate + 1):
                # Check conflicts
                new_bits = 0
                valid = True
                for a in row_so_far:
                    d = x - a
                    if (mask >> d) & 1:
                        valid = False
                        break
                    new_bits |= (1 << d)
                
                if not valid:
                    continue
                
                # Forward checking: 
                # After placing x, we have used mask | new_bits.
                # We need to place (k_plus_1 - pos - 1) more elements.
                # Simple check: Is there enough "room" in the difference set?
                # This is hard to check exactly. We rely on backtracking.
                
                row_so_far.append(x)
                result = build_row(pos + 1, x, row_so_far, mask | new_bits, remaining_rows)
                if result is not None:
                    row_so_far.pop()
                    return result
                row_so_far.pop()
                
            return None

        def build_full_rows(remaining, mask, deadline):
            if remaining == 0:
                return [list(r) for r in rows]
            row_so_far = [0]
            return build_row(1, 0, row_so_far, mask, remaining)

        return build_full_rows(n_rows, 0, deadline)

    # Phase 1: Get a valid initial solution
    # Try greedy with a reasonable upper bound
    # Lower bound is total_diffs - 1. 
    # Heuristic upper bound: greedy often works with slightly larger scope.
    initial_scope = lower_bound + 20 
    best_rows = greedy_construct(n_rows, k, initial_scope)
    
    if best_rows is None:
        # Fallback: try larger scope
        for scope in range(initial_scope + 10, initial_scope + 100, 10):
            best_rows = greedy_construct(n_rows, k, scope)
            if best_rows:
                break
        
    if best_rows is None:
        # Last resort: very large scope
        best_rows = greedy_construct(n_rows, k, 500)

    current_max = max(max(r) for r in best_rows)
    best_max = current_max

    # Phase 2: Iterative Deepening to minimize scope
    # Try to find a solution with max scope = current_max - 1, then -2, etc.
    # We stop if we hit the lower bound or time limit.
    
    # To handle local minima in the backtracker, we might need to randomize the order of rows or candidates.
    # But for (4,7), deterministic search is often strong.
    
    target_scope = current_max - 1
    while target_scope > lower_bound and time.time() < deadline - 5.0:
        found = False
        # Try a few random restarts of the CSP solver with different row orderings?
        # Or just standard backtracking.
        # Standard backtracking can get stuck if the first row is "bad".
        # Let's try to solve with the current best rows as a hint? No, CSP is from scratch.
        
        # To avoid getting stuck in one bad path, we can limit the depth of search or use randomization in candidate order.
        # Let's try a few runs.
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Simple randomization: Shuffle the order of rows to try?
            # The CSP solver builds rows 0..n_rows-1.
            # We can permute the "slots" but the rows are indistinguishable in the final set.
            # However, the *order* in which we build them affects the search tree.
            
            # Let's just run the CSP solver. If it's slow, we need better pruning.
            # For now, let's assume it's fast enough for small n, k.
            
            # To add variety, we can slightly modify the candidate ordering.
            # But `solve_csp` is deterministic.
            # Let's create a randomized version or just call it.
            # If it fails, it failed.
            
            # Actually, let's implement a randomized backtracking for the search phase.
            # We'll modify the inner loop to sometimes pick a random valid candidate from the first few.
            
            pass # Placeholder for logic below

        # If we didn't find it, decrement target
        if not found:
            target_scope -= 1
            if target_scope <= lower_bound:
                break

    # The above loop was a placeholder. Let's implement a proper search loop.
    
    # Re-implementation of the search loop with a better solver
    def solve_csp_randomized(n_rows, k, max_scope, deadline, rng):
        rows = []
        used_mask = 0
        
        def build_row(pos, current_last, row_so_far, mask, remaining_rows):
            if time.time() > deadline:
                return None

            if pos == k_plus_1:
                if remaining_rows == 0:
                    return [list(r) for r in rows] + [list(row_so_far)]
                rows.append(list(row_so_far))
                result = build_full_rows(remaining_rows - 1, mask, deadline)
                if result is not None:
                    rows.pop()
                    return result
                rows.pop()
                return None

            max_candidate = max_scope - (k_plus_1 - pos - 1)
            if max_candidate <= current_last:
                return None

            candidates = []
            for x in range(current_last + 1, max_candidate + 1):
                new_bits = 0
                valid = True
                for a in row_so_far:
                    d = x - a
                    if (mask >> d) & 1:
                        valid = False
                        break
                    new_bits |= (1 << d)
                if valid:
                    candidates.append((x, new_bits))
                    if len(candidates) > 20: # Limit candidates for speed
                        break
            
            if not candidates:
                return None

            # Randomize order slightly
            # Try the first few (smallest) more often
            if len(candidates) > 1 and rng.random() < 0.5:
                rng.shuffle(candidates[:min(5, len(candidates))])
            
            for x, new_bits in candidates:
                row_so_far.append(x)
                result = build_row(pos + 1, x, row_so_far, mask | new_bits, remaining_rows)
                if result is not None:
                    row_so_far.pop()
                    return result
                row_so_far.pop()
            return None

        def build_full_rows(remaining, mask, deadline):
            if remaining == 0:
                return [list(r) for r in rows]
            row_so_far = [0]
            return build_row(1, 0, row_so_far, mask, remaining)

        return build_full_rows(n_rows, 0, deadline)

    # Actual Search Loop
    current_best_max = max(max(r) for r in best_rows)
    target_scope = current_best_max - 1
    rng = random.Random(42)
    
    while target_scope > lower_bound and time.time() < deadline - 10.0:
        found = False
        # Try multiple random restarts for this scope
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            sol = solve_csp_randomized(n_rows, k, target_scope, deadline, rng)
            if sol is not None:
                # Verify
                all_diffs = set()
                valid = True
                for row in sol:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    best_rows = sol
                    current_best_max = target_scope
                    found = True
                    break
        
        if found:
            target_scope = current_best_max - 1
        else:
            target_scope -= 1

    return best_rows
