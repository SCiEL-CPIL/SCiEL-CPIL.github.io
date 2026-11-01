import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 138  # 12s margin before 150s kill
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        used = set()
        for row in rows:
            d = get_row_diffs(row)
            if len(d) != len(row) * (len(row) - 1) // 2:
                return False
            if d & used:
                return False
            used |= d
        return True
    
    def greedy_init(rng, n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                x = row[-1] + 1
                while x < 300:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows, used
    
    def random_greedy(rng, n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                base = row[-1]
                candidates = []
                for x in range(base + 1, min(base + 15, 250)):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                if not candidates:
                    x = base
                    while x < 250:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                            break
                    if not candidates:
                        return None
                pick = rng.choice(candidates[:min(3, len(candidates))])
                row.append(pick)
                used |= {pick - a for a in row[:-1]}
            rows.append(row)
        return rows, used
    
    def backtrack_solve(n_rows, k, deadline, rng):
        """Backtracking with forward checking to find valid row configurations."""
        used = set()
        rows = []
        max_val = 120  # upper bound for search
        
        def count_feasible(row, pos, used):
            """Count how many positions are feasible for the next element."""
            if pos >= k:
                return 1
            base = row[-1] if row else 0
            count = 0
            for x in range(base + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    count += 1
                    if count > 5:
                        break
            return count
        
        def bt(row, pos, used):
            if time.time() > deadline:
                return None
            if pos == k + 1:
                return row[:]
            
            base = row[-1] if row else 0
            candidates = []
            for x in range(base + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    candidates.append((x, new_diffs))
            
            # Sort by feasibility (forward checking)
            if candidates:
                scored = []
                for x, nd in candidates[:20]:
                    test_row = row + [x]
                    test_used = used | nd
                    feas = count_feasible(test_row, pos + 1, test_used)
                    scored.append((feas, x, nd))
                scored.sort(key=lambda t: -t[0])
                candidates = [(x, nd) for _, x, nd in scored]
            
            for x, nd in candidates:
                result = bt(row + [x], pos + 1, used | nd)
                if result is not None:
                    return result
            return None
        
        for _ in range(n_rows):
            if time.time() > deadline:
                return None
            row = bt([0], 0, used)
            if row is None:
                return None
            rows.append(row)
            used |= get_row_diffs(row)
        
        return rows
    
    def local_search(rows, used_diffs, n_rows, k, rng, deadline):
        """Local search with incremental diff computation."""
        best_rows = [r[:] for r in rows]
        best_M = max(max(r) for r in rows)
        current_M = best_M
        
        # Precompute per-row diffs for speed
        row_diffs = [get_row_diffs(r) for r in rows]
        
        for it in range(20000):
            if time.time() > deadline:
                break
            
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = rows[ri]
            
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_M + 5
            
            if lo > hi:
                continue
            
            old_val = row[pi]
            old_diffs = row_diffs[ri]
            
            # Generate candidates with bias toward smaller values
            if rng.random() < 0.6:
                hi_try = min(hi, current_M)
            else:
                hi_try = hi
            
            if lo > hi_try:
                continue
            
            new_val = rng.randint(lo, hi_try)
            if new_val == old_val:
                continue
            
            # Check ordering
            if pi < k and new_val >= row[pi + 1]:
                continue
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            
            # Incremental diff computation: only diffs involving position pi change
            # Old diffs involving pi: {row[pi] - row[j] for j < pi} | {row[j] - row[pi] for j > pi}
            # New diffs involving pi: {new_val - row[j] for j < pi} | {row[j] - new_val for j > pi}
            
            old_pi_diffs = set()
            for j in range(pi):
                old_pi_diffs.add(row[pi] - row[j])
            for j in range(pi + 1, len(row)):
                old_pi_diffs.add(row[j] - row[pi])
            
            new_pi_diffs = set()
            for j in range(pi):
                new_pi_diffs.add(new_val - row[j])
            for j in range(pi + 1, len(row)):
                new_pi_diffs.add(row[j] - new_val)
            
            # Check for internal conflicts in the row
            if len(new_pi_diffs) != len(old_pi_diffs):
                # Some diffs collapsed (not strictly increasing or duplicate)
                continue
            
            # Check against other rows' diffs
            added = new_pi_diffs - old_pi_diffs
            other_diffs = used_diffs - old_diffs
            if added & other_diffs:
                continue
            
            # Check new_pi_diffs don't conflict within the row's other diffs
            other_row_diffs = old_diffs - old_pi_diffs
            if new_pi_diffs & other_row_diffs:
                continue
            
            # Valid move
            new_row = row[:]
            new_row[pi] = new_val
            new_M = max(new_row[-1], max(max(r) for i, r in enumerate(rows) if i != ri))
            
            if new_M <= current_M:
                # Update
                used_diffs.difference_update(old_diffs)
                new_diffs = old_diffs - old_pi_diffs | new_pi_diffs
                used_diffs.update(new_diffs)
                row_diffs[ri] = new_diffs
                rows[ri] = new_row
                current_M = new_M
                
                if current_M < best_M:
                    best_M = current_M
                    best_rows = [r[:] for r in rows]
        
        return best_rows, best_M
    
    def simulated_anneal(rows, used_diffs, n_rows, k, rng, deadline):
        """Simulated annealing with incremental diff computation."""
        best_rows = [r[:] for r in rows]
        best_M = max(max(r) for r in rows)
        current_M = best_M
        
        row_diffs = [get_row_diffs(r) for r in rows]
        
        T = 40.0
        cooling = 0.9985
        
        while time.time() < deadline and T > 0.2:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = rows[ri]
            
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_M + 15
            
            if lo > hi:
                continue
            
            old_val = row[pi]
            old_diffs = row_diffs[ri]
            
            # Generate new value
            if rng.random() < 0.7:
                new_val = rng.randint(lo, min(hi, current_M))
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            if pi < k and new_val >= row[pi + 1]:
                continue
            if pi > 0 and new_val <= row[pi - 1]:
                continue
            
            # Incremental diff computation
            old_pi_diffs = set()
            for j in range(pi):
                old_pi_diffs.add(row[pi] - row[j])
            for j in range(pi + 1, len(row)):
                old_pi_diffs.add(row[j] - row[pi])
            
            new_pi_diffs = set()
            for j in range(pi):
                new_pi_diffs.add(new_val - row[j])
            for j in range(pi + 1, len(row)):
                new_pi_diffs.add(row[j] - new_val)
            
            if len(new_pi_diffs) != len(old_pi_diffs):
                continue
            
            added = new_pi_diffs - old_pi_diffs
            other_diffs = used_diffs - old_diffs
            if added & other_diffs:
                continue
            
            other_row_diffs = old_diffs - old_pi_diffs
            if new_pi_diffs & other_row_diffs:
                continue
            
            # Valid move
            new_row = row[:]
            new_row[pi] = new_val
            new_M = max(new_row[-1], max(max(r) for i, r in enumerate(rows) if i != ri))
            delta_E = new_M - current_M
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                used_diffs.difference_update(old_diffs)
                new_diffs = old_diffs - old_pi_diffs | new_pi_diffs
                used_diffs.update(new_diffs)
                row_diffs[ri] = new_diffs
                rows[ri] = new_row
                current_M = new_M
                
                if current_M < best_M:
                    best_M = current_M
                    best_rows = [r[:] for r in rows]
            
            T *= cooling
        
        return best_rows, best_M
    
    rng = random.Random(42)
    
    # Phase 1: Try multiple initializations
    best_initial_rows = None
    best_initial_M = float('inf')
    
    # Try backtracking first (most likely to find tight solutions)
    if time.time() < deadline - 60:
        bt_rows = backtrack_solve(n_rows, k, deadline - 55, rng)
        if bt_rows is not None:
            M = max(max(r) for r in bt_rows)
            if M < best_initial_M:
                best_initial_M = M
                best_initial_rows = [r[:] for r in bt_rows]
    
    # Try greedy initializations
    for attempt in range(8):
        if time.time() > deadline - 40:
            break
        
        rows = None
        if attempt == 0:
            rows, used = greedy_init(rng, n_rows, k)
        else:
            rows, used = random_greedy(rng, n_rows, k)
        
        if rows is None:
            continue
        
        M = max(max(r) for r in rows)
        if M < best_initial_M:
            best_initial_M = M
            best_initial_rows = [r[:] for r in rows]
    
    if best_initial_rows is None:
        rows, used = greedy_init(rng, n_rows, k)
        best_initial_rows = rows
        best_initial_M = max(max(r) for r in rows)
    
    # Phase 2: Local search
    used_diffs = set()
    for r in best_initial_rows:
        used_diffs |= get_row_diffs(r)
    
    best_rows, best_M = local_search(
        [r[:] for r in best_initial_rows],
        used_diffs,
        n_rows, k, rng, deadline - 20
    )
    
    # Rebuild used_diffs
    used_diffs = set()
    for r in best_rows:
        used_diffs |= get_row_diffs(r)
    
    # Phase 3: Simulated annealing
    best_rows, best_M = simulated_anneal(
        [r[:] for r in best_rows],
        used_diffs,
        n_rows, k, rng, deadline - 3
    )
    
    # Final validation
    if not is_valid(best_rows):
        return best_initial_rows
    
    return best_rows
