import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140  # 10s margin from 150s limit
    sys.setrecursionlimit(10000)
    
    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used
    
    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True
    
    def _scope(rows):
        return max(max(r) for r in rows)
    
    def greedy_construct(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows
    
    def greedy_construct_lookahead(seed_offset=0):
        """Greedy with lookahead: prefer choices that leave more room for future elements."""
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try candidates in order, but prefer smaller steps when possible
                # to keep the row compact
                best_x = None
                best_score = float('inf')
                start_x = row[-1] + 1
                max_x = start_x + 200  # look ahead
                
                for x in range(start_x, min(max_x, 500)):
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                    # Score: prefer smaller x, but also consider how many
                    # future differences this would block
                    blocked = len(new_diffs & set(range(1, 300)))
                    score = x + blocked * 0.1
                    if score < best_score:
                        best_score = score
                        best_x = x
                
                if best_x is None:
                    # Fallback: linear scan
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            break
                        x += 1
                    best_x = x
                
                new_diffs = {best_x - a for a in row}
                row.append(best_x)
                used |= new_diffs
            rows.append(row)
        
        return rows
    
    def greedy_construct_randomized(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                max_step = max(1, int((k + 1) * 2.5))
                found = False
                for _attempt in range(30):
                    step = rng.randint(1, max_step)
                    x = row[-1] + step
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        found = True
                        break
                if not found:
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        
        return rows
    
    def backtrack_row(used, max_val, row, depth, rng, deadline, nodes_left):
        """Build one row via backtracking within max_val with node budget."""
        if time.time() > deadline or nodes_left[0] <= 0:
            return None
        if depth == k + 1:
            return row[:]
        
        last = row[-1] if row else 0
        start = last + 1
        
        # Generate candidates, prefer smaller values for tighter packing
        candidates = []
        for x in range(start, max_val + 1):
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            candidates.append((x, new_diffs))
        
        if not candidates:
            return None
        
        # Sort by value (prefer smaller) with some randomization
        candidates.sort(key=lambda t: (t[0], rng.random()))
        
        for x, new_diffs in candidates:
            nodes_left[0] -= 1
            if nodes_left[0] <= 0:
                return None
            
            row.append(x)
            used_copy = used | new_diffs
            
            result = backtrack_row(used_copy, max_val, row, depth + 1, rng, deadline, nodes_left)
            if result is not None:
                return result
            
            row.pop()
        
        return None
    
    def backtrack_construct(max_val, num_rows, rng, deadline, node_budget=50000):
        """Try to construct all rows within max_val using backtracking."""
        used = set()
        rows = []
        nodes_left = [node_budget]
        
        for i in range(num_rows):
            if time.time() > deadline or nodes_left[0] <= 0:
                return None
            
            row = [0]
            result = backtrack_row(used, max_val, row, 1, rng, deadline, nodes_left)
            if result is None:
                return None
            
            rows.append(result)
            used |= _row_diffs(result)
        
        return rows
    
    def backtrack_construct_row_order(max_val, num_rows, rng, deadline, node_budget=30000):
        """Backtrack with row order optimization: try different row orderings."""
        used = set()
        rows = []
        nodes_left = [node_budget]
        
        for i in range(num_rows):
            if time.time() > deadline or nodes_left[0] <= 0:
                return None
            
            row = [0]
            result = backtrack_row(used, max_val, row, 1, rng, deadline, nodes_left)
            if result is None:
                return None
            
            rows.append(result)
            used |= _row_diffs(result)
        
        return rows
    
    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements.
        Works at the difference level: for each element position, try alternative values."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
        used = _all_diffs(rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
            # Try to reduce each element that is at the current maximum
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue
                    
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    
                    if low > high:
                        continue
                    
                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)
                    
                    for new_val in candidates:
                        if time.time() > deadline:
                            break
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        conflict = new_diffs & (used - old_diffs)
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        
        return rows
    
    def local_search_shrink_all(rows, deadline, rng):
        """Extended local search: try to reduce ANY element, not just max."""
        rows = [list(r) for r in rows]
        used = _all_diffs(rows)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    
                    if low > high:
                        continue
                    
                    # Try a few random candidates plus the smallest valid
                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)
                    candidates = candidates[:20]  # limit candidates
                    
                    for new_val in candidates:
                        if time.time() > deadline:
                            break
                        if new_val >= old_val:
                            continue
                        
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        
                        conflict = new_diffs & (used - old_diffs)
                        if conflict:
                            continue
                        
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        
        return rows
    
    def local_search_swap(rows, deadline, rng, iterations=5000):
        """Local search: try to improve rows by swapping elements."""
        best = [r[:] for r in rows]
        best_scope = _scope(best)
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
                
            ri = rng.randint(0, len(best) - 1)
            pi = rng.randint(1, k)
            
            old_val = best[ri][pi]
            
            if pi < k:
                new_val = rng.randint(best[ri][pi - 1] + 1, min(best_scope, best[ri][pi + 1] - 1))
            else:
                new_val = rng.randint(best[ri][pi - 1] + 1, best_scope)
            
            if new_val == old_val:
                continue
            
            test_row = best[ri][:]
            test_row[pi] = new_val
            
            if len(set(test_row)) != len(test_row):
                continue
            
            all_diffs = set()
            valid = True
            for r in best:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        d = r[j] - r[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            for i in range(len(test_row)):
                for j in range(i + 1, len(test_row)):
                    d = test_row[j] - test_row[i]
                    if d in all_diffs:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_scope = max(max(r) for r in best if r is not test_row)
            new_scope = max(new_scope, max(test_row))
            
            if new_scope <= best_scope:
                best[ri] = test_row
                best_scope = new_scope
        
        return best, best_scope
    
    # Start with multiple greedy constructions
    best_rows = greedy_construct()
    best_scope = _scope(best_rows)
    
    # Try randomized greedy
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        rows = greedy_construct_randomized(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Try lookahead greedy
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        rows = greedy_construct_lookahead(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope
    
    # Apply local search shrink
    rng = random.Random(123)
    if time.time() < deadline - 10:
        polished = local_search_shrink(best_rows, deadline - 5, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Apply extended local search
    if time.time() < deadline - 10:
        polished = local_search_shrink_all(best_rows, deadline - 5, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Apply swap-based local search
    if time.time() < deadline - 10:
        polished, polished_scope = local_search_swap(best_rows, deadline - 5, rng, iterations=3000)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    # Try backtracking at progressively lower scopes
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)
    
    # Try a few different node budgets for backtracking
    budgets = [100000, 50000, 20000]
    
    while time.time() < deadline - 5:
        target_scope = current_max - 1
        
        if target_scope < lower_bound:
            break
        
        found = False
        for budget in budgets:
            if time.time() > deadline - 5:
                break
            rng = random.Random(int(time.time() * 1000) % 10000)
            result = backtrack_construct(target_scope, n_rows, rng, deadline - 5, budget)
            if result is not None:
                best_rows = result
                best_scope = target_scope
                current_max = target_scope
                found = True
                break
        
        if not found:
            # Try a few more random seeds
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                rng = random.Random(attempt * 7919 + int(time.time()) % 1000)
                result = backtrack_construct(target_scope, n_rows, rng, deadline - 5, 30000)
                if result is not None:
                    best_rows = result
                    best_scope = target_scope
                    current_max = target_scope
                    found = True
                    break
        
        if found:
            # Polish the new solution
            if time.time() < deadline - 5:
                polished = local_search_shrink(best_rows, deadline - 5, rng)
                polished_scope = _scope(polished)
                if polished_scope < best_scope:
                    best_rows = polished
                    best_scope = polished_scope
                    current_max = polished_scope
        else:
            current_max = target_scope
    
    # Final polish
    if time.time() < deadline - 2:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 2, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope
    
    return best_rows
