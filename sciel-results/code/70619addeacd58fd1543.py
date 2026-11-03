import time
import random
import math

class SAState:
    def __init__(self, rows, k, max_scope_bound=300):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.max_scope_bound = max_scope_bound
        self._rebuild_state()

    def _rebuild_state(self):
        self.scope = max(r[-1] for r in self.rows)
        # used_diffs[d] = count of times difference d appears across all rows
        self.used_diffs = [0] * (self.max_scope_bound + 1)
        for r in self.rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= self.max_scope_bound:
                        self.used_diffs[d] += 1
        self.collisions = 0
        for d in range(1, self.max_scope_bound + 1):
            if self.used_diffs[d] > 1:
                self.collisions += self.used_diffs[d] - 1

    def get_cost(self):
        return self.scope + 1000 * self.collisions

    def try_move(self, i, j, new_val):
        row = self.rows[i]
        old_val = row[j]
        
        if new_val == old_val:
            return 0, True
        
        if j > 0 and new_val <= row[j-1]:
            return 0, False
        if j < self.k and new_val >= row[j+1]:
            return 0, False
            
        delta_coll = 0
        
        # Removal phase
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
                self.used_diffs[d] -= 1
                
        # Addition phase
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1
                self.used_diffs[d] += 1
                
        self.collisions += delta_coll
        
        # Scope update
        old_scope = self.scope
        if new_val > old_scope:
            self.scope = new_val
        elif old_val == old_scope and new_val < old_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s
            
        row[j] = new_val
        delta_cost = (self.scope - old_scope) + 1000 * delta_coll
        return delta_cost, True

    def revert_move(self, i, j, new_val, old_val):
        row = self.rows[i]
        delta_coll = 0
        
        # Remove new diffs
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
                self.used_diffs[d] -= 1

        # Add old diffs
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1
                self.used_diffs[d] += 1

        self.collisions += delta_coll
        
        # Scope update
        if old_val > self.scope:
            self.scope = old_val
        elif new_val == self.scope and old_val < new_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s

        row[j] = old_val

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _csp_construct(n_rows, k, rng):
    """
    Construct rows using backtracking with forward checking.
    This tends to find more compact valid sets faster than greedy random.
    """
    max_scope_limit = 130 # A reasonable upper bound for (7,5) is 112, so 130 is safe
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    def backtrack(idx, row, used, current_row_idx):
        # idx is the position in the current row (1 to k)
        if idx == k + 1:
            return [0] + row
        
        # Heuristic: choose the smallest valid value, or randomize
        # We'll try values starting from row[-1] + 1
        
        # To introduce diversity and escape local minima, we can randomize the order slightly
        # But for CSP, deterministic forward checking is strong.
        # Let's try a mix: mostly smallest, occasionally jump.
        
        last_val = row[-1] if row else 0
        
        # We need to find x such that x - y is not in used for all y in [0] + row
        # And x - y > 0 (guaranteed since x > last_val >= y)
        
        # Optimization: Pre-check if we have enough room
        # The largest diff in this row will be x - 0 = x.
        # The next diffs will be larger.
        
        # Let's iterate x from last_val + 1 upwards
        # We limit the search depth to avoid infinite loops if impossible
        start = last_val + 1
        end = max_scope_limit
        
        # Shuffle candidates slightly to allow different structures?
        # No, let's keep it deterministic first to find a valid solution quickly.
        # Then we can perturb it.
        
        # Actually, to get diverse solutions for SA, we can randomize the choice among valid candidates.
        # But CSP needs to find *a* solution.
        
        valid_candidates = []
        # We don't want to scan all way to max_scope_limit if we find a few early
        # Let's scan a window
        search_limit = min(end, last_val + 30) 
        
        for x in range(start, search_limit + 1):
            is_valid = True
            new_diffs = []
            for y in [0] + row:
                d = x - y
                if d in used:
                    is_valid = False
                    break
                new_diffs.append(d)
            
            if is_valid:
                valid_candidates.append((x, new_diffs))
                
                # Early termination if we have enough choices?
                # For the last element, we just need one.
                # For earlier elements, having more choices is better for backtracking.
                if len(valid_candidates) > 10:
                    break

        if not valid_candidates:
            # If no candidates in the window, try a bit further?
            # Or fail.
            # If we fail, backtracking will try a different previous value.
            return None

        # Select candidate
        # For the last element (idx == k), we want the smallest to minimize scope.
        # For earlier elements, we might want to keep options open.
        # Simple strategy: Pick the smallest.
        # To create diversity, we can pick randomly from the first few.
        if idx == k:
            choice = valid_candidates[0]
        else:
            # Pick randomly from the first 3 to vary the structure
            n_pick = min(3, len(valid_candidates))
            choice = valid_candidates[rng.randint(0, n_pick - 1)]
        
        x, new_diffs = choice
        used.update(new_diffs)
        result = backtrack(idx + 1, row + [x], used, current_row_idx)
        
        if result is not None:
            return result
        
        used.difference_update(new_diffs)
        
        # If backtracking fails, try the next candidate?
        # The above logic only tries one candidate.
        # We need to loop through candidates.
        
        # Let's re-implement with a loop over candidates
        # But the "window" approach above is inefficient for backtracking.
        
        return None

    # The above backtrack is slightly flawed because it doesn't iterate through all candidates on failure.
    # Let's write a proper backtracking CSP.
    
    def _bt(row_idx, row, used):
        if row_idx == n_rows:
            return None # All rows placed
        if len(row) == k + 1:
            # Row complete, move to next
            return _bt(row_idx + 1, [], used)
        
        last_val = row[-1] if row else 0
        # Try values
        # To speed up, we can estimate the minimum possible next value
        # and the maximum allowed value based on the scope limit.
        
        # Heuristic: Try smaller values first to keep scope low.
        # Randomization: Shuffle the range? No, that kills the "smallest first" heuristic which is good for feasibility.
        # Instead, we can randomize the *order* of rows? Or the starting point?
        
        # Let's stick to smallest first, but allow some randomization in the "window" of tried values
        # if we get stuck.
        
        # Actually, for (7,5), the scope is tight.
        # Let's use a standard backtracking with MRV (Minimum Remaining Values) is hard here.
        # Simple DFS with smallest-first.
        
        start = last_val + 1
        # Upper bound: We have to fit k - len(row) more elements.
        # The max element will be at least last_val + (k - len(row))
        # But we also have the global scope limit.
        
        # Let's just iterate.
        # To avoid infinite loops or slow search, we can limit the branch depth or use a time check.
        
        # Optimization: If we are at the last position of the last row, we can calculate the exact value needed? No.
        
        # Let's try values in order.
        # We'll add a small randomization to the start value sometimes to escape local minima in the CSP search itself?
        # No, CSP search is deterministic if we fix the order.
        # To get diverse solutions, we can vary the *row order* or *column order* (element order).
        # Here columns are fixed (0 to k).
        
        # Let's just run a standard backtracking.
        # We'll add a time check inside to bail out if it takes too long.
        
        current_time = time.time()
        
        # We'll try a limited number of values per node to keep it fast, 
        # and rely on the outer loop (SA) to fix things if it fails?
        # No, if CSP fails, we don't have a valid solution.
        # So CSP must be robust.
        
        # Given the small size (7 rows, 6 cols), standard backtracking with pruning should be fast enough
        # if we have a good bound.
        
        # Let's try up to max_scope_limit values.
        # But 130 is small.
        
        for x in range(start, max_scope_limit + 1):
            valid = True
            new_diffs = []
            for y in [0] + row:
                d = x - y
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                used.update(new_diffs)
                res = _bt(row_idx, row + [x], used)
                if res is not None:
                    return res
                used.difference_update(new_diffs)
                
            # Time check
            if time.time() - current_time > 0.1:
                return None

        return None

    # The function _bt returns None if it fails to place a row, 
    # but my logic above `if row_idx == n_rows: return None` is wrong.
    # It should return a list of rows.
    
    rows = []
    
    def _bt2(row_idx, row, used):
        if row_idx == n_rows:
            return rows[:]
        
        if len(row) == k + 1:
            rows.append(row)
            res = _bt2(row_idx + 1, [], used)
            if res is not None:
                return res
            rows.pop()
            return None
        
        last_val = row[-1] if row else 0
        
        # Heuristic: 
        # If we are near the scope limit, we must be careful.
        
        for x in range(last_val + 1, max_scope_limit + 1):
            valid = True
            new_diffs = []
            for y in [0] + row:
                d = x - y
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                used.update(new_diffs)
                res = _bt2(row_idx, row + [x], used)
                if res is not None:
                    return res
                used.difference_update(new_diffs)
                
            # Time check
            if time.time() > _bt2.start_time + 1.0:
                return None

    _bt2.start_time = time.time()
    result = _bt2(0, [], set())
    if result:
        return result
    
    # If CSP fails, fall back to greedy
    return None

def _greedy_random(n_rows, k, rng, aggressiveness=0.5):
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_search = last + 50
            while x <= max_search:
                valid = True
                new_diffs = []
                for i in range(len(row)):
                    d = x - row[i]
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    candidates.append((x, new_diffs))
                    if len(candidates) >= 5:
                        break
                x += 1
            
            if not candidates:
                while True:
                    valid = True
                    new_diffs = []
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        candidates.append((x, new_diffs))
                        break
                    x += 1
            
            power = 1.0 + aggressiveness
            weights = [1.0 / (c[0] - last + 1) ** power for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen = c
                    break
            
            x_val, new_diffs = chosen
            row.append(x_val)
            last = x_val
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows

def _sa_optimize(rows, k, deadline, rng):
    state = SAState(rows, k, 300)
    best_rows = [list(r) for r in state.rows]
    best_cost = state.get_cost()
    best_scope = state.scope
    
    T = 5.0
    cooling = 0.9999
    min_T = 0.01
    n_rows = len(rows)
    
    # Track best valid solution
    if state.collisions == 0:
        best_scope = state.scope
        best_rows = [list(r) for r in state.rows]

    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = state.rows[i][j]
        prev_val = state.rows[i][j-1]
        next_val = state.rows[i][j+1] if j < k else state.scope + 100
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
            
        # Propose new value
        if rng.random() < 0.6:
            # Try to move down (reduce scope)
            if old_val > lo:
                step = rng.randint(1, max(1, (old_val - lo) // 2))
                new_val = old_val - step
            else:
                new_val = old_val
        else:
            # Random perturbation
            span = hi - lo
            if span == 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.3)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
            
        if new_val == old_val:
            continue
            
        old_cost = state.get_cost()
        delta_cost, success = state.try_move(i, j, new_val)
        
        if not success:
            continue
            
        new_cost = old_cost + delta_cost
        
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / max(T, 0.01)):
            # Accept
            if state.collisions == 0:
                if state.scope < best_scope:
                    best_scope = state.scope
                    best_rows = [list(r) for r in state.rows]
        else:
            # Revert
            state.revert_move(i, j, new_val, old_val)
            
        T = max(T * cooling, min_T)

    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    # 1. Try CSP for a tight solution
    csp_deadline = start_time + 5.0
    while time.time() < csp_deadline:
        rows = _csp_construct(n_rows, k, rng)
        if rows and _valid(rows, k):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
            break # CSP is deterministic-ish, one good try is enough if it succeeds quickly
        else:
            break # If CSP fails or times out, don't retry
    
    # 2. Generate random greedy solutions
    init_deadline = start_time + 15.0
    candidates = []
    if best_rows:
        candidates.append((best_scope, best_rows))
        
    while time.time() < init_deadline:
        aggr = rng.uniform(0.2, 1.0)
        rows = _greedy_random(n_rows, k, rng, aggressiveness=aggr)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                
    if not candidates:
        # Fallback
        rows = _greedy_random(n_rows, k, rng, aggressiveness=0.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return [[0, 1, 2, 3, 4, 5]] * n_rows

    # Phase 2: Optimize the best few candidates
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:3]
    
    remaining_time = deadline - time.time()
    if remaining_time < 10:
        return best_rows
        
    time_per_sa = remaining_time / len(top_candidates)
    
    for i, (s, rows) in enumerate(top_candidates):
        if time.time() >= deadline - 5:
            break
            
        sa_deadline = time.time() + time_per_sa * 0.9
        sa_scope, sa_rows = _sa_optimize(rows, k, sa_deadline, rng)
        
        if _valid(sa_rows, k):
            actual_scope = _scope(sa_rows)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = sa_rows

    return best_rows
