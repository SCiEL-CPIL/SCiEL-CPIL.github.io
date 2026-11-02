import time
import random
import math

def _scope(rows):
    return max(a for r in rows for a in r[1:])

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

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, rng, bias=1.0):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        last = 0
        for j in range(1, k + 1):
            candidates = []
            x = last + 1
            max_x = last + 60
            while x <= max_x and len(candidates) < 10:
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
                x += 1
            if not candidates:
                x = last + 1
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
            weights = [1.0 / (c[0] - last + 1) ** bias for c in candidates]
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

class SAState:
    def __init__(self, rows, k, max_scope_bound=200):
        self.rows = [list(r) for r in rows]
        self.k = k
        self.n_rows = len(rows)
        self.max_scope_bound = max_scope_bound
        self.used_diffs = [0] * (max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        self._rebuild_state()

    def _rebuild_state(self):
        self.used_diffs = [0] * (self.max_scope_bound + 1)
        self.collisions = 0
        self.scope = 0
        for r in self.rows:
            if r[-1] > self.scope:
                self.scope = r[-1]
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= self.max_scope_bound:
                        self.used_diffs[d] += 1
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
        if j > 0 and new_val <= row[j - 1]:
            return 0, False
        if j < self.k and new_val >= row[j + 1]:
            return 0, False

        delta_coll = 0
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1

        for t in range(self.k + 1):
            if t == j:
                continue
            d_old = abs(row[t] - old_val)
            if d_old <= self.max_scope_bound:
                self.used_diffs[d_old] -= 1
            d_new = abs(row[t] - new_val)
            if d_new <= self.max_scope_bound:
                self.used_diffs[d_new] += 1

        self.collisions += delta_coll

        old_scope = self.scope
        if new_val > self.scope:
            self.scope = new_val
        elif old_val == self.scope and new_val < old_val:
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
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - new_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 1:
                    delta_coll -= 1
                self.used_diffs[d] -= 1
        for t in range(self.k + 1):
            if t == j:
                continue
            d = abs(row[t] - old_val)
            if d <= self.max_scope_bound:
                if self.used_diffs[d] > 0:
                    delta_coll += 1
                self.used_diffs[d] += 1
        self.collisions += delta_coll

        old_scope = self.scope
        if old_val > self.scope:
            self.scope = old_val
        elif new_val == self.scope and old_val < new_val:
            max_s = 0
            for r in self.rows:
                if r[-1] > max_s:
                    max_s = r[-1]
            self.scope = max_s

        row[j] = old_val


def _sa_optimize(rows, k, deadline, rng, T0=10.0, cooling=0.9995, min_T=0.01):
    n_rows = len(rows)
    max_scope_bound = max(_scope(rows), 150) + 50
    state = SAState(rows, k, max_scope_bound)
    
    best_scope = state.scope
    best_rows = [list(r) for r in state.rows]
    
    T = T0
    penalty = 1000
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = state.rows[i][j]
        prev_val = state.rows[i][j - 1]
        next_val = state.rows[i][j + 1] if j < k else state.scope + 200
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
        
        if rng.random() < 0.7 and old_val > lo:
            max_drop = old_val - lo
            drop = rng.randint(1, max(1, max_drop // 3))
            new_val = old_val - drop
        else:
            span = hi - lo
            if span <= 0:
                continue
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
            new_val = max(lo, min(hi, new_val))
        
        if new_val == old_val:
            continue
        
        delta_cost, success = state.try_move(i, j, new_val)
        
        if not success:
            T = max(T * cooling, min_T)
            continue
        
        if delta_cost < 0 or (T > min_T and rng.random() < math.exp(-delta_cost / T)):
            if state.collisions == 0 and state.scope < best_scope:
                best_scope = state.scope
                best_rows = [list(r) for r in state.rows]
                if rng.random() < 0.03:
                    T = min(T * 1.5, 15.0)
        else:
            state.revert_move(i, j, new_val, old_val)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _backtrack_construct(n_rows, k, max_scope, rng):
    used = set()
    rows = [None] * n_rows
    
    def backtrack(row_idx, col_idx, current_row):
        if row_idx == n_rows:
            return True
        
        if col_idx == 0:
            current_row = [0]
            return backtrack(row_idx, 1, current_row)
        
        if col_idx > k:
            rows[row_idx] = current_row
            row_diffs = _row_diffs(current_row)
            prev_used = set(used)
            used.update(row_diffs)
            
            if backtrack(row_idx + 1, 0, []):
                return True
            
            used.clear()
            used.update(prev_used)
            return False
        
        prev_val = current_row[-1]
        candidates = []
        for val in range(prev_val + 1, max_scope + 1):
            valid = True
            for i in range(len(current_row)):
                d = val - current_row[i]
                if d in used:
                    valid = False
                    break
            if valid:
                candidates.append(val)
                if len(candidates) > 20:
                    break
        
        if not candidates:
            return False
        
        rng.shuffle(candidates)
        
        for val in candidates:
            current_row.append(val)
            if backtrack(row_idx, col_idx + 1, current_row):
                return True
            current_row.pop()
        
        return False
    
    if backtrack(0, 0, []):
        return rows
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    init_deadline = start_time + 5.0
    candidates = []
    
    while time.time() < init_deadline:
        bias = rng.uniform(0.5, 2.0)
        rows = _greedy_construct(n_rows, k, rng, bias=bias)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
        
        if best_scope < float('inf') and rng.random() < 0.3:
            target_scope = best_scope - 1
            if target_scope > n_rows * k:
                bt_rows = _backtrack_construct(n_rows, k, target_scope, rng)
                if bt_rows and _valid(bt_rows, k):
                    s = _scope(bt_rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = [list(r) for r in bt_rows]
                    candidates.append((s, bt_rows))
    
    if not candidates:
        rows = _greedy_construct(n_rows, k, rng, bias=1.0)
        if _valid(rows, k):
            best_rows = rows
            best_scope = _scope(rows)
        else:
            return [[0, 1, 2, 3, 4, 5]] * n_rows
    
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:5]
    
    remaining_time = deadline - time.time()
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
                
                target_scope = best_scope - 1
                if target_scope > 0 and time.time() < deadline - 10:
                    bt_rows = _backtrack_construct(n_rows, k, target_scope, rng)
                    if bt_rows and _valid(bt_rows, k):
                        s2 = _scope(bt_rows)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = [list(r) for r in bt_rows]
    
    return best_rows
