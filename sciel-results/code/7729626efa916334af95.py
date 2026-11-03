import time
import random
import math
from collections import Counter

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

def _fast_greedy(k, n_rows, rng=None, spread=False):
    """Bitmask-based greedy construction for fast initialization."""
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    """Rebuild a row with minimal conflicts using the current global diff counts."""
    best_row = None
    best_score = float('inf')
    
    for attempt in range(10):
        row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.4:
                start = x + rng.randint(1, 4)
            for t in range(start, max_scope + 1):
                conflicts = 0
                new_diffs = []
                for a in row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
                elif attempt < 4 and conflicts <= 1 and rng.random() < 0.12:
                    row.append(t)
                    local_diffs.update(new_diffs)
                    x = t
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok and len(row) == k + 1:
            score = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if diff_counts.get(d, 0) > 0:
                        score += 1
            if score < best_score:
                best_score = score
                best_row = list(row)
            if best_score == 0:
                break
    
    return best_row

def _ga_optimize(rows, k, rng, deadline, pop_size=20):
    """
    Genetic algorithm operating in difference-space.
    Each individual is represented by per-row consecutive difference sequences.
    """
    n_rows = len(rows)
    row_len = k + 1
    
    # Convert rows to diff-space
    def rows_to_diffs(rows):
        return [[r[i+1] - r[i] for i in range(k)] for r in rows]
    
    def diffs_to_rows(diffs):
        out = []
        for d in diffs:
            row = [0]
            for v in d:
                row.append(row[-1] + v)
            out.append(row)
        return out
    
    def compute_fitness(individual):
        """Returns (collisions, max_val)"""
        rows = diffs_to_rows(individual)
        dc = _build_diff_counts(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        max_val = _scope(rows)
        return collisions, max_val
    
    # Initialize population
    best_ind = rows_to_diffs(rows)
    pop = [best_ind[:]]
    
    for _ in range(pop_size - 1):
        ind = [d[:] for d in best_ind]
        for _ in range(rng.randint(2, 5)):
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(0, k - 1)
            old = ind[r_idx][pos]
            new = max(1, old + rng.randint(-3, 3))
            ind[r_idx][pos] = new
        pop.append(ind)
    
    best_collisions, best_max = compute_fitness(best_ind)
    best_solution = best_ind
    
    T = 50.0
    cooling = 0.9998
    min_T = 0.5
    
    while time.time() < deadline:
        # Evaluate population
        scored = []
        for ind in pop:
            coll, mx = compute_fitness(ind)
            scored.append((coll, mx, ind))
        
        scored.sort(key=lambda x: (x[0], x[1]))
        
        # Update best
        if scored[0][0] < best_collisions or (scored[0][0] == best_collisions and scored[0][1] < best_max):
            best_collisions = scored[0][0]
            best_max = scored[0][1]
            best_solution = [d[:] for d in scored[0][2]]
        
        # Selection: tournament
        def select():
            a = rng.randint(0, min(5, len(scored) - 1))
            b = rng.randint(0, min(5, len(scored) - 1))
            if scored[a][0] < scored[b][0] or (scored[a][0] == scored[b][0] and scored[a][1] <= scored[b][1]):
                return scored[a][2]
            return scored[b][2]
        
        # Generate next generation
        new_pop = []
        
        # Elitism
        for i in range(min(3, len(scored))):
            new_pop.append([d[:] for d in scored[i][2]])
        
        # Crossover and mutation
        while len(new_pop) < pop_size and time.time() < deadline:
            p1 = select()
            p2 = select()
            
            child = [d[:] for d in p1]
            
            # Crossover: swap rows or segments
            for r_idx in range(n_rows):
                if rng.random() < 0.5:
                    child[r_idx] = p2[r_idx][:]
                elif rng.random() < 0.3 and k >= 2:
                    seg_start = rng.randint(0, k - 2)
                    seg_end = rng.randint(seg_start, k - 1)
                    child[r_idx][seg_start:seg_end+1] = p2[r_idx][seg_start:seg_end+1]
            
            # Mutation
            for _ in range(rng.randint(1, 3)):
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(0, k - 1)
                old = child[r_idx][pos]
                if rng.random() < 0.6:
                    delta = rng.randint(-2, 1)
                else:
                    delta = rng.randint(-3, 3)
                new_val = max(1, old + delta)
                child[r_idx][pos] = new_val
            
            new_pop.append(child)
        
        # Occasionally re-inject best with perturbation
        if rng.random() < 0.15:
            perturbed = [d[:] for d in best_solution]
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(0, k - 1)
            perturbed[r_idx][pos] = max(1, perturbed[r_idx][pos] + rng.randint(-2, 2))
            new_pop.append(perturbed)
        
        pop = new_pop[:pop_size]
        T = max(T * cooling, min_T)
    
    # Convert best back to rows
    best_rows = diffs_to_rows(best_solution)
    if _valid(best_rows, k):
        return _scope(best_rows), best_rows
    
    # Return original if GA didn't find valid
    return _scope(rows), [list(r) for r in rows]

def _sa_optimize(rows, k, iters, rng, deadline, polish=False):
    """Simulated annealing with row rebuilds, scope compression, and single-element moves."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    if polish:
        T = 0.5
        cooling = 0.9999
        min_T = 0.01
    else:
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        r = rng.random()
        
        if r < 0.15:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(current_rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, current_scope + 5)
            if new_row is None:
                _add_row_diffs(diff_counts, old_row)
                continue
            
            _add_row_diffs(diff_counts, new_row)
            current_rows[i] = new_row
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, new_row)
                _add_row_diffs(diff_counts, old_row)
                current_rows[i] = old_row
                
        elif r < 0.35:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if current_rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = current_rows[i][j]
            
            # Try to decrease this value
            prev_val = current_rows[i][j-1]
            if old_val - prev_val <= 1:
                continue
            new_val = rng.randint(prev_val + 1, old_val - 1)
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
                
        else:
            # Single-element move
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(1, k)
            old_val = current_rows[i][j]
            
            lo = current_rows[i][j-1] + 1
            hi = current_rows[i][j+1] - 1 if j < k else current_scope
            
            if hi <= lo:
                continue
            
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = new_val
            _add_row_diffs(diff_counts, current_rows[i])
            
            new_cost = get_cost(current_rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = _scope(current_rows)
                    best_rows = [list(r) for r in current_rows]
            else:
                _remove_row_diffs(diff_counts, current_rows[i])
                current_rows[i][j] = old_val
                _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def _local_polish(rows, k, deadline, rng):
    """Aggressive local search to compress scope."""
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    diff_counts = _build_diff_counts(current_rows)
    
    def get_cost(rows, dc):
        scope = _scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        return scope + 500 * collisions

    current_cost = get_cost(current_rows, diff_counts)
    best_cost = current_cost
    
    T = 0.3
    cooling = 0.99995
    min_T = 0.005
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current_rows[i][j]
        
        lo = current_rows[i][j-1] + 1
        hi = current_rows[i][j+1] - 1 if j < k else best_scope
        
        if hi <= lo:
            continue
        
        # Try decreasing first
        if old_val > lo:
            new_val = rng.randint(lo, old_val - 1)
        else:
            new_val = rng.randint(old_val + 1, hi)
        
        if new_val == old_val:
            continue
        
        _remove_row_diffs(diff_counts, current_rows[i])
        current_rows[i][j] = new_val
        _add_row_diffs(diff_counts, current_rows[i])
        
        new_cost = get_cost(current_rows, diff_counts)
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = _scope(current_rows)
                best_rows = [list(r) for r in current_rows]
        else:
            _remove_row_diffs(diff_counts, current_rows[i])
            current_rows[i][j] = old_val
            _add_row_diffs(diff_counts, current_rows[i])
        
        T = max(T * cooling, min_T)
    
    return best_scope, best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions via fast greedy
    candidates = []
    init_deadline = start_time + 5.0
    for trial in range(15):
        if time.time() > init_deadline:
            break
        spread = rng.random() < 0.3
        rows = _fast_greedy(k, n_rows, rng, spread=spread)
        if _valid(rows, k):
            s = _scope(rows)
            candidates.append((s, rows))
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if not candidates:
        # Last resort: simple sequential
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            for j in range(1, k+1):
                x = len(used) + j + 1
                while any((x - row[i]) in used for i in range(len(row))):
                    x += 1
                for i in range(len(row)):
                    used.add(x - row[i])
                row.append(x)
            rows.append(row)
        return rows
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Genetic Algorithm on top candidate (difference-space)
    if time.time() < deadline - 30:
        ga_deadline = time.time() + min(30, deadline - time.time() - 20)
        s, rows = candidates[0]
        ga_scope, ga_rows = _ga_optimize(rows, k, rng, ga_deadline, pop_size=15)
        
        if ga_rows is not None and _valid(ga_rows, k):
            actual = _scope(ga_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = ga_rows
                candidates.append((actual, ga_rows))
                candidates.sort(key=lambda x: x[0])
    
    # Phase 3: SA optimization on top candidates
    remaining = deadline - time.time()
    if remaining > 15:
        top_n = min(3, len(candidates))
        time_per = (remaining - 15) / max(1, top_n)
        
        for idx in range(top_n):
            if time.time() > deadline - 15:
                break
            s, rows = candidates[idx]
            sa_deadline = time.time() + time_per * 0.85
            sa_scope, sa_rows = _sa_optimize(rows, k, 10000, rng, sa_deadline)
            
            if sa_rows is not None and _valid(sa_rows, k):
                actual = _scope(sa_rows)
                if actual < best_scope:
                    best_scope = actual
                    best_rows = sa_rows
                    candidates.append((actual, sa_rows))
                    candidates.sort(key=lambda x: x[0])
    
    # Phase 4: Local polish on the best solution
    if best_rows is not None and time.time() < deadline - 5:
        polish_deadline = time.time() + min(10, deadline - time.time() - 5)
        polish_scope, polish_rows = _local_polish(best_rows, k, polish_deadline, rng)
        
        if polish_rows is not None and _valid(polish_rows, k):
            actual = _scope(polish_rows)
            if actual < best_scope:
                best_scope = actual
                best_rows = polish_rows
    
    return best_rows
