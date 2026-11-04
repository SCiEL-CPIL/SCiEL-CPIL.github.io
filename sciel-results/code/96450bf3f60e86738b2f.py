import time
import random
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

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

def _greedy_construct(n_rows, k, randomize=False, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                # Simple random step
                step = rng.randint(1, 10)
                x = row[-1] + step
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    # Fallback to sequential
                    x = row[-1]
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _simulated_annealing_improve(rows, k, deadline, rng=None):
    if rng is None:
        rng = random
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    current = [list(r) for r in best]
    current_scope = best_scope
    
    T = 10.0
    T_min = 0.01
    alpha = 0.95
    
    while time.time() < deadline - 2.0 and T > T_min:
        r_idx = rng.randint(0, len(current) - 1)
        row = current[r_idx]
        i = rng.randint(1, len(row) - 2)
        j = rng.randint(i + 1, len(row) - 1)
        delta = rng.randint(-5, 5)
        if delta == 0:
            continue
        valid_move = True
        for p in range(i, j + 1):
            new_val = row[p] + delta
            if p > 0 and new_val <= row[p-1]:
                valid_move = False
                break
            if p + 1 < len(row) and new_val >= row[p+1]:
                valid_move = False
                break
            row[p] = new_val
        if not valid_move:
            # Revert
            for p in range(i, j + 1):
                row[p] -= delta
            continue
        
        if _valid(current):
            new_scope = _scope(current)
            delta_e = new_scope - current_scope
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best = [list(r) for r in current]
                    best_scope = new_scope
            else:
                for p in range(i, j + 1):
                    row[p] -= delta
        else:
            for p in range(i, j + 1):
                row[p] -= delta
        T *= alpha

    return best, best_scope

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    best_rows = None
    best_scope = float('inf')
    
    try:
        # Phase 1: Greedy baseline
        rows = _greedy_construct(n_rows, k)
        best_rows = rows
        best_scope = _scope(rows)
        
        # Phase 2: Simulated Annealing
        if time.time() < deadline - 10.0:
            sa_rows, sa_scope = _simulated_annealing_improve(best_rows, k, deadline - 10.0)
            if sa_scope < best_scope:
                best_rows = sa_rows
                best_scope = sa_scope
                
        # Phase 3: More SA with different seed
        if time.time() < deadline - 5.0:
            rng2 = random.Random(42)
            sa_rows2, sa_scope2 = _simulated_annealing_improve(best_rows, k, deadline - 5.0, rng=rng2)
            if sa_scope2 < best_scope:
                best_rows = sa_rows2
                best_scope = sa_scope2
                
    except Exception as e:
        pass
        
    if best_rows is None:
        # Fallback
        best_rows = _greedy_construct(n_rows, k)
        
    return best_rows

# EVOLVE-BLOCK-END
