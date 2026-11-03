import time
import random
import sys

def _scope(rows):
    return max(max(row) for row in rows)

def _validate(rows):
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows.append(row)
    return rows

def _diffs_of_row(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

def _sa_optimize(rows, n_rows, k, deadline, rng):
    """
    Simulated annealing in the row-reconstruction space.
    Moves: pick a random row, try to reconstruct it with a random greedy that
    is biased towards keeping the scope small. Accept with Metropolis criterion.
    """
    rows = [r[:] for r in rows]
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    current_scope = best_scope
    
    # Compute total diffs for each row
    row_diffs = [_diffs_of_row(r) for r in rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    step = 0
    while time.time() < deadline:
        step += 1
        if step % 200 == 0:
            T = max(T_min, T * cooling)
        
        # Pick a random row to modify
        r_idx = rng.randrange(n_rows)
        
        # Compute diffs of all other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != r_idx:
                other_diffs |= row_diffs[i]
        
        # Try to reconstruct row r_idx with a randomized greedy
        # We want to keep the scope small
        target_max = current_scope - 1 if current_scope > 112 else current_scope
        # Actually, let's just try to build a row that fits within current_scope
        # and hope it's better, or at least valid.
        
        new_row = None
        for attempt in range(20):
            row = [0]
            local_diffs = set()
            success = True
            while len(row) < k + 1:
                cands = []
                x = row[-1] + 1
                # We need x - a not in other_diffs and not in local_diffs for all a in row
                # And we want x to be small
                max_x = min(target_max, current_scope + 5)  # Don't go too far above current
                for cand in range(x, max_x + 1):
                    new_local = {cand - a for a in row}
                    if (new_local & other_diffs) or (new_local & local_diffs):
                        continue
                    cands.append(cand)
                    # Collect a few candidates, don't scan too far
                    if len(cands) > 10:
                        break
                
                if not cands:
                    success = False
                    break
                
                # Pick with bias towards smaller
                if rng.random() < 0.7:
                    chosen = cands[0]
                else:
                    chosen = cands[rng.randrange(min(5, len(cands)))]
                
                row.append(chosen)
                for a in row[:-1]:
                    local_diffs.add(chosen - a)
            
            if success and row[-1] <= current_scope:
                new_row = row
                break
        
        if new_row is None:
            continue
        
        new_diffs = _diffs_of_row(new_row)
        new_scope = max(row_diffs[i] and _scope([rows[i]]) for i in range(n_rows) if i != r_idx)
        # Actually new_scope should be the max of the new row's max and the other rows' max
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_scope = max(new_row[-1], other_max)
        
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < (10 ** (-delta / (T + 1e-9))):
            # Accept
            rows[r_idx] = new_row
            row_diffs[r_idx] = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                
                # If we hit the counting bound, we're optimal
                if best_scope <= n_rows * k * (k + 1) // 2:
                    break
    
    return best_rows, best_scope

def _polish(rows, n_rows, k, deadline, rng):
    """
    Hill-climb: repeatedly try to decrease the largest entry in each row.
    """
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    
    def get_other_diffs(exclude_idx):
        od = set()
        for i in range(n_rows):
            if i == exclude_idx:
                continue
            row = rows[i]
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    od.add(row[b] - row[a])
        return od
    
    improved = True
    while time.time() < deadline and improved:
        improved = False
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for r_idx in row_order:
            row = rows[r_idx]
            other_diffs = get_other_diffs(r_idx)
            
            for p in range(len(row) - 1, 0, -1):
                current_val = row[p]
                lower_bound = row[p - 1] + 1
                if current_val <= lower_bound:
                    continue
                
                # Try to find the smallest valid value for position p
                # that is less than current_val
                best_cand = current_val
                for cand in range(lower_bound, current_val):
                    # Check validity
                    new_diffs = set()
                    valid = True
                    for i in range(p):
                        d = cand - row[i]
                        if d in other_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    for j in range(p + 1, len(row)):
                        d = row[j] - cand
                        if d in other_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        continue
                    # Check against static diffs (diffs not involving position p)
                    static_diffs = set()
                    for i in range(len(row)):
                        if i == p:
                            continue
                        for j in range(i + 1, len(row)):
                            if j == p:
                                continue
                        static_diffs.add(row[j] - row[i])
                    if new_diffs & static_diffs:
                        valid = False
                        continue
                    if valid:
                        best_cand = cand
                        break
                
                if best_cand < current_val:
                    row[p] = best_cand
                    improved = True
    
    return rows, _scope(rows)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    rng = random.Random(12345)
    
    # Generate initial solutions
    initial_solutions = []
    
    # Standard greedy
    initial_solutions.append(_greedy(n_rows, k))
    
    # Randomized greedy variants
    for seed in range(5):
        rng_sub = random.Random(seed * 7 + 1)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                cands = []
                x = row[-1] + 1
                for cand in range(x, x + 20):
                    new_diffs = {cand - a for a in row}
                    if not (new_diffs & used):
                        cands.append(cand)
                if not cands:
                    # Extend
                    cand = row[-1] + 1
                    while True:
                        new_diffs = {cand - a for a in row}
                        if not (new_diffs & used):
                            cands.append(cand)
                            break
                        cand += 1
                # Pick with randomness
                if rng_sub.random() < 0.75:
                    chosen = cands[0]
                else:
                    chosen = cands[rng_sub.randrange(min(3, len(cands)))]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
            rows.append(row)
        initial_solutions.append(rows)
    
    # Run SA + polish on each
    best_rows = None
    best_scope = float('inf')
    
    # Sort by initial scope
    initial_solutions.sort(key=lambda r: _scope(r))
    
    n_init = len(initial_solutions)
    time_per = (deadline - time.time() - 5) / n_init
    
    for i, sol in enumerate(initial_solutions):
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        sub_deadline = time.time() + max(1, min(time_per, remaining * 0.5))
        
        # SA phase
        sa_rows, sa_scope = _sa_optimize(sol, n_rows, k, sub_deadline, rng)
        
        # Polish phase
        polish_deadline = min(deadline - 5, time.time() + max(1, (deadline - time.time() - 5) * 0.5))
        pol_rows, pol_scope = _polish(sa_rows, n_rows, k, polish_deadline, rng)
        
        if pol_scope < best_scope:
            best_scope = pol_scope
            best_rows = pol_rows
        
        # Also try polish on the original sol
        pol2_rows, pol2_scope = _polish(sol, n_rows, k, min(deadline - 5, time.time() + max(1, (deadline - time.time() - 5) * 0.3)), rng)
        if pol2_scope < best_scope:
            best_scope = pol2_scope
            best_rows = pol2_rows
        
        if best_scope <= n_rows * k * (k + 1) // 2:
            break
    
    if best_rows is None:
        return _greedy(n_rows, k)
    
    return best_rows

