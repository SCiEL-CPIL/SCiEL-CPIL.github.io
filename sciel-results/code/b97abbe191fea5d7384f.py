import time
import random
import math
import sys

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    start_time = time.time()
    deadline = start_time + 140.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s
    
    def scope_of(rows):
        return max(max(r) for r in rows)
    
    # ---- Spectral/Greedy Hybrid Initial Construction ----
    def spectral_greedy(rng):
        used = set()
        rows = []
        # Pre-compute a "difficulty" order for differences based on a heuristic
        # We want to assign large differences to positions that are hard to satisfy
        # But for initial construction, we just build rows one by one.
        
        for r in range(n_rows):
            row = [0]
            cur_diffs = set()
            # We want to keep the row compact.
            # Greedy choice: smallest next value that doesn't conflict.
            # To escape local minima, we sometimes pick a slightly larger value.
            
            for pos in range(1, k + 1):
                candidates = []
                x = row[-1] + 1
                # Look ahead a bit
                while x < 500:
                    new_d = {x - a for a in row}
                    if not (new_d & used) and not (new_d & cur_diffs):
                        candidates.append(x)
                        if len(candidates) >= 10:
                            break
                    x += 1
                
                if not candidates:
                    return None
                
                # Heuristic: prefer smaller values, but with some noise
                # The "spectral" idea suggests looking at the structure of available slots.
                # Here we approximate by preferring values that create "new" large differences
                # which are less likely to conflict with future rows, or simply the smallest.
                
                # Let's try a mix: 80% smallest, 20% random from top 5 smallest
                if rng.random() < 0.8:
                    nxt = candidates[0]
                else:
                    nxt = rng.choice(candidates[:min(5, len(candidates))])
                
                row.append(nxt)
                cur_diffs |= {nxt - a for a in row[:-1]}
            
            used |= cur_diffs
            rows.append(row)
        
        return rows
    
    # ---- Standard Greedy (for fallback) ----
    def greedy_initial(rng):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            cur_diffs = set()
            for pos in range(1, k + 1):
                x = row[-1] + 1
                while True:
                    new_d = {x - a for a in row}
                    if not (new_d & used) and not (new_d & cur_diffs):
                        break
                    x += 1
                row.append(x)
                cur_diffs |= {x - a for a in row[:-1]}
            used |= cur_diffs
            rows.append(row)
        return rows
    
    # ---- Tightening: Local Search to reduce scope ----
    def tighten(rows, max_iter=500):
        rows = [list(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                # Try to reduce each element (except 0)
                for pos in range(len(row) - 1, 0, -1):
                    # Try to decrease row[pos]
                    target = row[pos] - 1
                    while target > row[pos - 1]:
                        # Check if target is valid
                        test_row = row[:]
                        test_row[pos] = target
                        # Check monotonicity
                        if test_row[pos] <= test_row[pos-1]:
                            break
                        if pos < len(test_row)-1 and test_row[pos] >= test_row[pos+1]:
                            break
                            
                        # Check internal diffs
                        idiffs = set()
                        ok = True
                        for i in range(len(test_row)):
                            for j in range(i + 1, len(test_row)):
                                d = test_row[j] - test_row[i]
                                if d in idiffs:
                                    ok = False
                                    break
                                idiffs.add(d)
                            if not ok:
                                break
                        if not ok:
                            break
                        
                        # Check cross-row diffs
                        conflict = False
                        for idx in range(n_rows):
                            if idx == r:
                                continue
                            od = row_diffs(rows[idx])
                            if idiffs & od:
                                conflict = True
                                break
                        if conflict:
                            break
                        
                        # Valid!
                        row[pos] = target
                        improved = True
                        break # Restart tightening for this row from right to left
                    # If we broke out of while without improving, try next pos
            if not improved:
                break
        return rows
    
    # ---- Simulated Annealing / Random Restart Local Search ----
    def anneal(rows, rng, max_iter=20000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]
        
        T = 10.0
        cooling = 0.999
        min_T = 0.5
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs
            
            # Generate a new row for r_idx
            # Strategy: Perturb the old row or generate a new one
            # Here we generate a new one using a greedy approach with noise
            new_row = [0]
            cur_diffs = set()
            success = True
            
            # Use the old row as a guide for initial values, but allow changes
            # We'll try to keep it close to the old row but allow "jumps"
            
            for pos in range(1, k + 1):
                # Candidate range: around old_row[pos] if it exists, else current end + 1
                base_val = old_row[pos] if pos < len(old_row) else new_row[-1] + 1
                start_val = max(new_row[-1] + 1, base_val - 5)
                
                candidates = []
                x = start_val
                # Look for valid candidates
                while x < 500 and len(candidates) < 20:
                    new_d = {x - a for a in new_row}
                    if not (new_d & used) and not (new_d & cur_diffs):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Choose candidate
                # Prefer those closer to base_val to maintain structure, but allow noise
                # Weight by distance to base_val
                weights = []
                for c in candidates:
                    dist = abs(c - base_val)
                    w = 1.0 / (1.0 + dist * 0.1)
                    weights.append(w)
                
                # Normalize
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                
                # Sample
                r_val = rng.random()
                cum = 0
                nxt = candidates[0]
                for i, w in enumerate(weights):
                    cum += w
                    if r_val <= cum:
                        nxt = candidates[i]
                        break
                
                new_row.append(nxt)
                cur_diffs |= {nxt - a for a in new_row[:-1]}
            
            if not success:
                used |= old_diffs
                continue
            
            new_diffs = row_diffs(new_row)
            # Check if new_diffs conflicts with used (should be handled in gen, but double check)
            if new_diffs & used:
                used |= old_diffs
                continue
                
            other_max = max(rows[i][-1] for i in range(n_rows) if i != r_idx)
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max
            
            if delta < 0:
                rows[r_idx] = new_row
                used |= new_diffs
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                else:
                    used |= old_diffs
            
            T = max(min_T, T * cooling)
        
        return best_rows

    # ---- Main Search ----
    rng = random.Random(42)
    
    # 1. Initial Solution
    best_rows = spectral_greedy(rng)
    if best_rows is None or not is_valid(best_rows):
        best_rows = greedy_initial(rng)
    
    # 2. Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng)
        best_rows = tighten(best_rows)
    
    best_scope = scope_of(best_rows)
    
    # 3. Anneal
    rows = anneal(best_rows, rng, max_iter=20000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
    
    # 4. Tighten again
    best_rows = tighten(best_rows)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)
    
    # 5. Diversified Restarts
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        rng2 = random.Random(100 + trial * 17)
        rows = spectral_greedy(rng2)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten(rows)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        
        # Anneal this diverse start
        rows2 = anneal(rows, rng2, max_iter=10000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
                best_rows = tighten(best_rows)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
    
    # Final Tighten
    best_rows = tighten(best_rows)
    if not is_valid(best_rows):
        # Fallback
        best_rows = greedy_initial(random.Random(7))
        best_rows = tighten(best_rows)
        
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
