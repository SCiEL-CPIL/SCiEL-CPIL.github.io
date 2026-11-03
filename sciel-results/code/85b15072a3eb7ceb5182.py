import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    lb = total_diffs_needed  # Counting bound: smallest M such that M >= total_diffs
    
    # 1. Fast greedy baseline
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < m:
                # Check if x - a is in used for all a in row
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
                x += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = max(max(r) for r in best_rows)
    
    if best_scope <= lb:
        return best_rows

    # 2. Local search / Polish to reduce scope using simulated annealing
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for d in get_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True

    def polish_sa(rows, time_limit):
        """Simulated annealing with element moves and row regeneration."""
        rows = [list(r) for r in rows]
        rng = random.Random(42)
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        # Precompute diff sets for each row
        row_diffs = [get_diffs(r) for r in rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs.update(rd)
        
        # Temperature schedule
        T = 100.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < time_limit:
            # Pick a random move type: element move (70%) or row regen (30%)
            if rng.random() < 0.7:
                # Element move: change one element in one row
                r_idx = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, m - 1)
                row = rows[r_idx]
                orig = row[pos]
                
                # Try to move to a nearby value
                delta = rng.randint(-15, 15)
                new_val = orig + delta
                
                if new_val <= row[pos - 1] or (pos < m - 1 and new_val >= row[pos + 1]):
                    T *= cooling
                    continue
                
                # Compute new diffs for this row
                new_diffs = set()
                for i in range(m):
                    for j in range(i + 1, m):
                        if i == pos:
                            new_diffs.add(row[j] - new_val)
                        elif j == pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[j] - row[i])
                
                # Check validity: no conflicts with other rows
                other_diffs = set()
                for i, rd in enumerate(row_diffs):
                    if i != r_idx:
                        other_diffs.update(rd)
                
                if new_diffs & other_diffs:
                    T *= cooling
                    continue
                
                # Accept based on simulated annealing
                new_scope = max(max(r) for r in rows[:r_idx] + [row[:pos] + [new_val] + row[pos+1:]] + rows[r_idx+1:])
                delta_scope = new_scope - current_scope
                if delta_scope < 0 or rng.random() < math.exp(-delta_scope / T):
                    # Apply move
                    old_diffs = row_diffs[r_idx]
                    row[pos] = new_val
                    row_diffs[r_idx] = new_diffs
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                T *= cooling
            else:
                # Row regeneration: rebuild one row from scratch
                r_idx = rng.randint(0, n_rows - 1)
                other_diffs = set()
                for i, rd in enumerate(row_diffs):
                    if i != r_idx:
                        other_diffs.update(rd)
                
                # Try to build a new row
                new_row = [0]
                x = 1
                max_x = min(best_scope, 200)
                while len(new_row) < m and x <= max_x:
                    ok = True
                    for a in new_row:
                        if (x - a) in other_diffs:
                            ok = False
                            break
                    if ok:
                        new_row.append(x)
                    x += 1
                
                if len(new_row) == m:
                    new_diffs = get_diffs(new_row)
                    if not (new_diffs & other_diffs):
                        # Accept if it improves or with SA probability
                        new_scope = max(max(r) for r in rows[:r_idx] + [new_row] + rows[r_idx+1:])
                        delta_scope = new_scope - current_scope
                        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / T):
                            rows[r_idx] = new_row
                            row_diffs[r_idx] = new_diffs
                            current_scope = new_scope
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [list(r) for r in rows]
                T *= cooling
            
            if T < T_min:
                T = T_min
        
        return best_rows

    # Polish the greedy solution
    best_rows = polish_sa(best_rows, deadline - 10.0)
    best_scope = max(max(r) for r in best_rows)

    # 3. Backtracking search for smaller scopes with bitset optimization
    def solve_with_scope(M, seed):
        rng = random.Random(seed)
        # Use a list of booleans for used diffs, indexed by diff value
        used_diffs = [False] * (M + 1)
        rows = []
        
        # Backtracking for one row with MRV heuristic
        def build_row(pos, current_val, row_so_far, remaining_slots):
            if time.time() > deadline:
                return None
            if pos == m:
                return list(row_so_far)
            
            # Pruning: need space for remaining elements
            max_x = M - (remaining_slots - 1)
            if max_x <= current_val:
                return None
            
            # Find valid candidates using MRV: try values that leave most options
            valid_candidates = []
            for x in range(current_val + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Heuristic: sort by how many future options they leave
            # For simplicity, use a mix of small and random
            valid_candidates.sort()
            # Take a diverse subset: first 10, then random from rest
            if len(valid_candidates) <= 15:
                candidates_to_try = valid_candidates
            else:
                candidates_to_try = valid_candidates[:10] + rng.sample(valid_candidates[10:], min(5, len(valid_candidates) - 10))
            rng.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                # Mark diffs
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True
                
                res = build_row(pos + 1, x, row_so_far + [x], remaining_slots - 1)
                if res is not None:
                    return res
                
                # Unmark
                for d in new_diffs:
                    used_diffs[d] = False
            
            return None

        # Build rows one by one
        for _ in range(n_rows):
            row = build_row(1, 0, [0], m - 1)
            if row is None:
                return None
            rows.append(row)
            # Mark all diffs in this row as used
            for i in range(m):
                for j in range(i + 1, m):
                    used_diffs[row[j] - row[i]] = True
        
        return rows

    # Try to improve scope
    attempts_per_scope = 8
    seed_counter = 0
    
    for M in range(best_scope - 1, lb - 1, -1):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        for _ in range(attempts_per_scope):
            if time.time() > deadline - 5.0:
                break
            
            rows = solve_with_scope(M, seed_counter)
            seed_counter += 1
            
            if rows is not None:
                if is_valid(rows):
                    best_rows = rows
                    best_scope = M
                    found = True
                    break
        
        if not found:
            break

    # Final polish
    if time.time() < deadline - 2.0:
        best_rows = polish_sa(best_rows, deadline - 2.0)
        best_scope = max(max(r) for r in best_rows)

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
