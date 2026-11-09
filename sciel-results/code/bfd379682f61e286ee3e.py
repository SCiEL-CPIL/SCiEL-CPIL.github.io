import time
import random

def construct(n_rows, k):
    deadline = time.time() + 110.0
    
    # n_rows=4, k=7 => 8 elements
    # Total diffs = 112
    
    def greedy_solve():
        used = 0
        rows = []
        for r in range(n_rows):
            row = [0]
            for p in range(1, k+1):
                x = row[-1] + 1
                while True:
                    valid = True
                    new_mask = 0
                    for i in range(p):
                        d = x - row[i]
                        if (used >> d) & 1:
                            valid = False
                            break
                        new_mask |= (1 << d)
                    if valid:
                        used |= new_mask
                        row.append(x)
                        break
                    x += 1
                    if x > 300:
                        return None
            rows.append(row)
        return rows

    init_rows = greedy_solve()
    if init_rows:
        best_rows = init_rows
        best_scope = max(max(r) for r in init_rows)
    else:
        best_rows = None
        best_scope = 200

    def attempt_solve(S_max, time_limit):
        start_time = time.time()
        rows = [[0]*(k+1) for _ in range(n_rows)]
        
        def bt(r, p, used_mask):
            if time.time() > start_time + time_limit:
                return False
            
            if r == n_rows:
                return True
            
            if p == k + 1:
                return bt(r + 1, 1, used_mask)
            
            prev_val = rows[r][p-1]
            
            # Generate candidates
            candidates = []
            for x in range(prev_val + 1, S_max + 1):
                new_mask = 0
                valid = True
                for i in range(p):
                    d = x - rows[r][i]
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                if valid:
                    candidates.append((x, new_mask))
            
            if not candidates:
                return False
            
            # Heuristic: If few candidates, shuffle. If many, keep order.
            if len(candidates) < 50:
                random.shuffle(candidates)
            
            for x, new_mask in candidates:
                rows[r][p] = x
                if bt(r, p + 1, used_mask | new_mask):
                    return True
            return False

        return bt(0, 1, 0)

    # Search S from 134 down to 112
    current_S = min(best_scope - 1, 134)
    
    while current_S >= 112:
        if time.time() > deadline:
            break
        
        # Give 5 seconds to this S
        if attempt_solve(current_S, 5.0):
            # Verify
            all_diffs = set()
            valid = True
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid: break
            if valid:
                best_scope = current_S
                best_rows = [list(r) for r in rows]
        
        current_S -= 1

    return best_rows
