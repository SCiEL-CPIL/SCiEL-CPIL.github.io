import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin

    def greedy_init():
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
        return rows, used

    rows, used_diffs = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    def get_forbidden(i):
        forbidden = set()
        for j in range(n_rows):
            if j == i: continue
            r = rows[j]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    forbidden.add(r[b] - r[a])
        return forbidden

    def find_row(max_val, forbidden, timeout=0.1):
        row = [0]
        used_in_row = set()
        t_start = time.time()
        
        def backtrack(start, idx):
            if time.time() - t_start > timeout:
                return False
            if idx == k:
                return True
            
            # Try values from max_val down to start
            for v in range(max_val, start - 1, -1):
                new_diffs = {v - a for a in row}
                if (new_diffs & forbidden) or (new_diffs & used_in_row):
                    continue
                
                row.append(v)
                used_in_row.update(new_diffs)
                
                if backtrack(v + 1, idx + 1):
                    return True
                
                row.pop()
                used_in_row -= new_diffs
            return False
        
        if backtrack(1, 1):
            return row[:]
        return None

    while time.time() < deadline:
        improved = False
        indices = sorted(range(n_rows), key=lambda i: rows[i][-1], reverse=True)
        
        for i in indices:
            if time.time() >= deadline:
                break
                
            current_max = rows[i][-1]
            if current_max <= k: # Min possible max is k (0, 1, ..., k)
                continue
                
            forbidden = get_forbidden(i)
            
            # Try to find a row with max value current_max - 1
            new_row = find_row(current_max - 1, forbidden)
            
            if new_row:
                # Update
                old_r = rows[i]
                for a in range(len(old_r)):
                    for b in range(a + 1, len(old_r)):
                        used_diffs.discard(old_r[b] - old_r[a])
                
                for a in range(len(new_row)):
                    for b in range(a + 1, len(new_row)):
                        used_diffs.add(new_row[b] - new_row[a])
                        
                rows[i] = new_row
                improved = True
                
                current_scope = max(max(r) for r in rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in rows]
        
        if not improved:
            break

    return best_rows
