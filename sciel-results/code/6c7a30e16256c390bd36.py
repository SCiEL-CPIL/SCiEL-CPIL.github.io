import time
import random

def construct(n_rows, k):
    deadline = time.time() + 100
    total_diffs = n_rows * k * (k+1) // 2
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    # Greedy initial
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
            x += 1
        rows.append(row)
    
    best = [r[:] for r in rows]
    best_scope = get_scope(best)
    
    # Backtracking
    node_count = [0]
    
    def backtrack(row_idx, used_diffs, current_rows, current_max):
        nonlocal best, best_scope
        if time.time() > deadline:
            return
        node_count[0] += 1
        if node_count[0] > 300000:
            return
        if row_idx == n_rows:
            if len(used_diffs) == total_diffs:
                if current_max < best_scope:
                    best_scope = current_max
                    best = [r[:] for r in current_rows]
            return
        
        row = [0]
        def build(pos, row, row_diffs, row_max):
            nonlocal best_scope
            if time.time() > deadline:
                return
            node_count[0] += 1
            if node_count[0] > 300000:
                return
            if pos == k + 1:
                if row_diffs & used_diffs:
                    return
                new_used = used_diffs | row_diffs
                current_rows.append(row[:])
                backtrack(row_idx + 1, new_used, current_rows, max(current_max, row_max))
                current_rows.pop()
                return
            
            start_x = row[-1] + 1
            for x in range(start_x, best_scope + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & row_diffs or new_diffs & used_diffs:
                    continue
                row.append(x)
                build(pos + 1, row, row_diffs | new_diffs, max(row_max, x))
                row.pop()
        
        build(1, row, set(), 0)
    
    # Run backtracking
    node_count[0] = 0
    backtrack(0, set(), [], 0)
    
    # Local search
    current = [r[:] for r in best]
    current_scope = best_scope
    
    while time.time() < deadline:
        improved = False
        for r_idx in range(n_rows):
            for m_idx in range(1, len(current[r_idx])):
                orig = current[r_idx][m_idx]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = orig + delta
                    if new_val <= 0:
                        continue
                    if m_idx > 1 and new_val <= current[r_idx][m_idx-1]:
                        continue
                    if m_idx < len(current[r_idx])-1 and new_val >= current[r_idx][m_idx+1]:
                        continue
                    current[r_idx][m_idx] = new_val
                    if is_valid(current):
                        new_scope = get_scope(current)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved = True
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best = [r[:] for r in current]
                        else:
                            current[r_idx][m_idx] = orig
                    else:
                        current[r_idx][m_idx] = orig
        if not improved:
            # Random perturbation
            for _ in range(10):
                r_idx = random.randint(0, n_rows-1)
                m_idx = random.randint(1, len(current[r_idx])-1)
                delta = random.randint(-10, 10)
                new_val = current[r_idx][m_idx] + delta
                if new_val > 0 and (m_idx == 1 or new_val > current[r_idx][m_idx-1]) and \
                   (m_idx == len(current[r_idx])-1 or new_val < current[r_idx][m_idx+1]):
                    current[r_idx][m_idx] = new_val
            if not is_valid(current):
                current = [r[:] for r in best]
                current_scope = best_scope
    
    return best
