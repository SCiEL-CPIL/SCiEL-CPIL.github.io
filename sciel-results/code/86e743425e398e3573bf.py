import random
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def get_max_val(rows):
        return max(max(row) for row in rows) if rows else 0

    def generate_row(used_diffs, k, max_val_hint=None):
        row = [0]
        current_max = 0
        window_size = 3 
        while len(row) < k + 1:
            candidates = []
            x = row[-1] + 1
            limit = 200 if max_val_hint is None else max_val_hint + 50
            while x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    candidates.append(x)
                    if len(candidates) >= window_size:
                        break
                x += 1
            
            if not candidates:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        break
                    x += 1
                    if x > 5000:
                        return None
            
            chosen_x = candidates[0]
            row.append(chosen_x)
            used_diffs.update({chosen_x - a for a in row[:-1]})
        
        return row

    def try_shrink_row(rows, used_diffs, row_idx):
        row = rows[row_idx]
        improved = False
        for pos in range(len(row) - 1, 0, -1):
            current_val = row[pos]
            if pos == 0:
                continue
            lower_bound = row[pos - 1] + 1
            upper_bound = current_val - 1
            
            if lower_bound > upper_bound:
                continue
                
            old_diffs_involving_pos = set()
            for i in range(pos):
                old_diffs_involving_pos.add(row[pos] - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs_involving_pos.add(row[j] - row[pos])
            
            found = False
            for new_val in range(lower_bound, upper_bound + 1):
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    break
                    
                new_diffs_involving_pos = set()
                for i in range(pos):
                    new_diffs_involving_pos.add(new_val - row[i])
                for j in range(pos + 1, len(row)):
                    new_diffs_involving_pos.add(row[j] - new_val)
                
                conflict = False
                for d in new_diffs_involving_pos:
                    if d in used_diffs and d not in old_diffs_involving_pos:
                        conflict = True
                        break
                
                if not conflict:
                    used_diffs -= old_diffs_involving_pos
                    used_diffs |= new_diffs_involving_pos
                    row[pos] = new_val
                    improved = True
                    break
            
            if not improved:
                continue
            
        return improved

    def run_ils():
        best_rows = None
        best_score = float('inf')
        
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = generate_row(used_diffs, k)
            if row is None:
                return None
            rows.append(row)
        
        current_score = get_max_val(rows)
        if current_score < best_score:
            best_score = current_score
            best_rows = [r[:] for r in rows]
        
        no_improve_count = 0
        while time.time() - start_time < time_limit:
            improved = True
            while improved:
                improved = False
                for i in range(n_rows):
                    if try_shrink_row(rows, used_diffs, i):
                        improved = True
                        curr = get_max_val(rows)
                        if curr < best_score:
                            best_score = curr
                            best_rows = [r[:] for r in rows]
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
            
            if no_improve_count > 5:
                idx = random.randint(0, n_rows - 1)
                old_row = rows[idx]
                old_diffs = set()
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        old_diffs.add(old_row[j] - old_row[i])
                used_diffs -= old_diffs
                
                new_row = generate_row(used_diffs, k, max_val_hint=best_score)
                if new_row is None:
                    new_row = generate_row(used_diffs, k)
                
                if new_row:
                    rows[idx] = new_row
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    used_diffs |= new_diffs
                    no_improve_count = 0
                else:
                    rows[idx] = old_row
                    used_diffs |= old_diffs
                    break
            
            if no_improve_count > 50:
                used_diffs = set()
                rows = []
                for _ in range(n_rows):
                    row = generate_row(used_diffs, k)
                    rows.append(row)
                no_improve_count = 0
                
        return best_rows

    return run_ils()
