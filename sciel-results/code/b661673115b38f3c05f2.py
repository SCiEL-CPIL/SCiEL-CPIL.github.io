"""New design: greedy init + simulated annealing with row-replacement and swap moves."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def get_all_diffs(rows):
        """Get all differences across all rows."""
        all_d = set()
        for row in rows:
            all_d |= get_diffs(row)
        return all_d
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_d = set()
        for row in rows:
            d = get_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True
    
    def greedy_row(used_diffs, k, start=0):
        """Greedy: build a row starting at 0 with k+1 entries, avoiding used_diffs."""
        row = [0]
        x = start
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used_diffs):
                row.append(x)
                used_diffs |= new
        return row
    
    def greedy_init(n_rows, k):
        """Build initial solution greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def compute_score(rows):
        """Score = largest entry (lower is better)."""
        return max(max(row) for row in rows)
    
    # Initial greedy solution
    rows = greedy_init(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_score = compute_score(best_rows)
    
    # Simulated annealing
    temp = max(10, best_score * 0.1)
    cooling = 0.9995
    min_temp = 1.0
    
    random.seed(42)
    
    while time.time() < deadline and temp > min_temp:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace one row entirely with a new greedy row
            idx = random.randint(0, n_rows - 1)
            old_row = rows[idx]
            old_diffs = get_diffs(old_row)
            
            # Remove old row's diffs
            other_diffs = set()
            for i, r in enumerate(rows):
                if i != idx:
                    other_diffs |= get_diffs(r)
            
            # Try to build a new row avoiding other_diffs
            new_row = [0]
            x = 0
            success = False
            for _ in range(200):  # limit attempts
                x += 1
                new = {x - a for a in new_row}
                if not (new & other_diffs) and not (new & old_diffs if False else False):
                    # Check against other rows only
                    if not (new & other_diffs):
                        new_row.append(x)
                        if len(new_row) == k + 1:
                            success = True
                            break
            if success:
                rows[idx] = new_row
                new_score = compute_score(rows)
                delta = new_score - best_score
                if delta < 0 or random.random() < _exp_neg(delta / temp):
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [list(r) for r in rows]
                else:
                    rows[idx] = old_row
        elif move_type < 0.7:
            # Swap two elements within a single row (maintain increasing order)
            idx = random.randint(0, n_rows - 1)
            row = rows[idx]
            if len(row) >= 3:
                i, j = random.sample(range(1, len(row)), 2)
                if i > j:
                    i, j = j, i
                # Swap and re-sort (but keep 0 at start)
                row[i], row[j] = row[j], row[i]
                row = [0] + sorted(row[1:])
                # Check validity
                if is_valid(rows):
                    new_score = compute_score(rows)
                    delta = new_score - best_score
                    if delta < 0 or random.random() < _exp_neg(delta / temp):
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [list(r) for r in rows]
                    else:
                        rows[idx] = [0] + sorted(old_row[1:])
                else:
                    rows[idx] = [0] + sorted(old_row[1:])
        else:
            # Perturb: change one non-zero element of a row
            idx = random.randint(0, n_rows - 1)
            old_row = list(rows[idx])
            pos = random.randint(1, len(old_row) - 1)
            delta_val = random.choice([-2, -1, 1, 2, 3, -3])
            new_val = old_row[pos] + delta_val
            if new_val > old_row[pos - 1] and (pos == len(old_row) - 1 or new_val < old_row[pos + 1]):
                rows[idx][pos] = new_val
                if is_valid(rows):
                    new_score = compute_score(rows)
                    delta = new_score - best_score
                    if delta < 0 or random.random() < _exp_neg(delta / temp):
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [list(r) for r in rows]
                    else:
                        rows[idx] = old_row
                else:
                    rows[idx] = old_row
        
        temp *= cooling
    
    # Final greedy refinement: try to tighten each row
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(n_rows):
            old_row = list(best_rows[idx])
            old_diffs = get_diffs(old_row)
            other_diffs = set()
            for i, r in enumerate(best_rows):
                if i != idx:
                    other_diffs |= get_diffs(r)
            
            # Try to rebuild row more tightly
            new_row = [0]
            x = 0
            while len(new_row) < k + 1 and x < max(old_row[-1], 200):
                x += 1
                new = {x - a for a in new_row}
                if not (new & other_diffs):
                    new_row.append(x)
            if len(new_row) == k + 1 and new_row[-1] < old_row[-1]:
                best_rows[idx] = new_row
                improved = True
        if not improved:
            break
    
    return best_rows

def _exp_neg(x):
    """Safe exp(-x) for x >= 0."""
    if x > 700:
        return 0.0
    import math
    return math.exp(-x)
# EVOLVE-BLOCK-END
