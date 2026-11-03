"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used |= diffs
        return True
    
    def greedy_build(n_rows, k, order=None):
        """Build rows greedily, optionally in a specific order."""
        if order is None:
            order = list(range(n_rows))
        used = set()
        rows = [None] * n_rows
        for idx in order:
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows[idx] = row
        return rows
    
    def local_search(rows, n_rows, k, max_iter=5000):
        """Try to reduce the maximum entry via local search."""
        rows = [list(r) for r in rows]
        current_max = get_max_entry(rows)
        best_rows = [list(r) for r in rows]
        best_max = current_max
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to reduce the last element of a random row
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) != k + 1:
                continue
            old_last = row[-1]
            if old_last == 0:
                continue
            
            # Try to reduce the last element
            for new_last in range(old_last - 1, row[-2] if len(row) > 1 else -1, -1):
                candidate = row[:-1] + [new_last]
                # Check validity of this row's differences against all used diffs
                used = set()
                valid = True
                for ri in range(n_rows):
                    r = candidate if ri == row_idx else rows[ri]
                    diffs = set()
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in used or d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                    if not valid:
                        break
                    used |= diffs
                if valid:
                    rows[row_idx] = candidate
                    new_max = get_max_entry(rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                    break
        
        return best_rows, best_max
    
    # Initial greedy solution
    best_rows = greedy_build(n_rows, k)
    best_max = get_max_entry(best_rows)
    
    # Try different row orderings for greedy
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = list(range(n_rows))
        random.shuffle(order)
        rows = greedy_build(n_rows, k, order)
        m = get_max_entry(rows)
        if m < best_max:
            best_max = m
            best_rows = rows
    
    # Local search to improve
    for attempt in range(10):
        if time.time() > deadline:
            break
        improved_rows, improved_max = local_search(best_rows, n_rows, k, max_iter=2000)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
    
    return best_rows
# EVOLVE-BLOCK-END
