import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k * (k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    
    def try_compact_rows(rows, used_set, max_iter=200):
        """Try to compact rows by removing and re-inserting elements."""
        nonlocal best_scope, best_rows
        scope = max(max(r) for r in rows)
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            improved = False
            # Pick a random row and try to shift it left
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue
            # Try to reduce the last element
            last = row[-1]
            if last <= 1:
                continue
            # Remove last element
            removed = row.pop()
            # Remove its differences from used_set
            for a in row:
                used_set.discard(removed - a)
            
            # Try to re-insert at a smaller position
            found = False
            for x in range(last - 1, 0, -1):
                if x in row:
                    continue
                valid = True
                for a in row:
                    d = abs(x - a)
                    if d == 0 or d in used_set:
                        valid = False
                        break
                if valid:
                    # Find correct position to maintain sorted order
                    pos = 0
                    while pos < len(row) and row[pos] < x:
                        pos += 1
                    row.insert(pos, x)
                    for a in row:
                        if a != x:
                            used_set.add(abs(x - a))
                    found = True
                    improved = True
                    break
            
            if not found:
                # Put it back
                row.append(removed)
                row.sort()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used_set.add(row[j] - row[i])
            
            if improved:
                new_scope = max(max(r) for r in rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
        
        return scope
    
    def greedy_construct(seed_val, strategy='compact'):
        """Construct rows using greedy with various strategies."""
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Adaptive search window based on how many elements we've placed
                elements_placed = len(row)
                if strategy == 'compact':
                    # Prefer smaller values
                    window = 30 + elements_placed * 10
                    start_offset = 1
                elif strategy == 'spread':
                    # More spread out
                    window = 50 + elements_placed * 15
                    start_offset = 1
                else:  # 'random'
                    window = 40 + elements_placed * 12
                    start_offset = 1
                
                search_end = min(last + window, lb + 200)
                
                # Generate candidates
                candidates = list(range(last + start_offset, search_end + 1))
                
                # Apply randomization based on strategy
                if strategy == 'compact':
                    # Bias toward smaller values
                    random.shuffle(candidates)
                    candidates.sort(key=lambda x: x + random.random() * 3)
                elif strategy == 'spread':
                    random.shuffle(candidates)
                else:
                    # Random mix
                    if random.random() < 0.5:
                        candidates.sort(key=lambda x: x + random.random() * 10)
                    else:
                        random.shuffle(candidates)
                
                for x in candidates:
                    # Check if x creates any used differences
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    # Wider search
                    for x in range(last + 1, lb + 500):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows.append(row)
        
        scope = max(max(r) for r in rows)
        return rows, scope
    
    # Strategy 1: Standard greedy with compaction
    num_attempts = 300
    strategies = ['compact', 'spread', 'random']
    
    for i in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        random.seed(i * 7919 + 13)
        strategy = strategies[i % len(strategies)]
        
        rows, scope = greedy_construct(i, strategy)
        
        if rows is not None:
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
            
            # Try to compact
            if scope <= best_scope:
                used_set = set()
                for r in rows:
                    for idx in range(len(r)):
                        for jdx in range(idx + 1, len(r)):
                            used_set.add(r[jdx] - r[idx])
                try_compact_rows(rows, used_set, max_iter=50)
            
            if best_scope == lb:
                break
    
    # Strategy 2: Build rows in random order
    for i in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(100000 + i * 104729)
        
        used = set()
        rows = [None] * n_rows
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        success = True
        for row_idx in row_order:
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Randomized search
                window = 40 + len(row) * 12
                search_end = min(last + window, lb + 300)
                
                candidates = list(range(last + 1, search_end + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    for x in range(last + 1, lb + 600):
                        valid = True
                        new_diffs = []
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.append(d)
                        
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        success = False
                        break
            
            if not success:
                break
            rows[row_idx] = row
        
        if success and all(r is not None for r in rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
            
            # Try compaction
            if scope <= best_scope:
                used_set = set()
                for r in rows:
                    for idx in range(len(r)):
                        for jdx in range(idx + 1, len(r)):
                            used_set.add(r[jdx] - r[idx])
                try_compact_rows(rows, used_set, max_iter=50)
            
            if best_scope == lb:
                break
    
    # Final attempt: if we have a good solution, try more aggressive compaction
    if best_rows is not None and best_scope > lb:
        used_set = set()
        for r in best_rows:
            for idx in range(len(r)):
                for jdx in range(idx + 1, len(r)):
                    used_set.add(r[jdx] - r[idx])
        try_compact_rows(best_rows, used_set, max_iter=500)
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    x += 1
                    valid = True
                    for a in row:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                row.append(x)
                used.update({x - a for a in row if a != x})
            best_rows.append(row)
    
    return best_rows
