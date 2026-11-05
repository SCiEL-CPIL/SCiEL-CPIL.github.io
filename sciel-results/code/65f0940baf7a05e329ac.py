import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140.0
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def get_diffs(rows):
        """Get all differences and count them."""
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    def violation_count(counts):
        return sum(c - 1 for c in counts.values() if c > 1)
    
    def greedy_init():
        """Deterministic greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used and len(new) == len(row):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    def randomized_greedy():
        """Randomized greedy with exploration."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                candidates = []
                x = row[-1] + 1
                bound = max(row[-1] * 2 + 100, 300)
                while x <= bound:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used and len(new_diffs) == len(row):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Fallback: find first valid
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used and len(new_diffs) == len(row):
                            candidates = [x]
                            break
                        x += 1
                
                # Randomized selection
                if len(candidates) > 1:
                    if random.random() < 0.6:
                        top_n = min(5, len(candidates))
                        val = candidates[random.randint(0, top_n - 1)]
                    else:
                        val = random.choice(candidates)
                else:
                    val = candidates[0]
                
                row.append(val)
                used.update({val - a for a in row[:-1]})
            rows.append(row)
        return rows
    
    def min_conflict_sa(rows, time_limit=60.0):
        """
        Min-conflict style local search with simulated annealing.
        Focus on reducing the maximum entry while maintaining validity.
        """
        current = [r[:] for r in rows]
        current_max = max_entry(current)
        
        # Precompute difference counts
        diff_counts = get_diffs(current)
        current_violations = violation_count(diff_counts)
        
        # Temperature schedule
        T = 50.0
        T_min = 0.01
        cooling_rate = 0.99995
        start_time = time.time()
        
        # Positions to modify: (row_idx, col_idx) where col_idx >= 1
        positions = [(r, c) for r in range(n_rows) for c in range(1, k + 1)]
        
        best_rows = [r[:] for r in current]
        best_max = current_max
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
            
            # Pick a random position
            r, c = random.choice(positions)
            row = current[r]
            old_val = row[c]
            
            # Determine valid range for new value
            lower = row[c - 1] + 1
            upper = row[c + 1] - 1 if c < k else old_val + 50
            
            # Generate candidate new value
            if random.random() < 0.5:
                # Try to reduce
                span = max(1, old_val - lower)
                new_val = lower + int(random.random() * span * 0.8)
                if new_val >= old_val:
                    new_val = old_val - 1
            else:
                # Random nearby
                new_val = random.randint(lower, min(upper, old_val + 20))
            
            if new_val == old_val or new_val < lower or (c < k and new_val > upper):
                continue
            
            # Compute old and new differences
            old_diffs = []
            new_diffs = []
            for j in range(len(row)):
                if j == c:
                    continue
                if j < c:
                    old_diffs.append(old_val - row[j])
                    new_diffs.append(new_val - row[j])
                else:
                    old_diffs.append(row[j] - old_val)
                    new_diffs.append(row[j] - new_val)
            
            # Check for conflicts in new diffs
            temp_counts = dict(diff_counts)
            for d in old_diffs:
                temp_counts[d] -= 1
                if temp_counts[d] == 0:
                    del temp_counts[d]
            
            conflict = False
            for d in new_diffs:
                if d in temp_counts:
                    conflict = True
                    break
                temp_counts[d] = temp_counts.get(d, 0) + 1
            
            if conflict:
                # Reject this move
                continue
            
            # Compute new max
            new_row = row[:]
            new_row[c] = new_val
            new_max = max(max(row) if ri != r else max(new_row)
                         for ri, row in enumerate(current))
            
            # Acceptance criterion
            delta_max = new_max - current_max
            if delta_max <= 0:
                # Accept if no worse
                # Update state
                for d in old_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                current[r][c] = new_val
                current_max = new_max
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [row[:] for row in current]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.718281828, -delta_max / T):
                    for d in old_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in new_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    current[r][c] = new_val
                    current_max = new_max
            
            T *= cooling_rate
        
        return best_rows, best_max
    
    def hill_climbing(rows, time_limit=30.0):
        """Simple hill climbing to reduce max entry."""
        current = [r[:] for r in rows]
        current_max = max_entry(current)
        
        diff_counts = get_diffs(current)
        positions = [(r, c) for r in range(n_rows) for c in range(1, k + 1)]
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            random.shuffle(positions)
            
            for r, c in positions:
                if time.time() - start_time >= time_limit:
                    break
                
                row = current[r]
                old_val = row[c]
                lower = row[c - 1] + 1
                upper = row[c + 1] - 1 if c < k else old_val
                
                # Try to find a smaller valid value
                for new_val in range(lower, old_val):
                    old_diffs = []
                    new_diffs = []
                    for j in range(len(row)):
                        if j == c:
                            continue
                        if j < c:
                            old_diffs.append(old_val - row[j])
                            new_diffs.append(new_val - row[j])
                        else:
                            old_diffs.append(row[j] - old_val)
                            new_diffs.append(row[j] - new_val)
                    
                    # Check conflicts
                    temp_counts = dict(diff_counts)
                    for d in old_diffs:
                        temp_counts[d] -= 1
                        if temp_counts[d] == 0:
                            del temp_counts[d]
                    
                    conflict = False
                    for d in new_diffs:
                        if d in temp_counts:
                            conflict = True
                            break
                        temp_counts[d] = temp_counts.get(d, 0) + 1
                    
                    if not conflict:
                        # Accept
                        for d in old_diffs:
                            diff_counts[d] -= 1
                            if diff_counts[d] == 0:
                                del diff_counts[d]
                        for d in new_diffs:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        current[r][c] = new_val
                        current_max = max(max(row) for row in current)
                        improved = True
                        break
        
        return current, current_max
    
    # === Main strategy ===
    # Phase 1: Generate diverse initial solutions
    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    # Try multiple randomized starts
    n_restarts = 12
    per_restart_time = 8.0
    start_phase = time.time()
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        rows = randomized_greedy()
        cur_max = max_entry(rows)
        
        # Only proceed if competitive
        if cur_max <= best_max + 30:
            # Run hill climbing
            hc_time = min(5.0, deadline - time.time() - 15)
            if hc_time > 1.0:
                refined, refined_max = hill_climbing(rows, time_limit=hc_time)
                if refined_max < best_max and is_valid(refined):
                    best_rows = refined
                    best_max = refined_max
    
    # Phase 2: SA polish on best
    remaining = deadline - time.time() - 10
    if remaining > 5.0:
        polished, polished_max = min_conflict_sa(best_rows, time_limit=min(remaining, 60.0))
        if polished_max < best_max and is_valid(polished):
            best_rows = polished
            best_max = polished_max
    
    # Final validation
    if not is_valid(best_rows):
        best_rows = greedy_init()
    
    return best_rows
