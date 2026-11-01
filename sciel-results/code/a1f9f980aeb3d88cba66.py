"""Stronger algorithm for difference triangle set construction."""
import random
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    best_rows = None
    best_max = float('inf')
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Check if all pairwise differences within rows are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_construct(n_rows, k, randomize=False):
        """Construct rows using greedy with optional randomization."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if randomize:
                    # Try a random range of next values
                    candidates = sorted(random.sample(range(1, 200), min(50, 199)))
                    found = False
                    for c in candidates:
                        new_diffs = {c - a for a in row}
                        if not new_diffs & used:
                            row.append(c)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        # Fall back to sequential
                        x = max(row)
                        while True:
                            x += 1
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                row.append(x)
                                used |= new_diffs
                                break
                else:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, time_budget):
        """Simulated annealing local search to reduce max entry."""
        current = [list(r) for r in rows]
        current_max = get_max_entry(current)
        best = [list(r) for r in current]
        best_max = current_max
        
        used = set()
        for row in current:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        def compute_used(rows):
            s = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        s.add(row[j] - row[i])
            return s
        
        temp = 10.0
        cooling = 0.995
        no_improve = 0
        
        while time.time() < time_budget and temp > 0.1:
            # Pick a random row and entry (not index 0)
            r_idx = random.randrange(n_rows)
            pos = random.randrange(1, k + 1)
            
            old_val = current[r_idx][pos]
            # Try to move to a smaller value
            new_val = random.randrange(1, old_val)
            
            if new_val == old_val:
                continue
            
            # Check if new value maintains sorted order
            if pos > 0 and new_val <= current[r_idx][pos - 1]:
                continue
            if pos < k and new_val >= current[r_idx][pos + 1]:
                continue
            
            # Compute differences that change
            row = current[r_idx]
            # Old differences involving position pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, k + 1):
                old_diffs.add(row[j] - old_val)
            
            # New differences involving position pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, k + 1):
                new_diffs.add(row[j] - new_val)
            
            # Check if new_diffs are all unused (after removing old_diffs)
            available = used - old_diffs
            if new_diffs & available:
                continue
            
            # Accept move
            used = (used - old_diffs) | new_diffs
            current[r_idx][pos] = new_val
            
            new_max = get_max_entry(current)
            if new_max < best_max:
                best_max = new_max
                best = [list(r) for r in current]
                no_improve = 0
            else:
                no_improve += 1
                # Accept with probability based on temperature
                delta = new_max - best_max
                if random.random() < min(1.0, temp / max(1, delta)):
                    pass  # Already accepted
                else:
                    # Reject
                    current[r_idx][pos] = old_val
                    used = (used - new_diffs) | old_diffs
            
            temp *= cooling
        
        return best, best_max
    
    # Initial greedy construction
    initial = greedy_construct(n_rows, k, randomize=False)
    if is_valid(initial):
        best_rows = initial
        best_max = get_max_entry(initial)
    
    # Try multiple randomized restarts and local search
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Randomized greedy start
        cand = greedy_construct(n_rows, k, randomize=True)
        if not is_valid(cand):
            continue
        
        cand_max = get_max_entry(cand)
        
        # Local search
        time_budget = min(deadline - 5, time.time() + 5)
        improved, improved_max = local_search(cand, n_rows, k, time_budget)
        
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved
        
        # If we've reached a good value, do more local search on best
        if best_max <= 115 and time.time() < deadline - 20:
            time_budget = min(deadline - 5, time.time() + 10)
            improved, improved_max = local_search(best_rows, n_rows, k, time_budget)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved
    
    if best_rows is None:
        best_rows = initial
    
    return best_rows
# EVOLVE-BLOCK-END
