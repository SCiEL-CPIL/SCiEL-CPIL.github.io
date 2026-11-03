import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety

    def greedy_init(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                    row.append(x)
                    local_diffs.update(new_diffs)
                x += 1
            rows.append(row)
            used_diffs.update(local_diffs)
        return rows, used_diffs

    def generate_row(used_diffs, k, max_val_hint):
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            candidates = []
            start = row[-1] + 1
            # Scan for valid candidates
            # We want small x, so we scan from start upwards
            # Limit search to max_val_hint
            for x in range(start, max_val_hint + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                if new_diffs & local_diffs:
                    continue
                candidates.append(x)
                # Early exit if we have enough small candidates
                if len(candidates) >= 20:
                    break
            
            if not candidates:
                return None
            
            # Choose a candidate
            # To allow exploration, pick randomly from candidates
            # But bias towards smaller ones by picking from the first half?
            # Or just random.choice.
            x = random.choice(candidates)
            
            row.append(x)
            # Update local_diffs
            for a in row[:-1]:
                local_diffs.add(x - a)
        
        return row

    # 1. Initialize
    rows, used_diffs = greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    current_scope = best_scope

    # 2. Simulated Annealing
    T = 5.0  # Initial temperature
    cooling = 0.9995
    min_T = 0.1
    max_val_hint = best_scope + 50 # Allow some room for random generation

    while time.time() < deadline and T > min_T:
        # Pick a random row to rebuild
        i = random.randint(0, n_rows - 1)
        
        old_row = rows[i]
        
        # Calculate old diffs
        old_diffs = set()
        for a in range(len(old_row)):
            for b in range(a + 1, len(old_row)):
                old_diffs.add(old_row[b] - old_row[a])
        
        # Remove old diffs from global set
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Generate new row
        # Use current best scope + buffer as hint to encourage smaller/equal scope
        # If we allow larger, we might escape local minima, but we want to minimize scope.
        # Let's try to generate rows with scope <= current_scope first.
        # If that fails, try larger.
        
        new_row = generate_row(used_diffs, k, current_scope)
        
        if new_row is None:
            # Try with a larger hint
            new_row = generate_row(used_diffs, k, current_scope + 20)
            
        if new_row is None:
            # Restore old row
            used_diffs.update(old_diffs)
            T *= cooling
            continue
            
        # Calculate new diffs
        new_diffs = set()
        for a in range(len(new_row)):
            for b in range(a + 1, len(new_row)):
                new_diffs.add(new_row[b] - new_row[a])
                
        # Check conflicts with remaining rows
        if new_diffs & used_diffs:
            # Conflict, restore
            used_diffs.update(old_diffs)
            T *= cooling
            continue
            
        # No conflict. Calculate new scope
        rows[i] = new_row
        new_scope = max(max(r) for r in rows)
        
        # Acceptance criterion
        delta = new_scope - current_scope
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_scope = new_scope
            used_diffs.update(new_diffs)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
        else:
            # Reject
            rows[i] = old_row
            used_diffs.update(old_diffs)
            
        T *= cooling

    # 3. Final Polish: Try to reduce scope further with deterministic local search
    # This might not be necessary if SA is good, but adds safety.
    # Simple hill climbing on scope
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_rows):
            # Try to rebuild row i with a tighter bound
            old_row = rows[i]
            old_diffs = set()
            for a in range(len(old_row)):
                for b in range(a + 1, len(old_row)):
                    old_diffs.add(old_row[b] - old_row[a])
            for d in old_diffs:
                used_diffs.discard(d)
                
            # Try to find a row with max element < current max of this row
            target_max = max(old_row) - 1
            new_row = generate_row(used_diffs, k, target_max)
            
            if new_row is not None:
                new_diffs = set()
                for a in range(len(new_row)):
                    for b in range(a + 1, len(new_row)):
                        new_diffs.add(new_row[b] - new_row[a])
                        
                if not (new_diffs & used_diffs):
                    # Valid and smaller
                    rows[i] = new_row
                    used_diffs.update(new_diffs)
                    improved = True
                    if max(max(r) for r in rows) < best_scope:
                        best_scope = max(max(r) for r in rows)
                        best_rows = [r[:] for r in rows]
            else:
                # Restore
                used_diffs.update(old_diffs)

    return best_rows
