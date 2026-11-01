import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        diffs = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def count_conflicts(rows):
        """Count number of duplicate differences (conflicts)."""
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    diffs.append(r[j] - r[i])
        return len(diffs) - len(set(diffs))

    def greedy_init():
        """Generate a valid configuration using greedy approach."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                x = last + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
            rows.append(row)
        return rows

    def beam_search_init(beam_width=20, max_depth=None):
        """
        Beam search to find a low-scope valid configuration.
        State: list of partially-built rows.
        """
        if max_depth is None:
            max_depth = 140

        # State: (rows, used_diffs)
        # We build row by row, position by position
        # Each state is (row_idx, pos, rows, used_diffs)
        
        best_score = float('inf')
        best_rows = None
        
        # Initial state
        initial_state = (0, 1, [[0] * row_len for _ in range(n_rows)], set())
        
        beam = [initial_state]
        
        for step in range(1, 1000):
            if time.time() > deadline - 30:
                break
            if not beam:
                break
                
            new_beam = []
            for state in beam:
                row_idx, pos, rows, used_diffs = state
                
                if row_idx == n_rows:
                    # Complete solution
                    scope = get_scope(rows)
                    if scope < best_score:
                        best_score = scope
                        best_rows = [r[:] for r in rows]
                    continue
                
                if pos == row_len:
                    # Move to next row
                    new_state = (row_idx + 1, 1, rows, used_diffs)
                    new_beam.append(new_state)
                    continue
                
                prev_val = rows[row_idx][pos - 1]
                lower = prev_val + 1
                
                # Generate candidate values
                candidates = []
                for val in range(lower, min(lower + 30, max_depth + 1)):
                    valid = True
                    for i in range(pos):
                        d = val - rows[row_idx][i]
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(val)
                
                if not candidates:
                    continue
                
                # Take top few candidates (smallest values preferred)
                candidates = candidates[:5]
                
                for val in candidates:
                    new_rows = [r[:] for r in rows]
                    new_rows[row_idx][pos] = val
                    new_used = set(used_diffs)
                    new_diffs = []
                    for i in range(pos):
                        d = val - new_rows[row_idx][i]
                        new_used.add(d)
                        new_diffs.append(d)
                    
                    # Pruning: if current partial scope already exceeds best known, skip
                    current_max = max(new_rows[row_idx][:pos+1])
                    if best_rows is not None and current_max > best_score:
                        continue
                    
                    new_state = (row_idx, pos + 1, new_rows, new_used)
                    new_beam.append(new_state)
            
            if not new_beam:
                break
            
            # Sort by heuristic: prefer states with smaller current max and more progress
            def state_heuristic(s):
                row_idx, pos, rows, used_diffs = s
                if row_idx == n_rows:
                    return (0, get_scope(rows))
                # Prefer: more rows completed, then smaller current max
                current_max = max(rows[row_idx][:pos+1]) if pos > 0 else 0
                return (-(row_idx * row_len + pos), current_max)
            
            new_beam.sort(key=state_heuristic)
            beam = new_beam[:beam_width]
        
        if best_rows is not None:
            return best_rows
        return None

    def simulated_annealing(rows, time_limit):
        """
        Simulated annealing to minimize scope.
        Moves: pick a random row and position (except pos 0), change value to a nearby valid value.
        """
        end_time = time.time() + time_limit
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        temp = 10.0
        cooling_rate = 0.995
        
        while time.time() < end_time:
            if temp < 0.01:
                temp = 10.0  # Restart annealing
                current_rows = [r[:] for r in best_rows]
                current_scope = best_scope
            
            # Pick a random move
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, row_len - 1)
            
            old_val = current_rows[row_idx][pos]
            # Try a nearby value
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Check constraints
            if new_val <= current_rows[row_idx][pos - 1]:
                continue
            if pos < row_len - 1 and new_val >= current_rows[row_idx][pos + 1]:
                continue
            
            # Check if new value creates conflicts
            valid = True
            for i in range(pos):
                d = new_val - current_rows[row_idx][i]
                # Check if this difference is used elsewhere or by another element in same row
                for r_idx in range(n_rows):
                    for p_idx in range(row_len):
                        if r_idx == row_idx and p_idx >= pos:
                            continue
                        for q_idx in range(p_idx + 1, row_len):
                            if r_idx == row_idx and p_idx < pos and q_idx == pos:
                                continue  # This is the element we're changing
                            if r_idx == row_idx and q_idx < pos and p_idx == pos:
                                continue
                            if current_rows[r_idx][q_idx] - current_rows[r_idx][p_idx] == d:
                                # Check if this is from the element we're changing
                                if not (r_idx == row_idx and (p_idx == pos or q_idx == pos)):
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Calculate new scope
            new_max = max(current_rows[row_idx][:])
            other_max = max(max(r) for i, r in enumerate(current_rows) if i != row_idx)
            new_scope = max(new_max, other_max)
            
            delta_scope = new_scope - current_scope
            
            # Accept with probability based on annealing
            if delta_scope <= 0 or random.random() < math.exp(-delta_scope / temp):
                current_rows[row_idx][pos] = new_val
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            
            temp *= cooling_rate
        
        return best_rows

    def local_search_polish(rows):
        """
        Final polish: try to reduce each element by 1 if valid.
        """
        rows = [r[:] for r in rows]
        improved = True
        while improved:
            improved = False
            for row_idx in range(n_rows):
                for pos in range(row_len - 1, 0, -1):  # From right to left
                    if rows[row_idx][pos] > rows[row_idx][pos-1] + 1:
                        old_val = rows[row_idx][pos]
                        new_val = old_val - 1
                        # Check if valid
                        valid = True
                        for i in range(pos):
                            d = new_val - rows[row_idx][i]
                            for r_idx in range(n_rows):
                                for p_idx in range(row_len):
                                    for q_idx in range(p_idx + 1, row_len):
                                        if r_idx == row_idx and (p_idx == pos or q_idx == pos):
                                            continue
                                        if rows[r_idx][q_idx] - rows[r_idx][p_idx] == d:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            rows[row_idx][pos] = new_val
                            improved = True
        return rows

    # Phase 1: Get initial valid solution
    best_rows = greedy_init()
    best_scope = get_scope(best_rows)
    
    # Phase 2: Try beam search for better initial solution
    if time.time() < deadline - 40:
        beam_rows = beam_search_init(beam_width=30)
        if beam_rows is not None:
            beam_scope = get_scope(beam_rows)
            if beam_scope < best_scope:
                best_rows = beam_rows
                best_scope = beam_scope
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 20:
        sa_time = min(60.0, deadline - time.time() - 15)
        sa_rows = simulated_annealing(best_rows, sa_time)
        sa_scope = get_scope(sa_rows)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5:
        polished_rows = local_search_polish(best_rows)
        polished_scope = get_scope(polished_rows)
        if polished_scope < best_scope:
            best_rows = polished_rows
            best_scope = polished_scope
    
    return best_rows

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", get_scope(rows))
    diffs = set()
    valid = True
    for r in rows:
        if r[0] != 0:
            valid = False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    print("Valid:", valid)
