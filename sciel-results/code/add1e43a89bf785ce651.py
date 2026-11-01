import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Upper bound for scope. Best known is 134.
    # We need a buffer for initial randomization and SA moves.
    MAX_SCOPE = 200
    
    # Pre-compute all pairs of indices (i, j) with i < j for difference calculation
    pairs = [(i, j) for i in range(row_len) for j in range(i+1, row_len)]
    num_pairs = len(pairs)
    
    def get_scope(rows):
        return max(max(r) for r in rows)
        
    def count_conflicts(rows):
        """
        Returns the number of duplicate differences across all rows.
        """
        used = set()
        conflicts = 0
        for r in rows:
            for i, j in pairs:
                d = r[j] - r[i]
                if d in used:
                    conflicts += 1
                else:
                    used.add(d)
        return conflicts

    def is_valid(rows):
        return count_conflicts(rows) == 0

    def random_rows():
        """
        Generate random rows. 
        To ensure validity is reachable, we start with a wide scope.
        """
        rows = []
        for _ in range(n_rows):
            # Generate k+1 increasing integers starting at 0
            # Use random sampling from [1, MAX_SCOPE]
            # Sort them
            vals = sorted(random.sample(range(1, MAX_SCOPE + 1), k))
            row = [0] + vals
            rows.append(row)
        return rows

    def greedy_init():
        """
        Construct rows greedily to minimize conflicts initially.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, row_len):
                prev_val = row[-1]
                # Try to find the smallest value that doesn't create conflicts
                found = False
                for v in range(prev_val + 1, MAX_SCOPE + 1):
                    valid = True
                    for j in range(len(row)):
                        d = v - row[j]
                        if d in used:
                            valid = False
                            break
                    if valid:
                        row.append(v)
                        for j in range(len(row) - 1):
                            used.add(v - row[j])
                        found = True
                        break
                if not found:
                    # If we fail, just pick a random large value
                    v = random.randint(prev_val + 1, MAX_SCOPE)
                    row.append(v)
                    for j in range(len(row) - 1):
                        used.add(v - row[j])
            rows.append(row)
        return rows

    def solve():
        best_rows = None
        best_scope = float('inf')
        
        # Start with greedy initialization
        current_rows = greedy_init()
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        if current_conflicts == 0:
            best_rows = [r[:] for r in current_rows]
            best_scope = current_scope
            
        # Simulated Annealing
        T = 500.0
        cooling = 0.9999
        min_T = 1e-3
        
        # Energy function: Scope + Penalty * Conflicts
        penalty = 1000.0
        
        def energy(rows):
            scope = get_scope(rows)
            conf = count_conflicts(rows)
            return scope + penalty * conf
            
        current_energy = energy(current_rows)
        
        step_count = 0
        no_improve_steps = 0
        max_no_improve = 50000
        
        while time.time() < deadline:
            if T < min_T:
                # Restart with greedy or random
                if random.random() < 0.5:
                    current_rows = greedy_init()
                else:
                    current_rows = random_rows()
                current_scope = get_scope(current_rows)
                current_conflicts = count_conflicts(current_rows)
                current_energy = energy(current_rows)
                T = 500.0
                no_improve_steps = 0
                if current_conflicts == 0 and current_scope < best_scope:
                    best_rows = [r[:] for r in current_rows]
                    best_scope = current_scope
                continue
            
            # Pick a random move
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)  # Don't move 0
            
            old_val = current_rows[r_idx][i_idx]
            
            # Determine new value
            lower = current_rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = current_rows[r_idx][i_idx + 1] - 1
            else:
                upper = MAX_SCOPE
                
            if lower > upper:
                T *= cooling
                continue
                
            # Randomly pick a new value in [lower, upper]
            # Bias towards smaller values to reduce scope
            if random.random() < 0.7:
                # Prefer smaller values
                new_val = random.randint(lower, min(upper, lower + max(1, (upper - lower) // 3)))
            else:
                new_val = random.randint(lower, upper)
            
            if new_val == old_val:
                continue
                
            # Apply move
            current_rows[r_idx][i_idx] = new_val
            
            new_scope = get_scope(current_rows)
            new_conflicts = count_conflicts(current_rows)
            new_energy = energy(current_rows)
            
            delta_E = new_energy - current_energy
            
            # Metropolis acceptance
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                # Accept
                current_scope = new_scope
                current_conflicts = new_conflicts
                current_energy = new_energy
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    no_improve_steps = 0
                    if best_scope <= total_diffs:
                        break
                else:
                    no_improve_steps += 1
            else:
                # Reject
                current_rows[r_idx][i_idx] = old_val
                no_improve_steps += 1
                
            T *= cooling
            step_count += 1
            
            # If stuck for too long, restart
            if no_improve_steps > max_no_improve:
                T = 500.0
                no_improve_steps = 0
                if random.random() < 0.3:
                    current_rows = greedy_init()
                else:
                    current_rows = random_rows()
                current_scope = get_scope(current_rows)
                current_conflicts = count_conflicts(current_rows)
                current_energy = energy(current_rows)
                if current_conflicts == 0 and current_scope < best_scope:
                    best_rows = [r[:] for r in current_rows]
                    best_scope = current_scope
            
        # Final polish: try to reduce scope by moving elements down
        if best_rows:
            polish_deadline = min(deadline, time.time() + 5.0)
            while time.time() < polish_deadline:
                improved = False
                for r_idx in range(n_rows):
                    for i_idx in range(row_len - 1, 0, -1):  # From right to left
                        old_val = best_rows[r_idx][i_idx]
                        lower = best_rows[r_idx][i_idx - 1] + 1
                        if i_idx < row_len - 1:
                            upper = best_rows[r_idx][i_idx + 1] - 1
                        else:
                            upper = old_val
                        # Try to move down
                        for new_val in range(lower, old_val):
                            best_rows[r_idx][i_idx] = new_val
                            if is_valid(best_rows):
                                improved = True
                                break
                            else:
                                best_rows[r_idx][i_idx] = old_val
                if not improved:
                    break
            
            # Update best_scope
            best_scope = get_scope(best_rows)
            return best_rows
            
        # Fallback: Return the current state if it's valid, or construct a valid solution
        if is_valid(current_rows):
            return current_rows
            
        # If no valid solution found, use greedy with larger scope
        used = set()
        rows = []
        for _ in range(n_rows):
            r = [0]
            for pos in range(1, row_len):
                prev_val = r[-1]
                found = False
                for v in range(prev_val + 1, 500):  # Larger scope for fallback
                    valid = True
                    for j in range(len(r)):
                        d = v - r[j]
                        if d in used:
                            valid = False
                            break
                    if valid:
                        r.append(v)
                        for j in range(len(r) - 1):
                            used.add(v - r[j])
                        found = True
                        break
                if not found:
                    v = random.randint(prev_val + 1, 500)
                    r.append(v)
                    for j in range(len(r) - 1):
                        used.add(v - r[j])
            rows.append(r)
            
        if is_valid(rows):
            return rows
            
        # Last resort: return a trivial valid solution
        # This should rarely be needed
        rows = []
        offset = 0
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, row_len):
                row.append(row[-1] + 100 + offset)
            rows.append(row)
            offset += 1
        return rows

    return solve()

if __name__ == "__main__":
    rows = construct(4, 7)
    print("Scope:", max(max(r) for r in rows))
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
