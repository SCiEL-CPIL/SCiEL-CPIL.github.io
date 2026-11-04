"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard: kill before 150s wall-clock limit
    deadline = time.time() + 142.0

    def max_entry(rows):
        return max(max(row) for row in rows)

    def compute_cost(rows):
        """Cost = violations * 10000 + max_entry. Lower is better."""
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.append(row[j] - row[i])
        counts = {}
        for d in diffs:
            counts[d] = counts.get(d, 0) + 1
        violations = sum(c - 1 for c in counts.values() if c > 1)
        m = max_entry(rows)
        return violations * 10000 + m

    # --- Phase 1: Greedy initialization ---
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max_entry(best_rows)

    # --- Phase 2: Row-by-row DFS with bitset pruning ---
    # Key structural idea: build rows one at a time. For each row, do a DFS
    # that only considers values whose differences don't conflict with
    # already-placed rows. This is much more directed than global SA.
    
    def build_row_dfs(row_idx, pos_idx, used_diffs_bitset, current_row, max_scope):
        """
        Recursively fill current_row[pos_idx..k] given that differences
        in used_diffs_bitset are already taken by previous rows.
        Returns True on success (current_row is modified in place).
        """
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            return True
        
        if pos_idx == k + 1:
            # Row complete, move to next row
            return build_row_dfs(row_idx + 1, 1, used_diffs_bitset, [0], max_scope)
        
        prev_val = current_row[pos_idx - 1]
        
        # Try candidate values
        for x in range(prev_val + 1, max_scope + 1):
            if time.time() > deadline:
                return False
            
            # Check conflicts with used_diffs_bitset
            conflict = False
            new_bits = 0
            for i in range(pos_idx):
                d = x - current_row[i]
                if (used_diffs_bitset >> d) & 1:
                    conflict = True
                    break
                new_bits |= (1 << d)
            
            if conflict:
                continue
            
            # Also need to ensure remaining positions can be filled
            # We need k - pos_idx more entries, each >= prev_val + 1
            # The last entry must be <= max_scope
            # Minimum last entry: x + (k - pos_idx)
            if x + (k - pos_idx) > max_scope:
                break  # No point trying larger x
            
            current_row.append(x)
            
            if build_row_dfs(row_idx, pos_idx + 1, used_diffs_bitset | new_bits, current_row, max_scope):
                return True
            
            current_row.pop()
        
        return False

    def try_build_all_rows(max_scope, timeout=10.0):
        """Try to build all n_rows rows with max entry <= max_scope."""
        start_time = time.time()
        
        # We need to track used_diffs across rows
        # Use a list to hold the built rows
        built_rows = []
        
        def dfs_rows(row_idx, used_diffs_bitset):
            if time.time() - start_time > timeout:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Build this row
            row = [0]
            # Try different starting strategies
            for first_val in range(1, min(max_scope // k, 20) + 1):
                if time.time() - start_time > timeout:
                    return False
                
                row = [0, first_val]
                # Check if first_val's diff conflicts
                if (used_diffs_bitset >> first_val) & 1:
                    continue
                
                new_bits = 1 << first_val
                
                if dfs_rows_row(1, row, used_diffs_bitset | new_bits):
                    built_rows.append(row[:])
                    if dfs_rows(row_idx + 1, used_diffs_bitset | new_bits | _get_row_bits(row)):
                        return True
                    built_rows.pop()
            
            return False
        
        def _get_row_bits(row):
            bits = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    bits |= (1 << (row[j] - row[i]))
            return bits
        
        def dfs_rows_row(pos_idx, row, used_diffs_bitset):
            if time.time() - start_time > timeout:
                return False
            
            if pos_idx == k + 1:
                return True
            
            prev_val = row[pos_idx - 1]
            
            for x in range(prev_val + 1, max_scope + 1):
                if time.time() - start_time > timeout:
                    return False
                
                conflict = False
                new_bits = 0
                for i in range(pos_idx):
                    d = x - row[i]
                    if (used_diffs_bitset >> d) & 1:
                        conflict = True
                        break
                    new_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Feasibility: need k - pos_idx more entries
                if x + (k - pos_idx) > max_scope:
                    break
                
                row.append(x)
                if dfs_rows_row(pos_idx + 1, row, used_diffs_bitset | new_bits):
                    return True
                row.pop()
            
            return False
        
        # Simpler approach: just try to build all rows sequentially
        def build_all():
            used_bits = 0
            rows = []
            for r in range(n_rows):
                row = [0]
                if not fill_row(r, row, used_bits):
                    return None
                # Update used_bits
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used_bits |= (1 << (row[j] - row[i]))
                rows.append(row)
            return rows
        
        def fill_row(r, row, used_bits):
            """Fill row starting from [0] using DFS."""
            if len(row) > k + 1:
                return False
            pos = len(row) - 1  # last filled position
            if pos == k:
                return True
            
            prev_val = row[pos]
            
            for x in range(prev_val + 1, max_scope + 1):
                if time.time() - start_time > timeout:
                    return False
                
                conflict = False
                for i in range(pos + 1):
                    d = x - row[i]
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                if x + (k - pos - 1) > max_scope:
                    break
                
                row.append(x)
                if fill_row(r, row, used_bits):
                    return True
                row.pop()
            
            return False
        
        result = build_all()
        if result is not None:
            return result
        return None

    # --- Phase 3: Iterative deepening on scope ---
    # Try to find solutions with decreasing max_scope
    search_scope = best_max - 1
    min_scope = max(100, best_max - 30)
    
    found_improvement = False
    while search_scope >= min_scope and time.time() < deadline:
        timeout = min(5.0, deadline - time.time() - 2.0)
        if timeout <= 1.0:
            break
        
        result = try_build_all_rows(search_scope, timeout=timeout)
        if result is not None:
            best_rows = result
            best_max = search_scope
            found_improvement = True
            # Try to go lower
            search_scope -= 1
        else:
            # Try SA to potentially find a better starting point
            break

    # --- Phase 4: Simulated Annealing polish ---
    # Use SA to fine-tune and potentially find even better solutions
    current_rows = [r[:] for r in best_rows]
    current_cost = compute_cost(current_rows)
    best_max_sa = best_max
    best_rows_sa = [r[:] for r in best_rows]

    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    T = 30.0
    T_min = 0.01
    cooling = 0.99998
    stuck_count = 0
    max_stuck = 5000

    while time.time() < deadline - 3.0:
        if T < T_min:
            break
        
        if stuck_count > max_stuck:
            # Restart with higher temperature
            T = 20.0
            stuck_count = 0
            # Random restart: perturb current solution
            for _ in range(5):
                r, i = random.choice(indices)
                old_val = current_rows[r][i]
                lower = current_rows[r][i - 1] + 1
                upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 20
                if upper > lower:
                    current_rows[r][i] = random.randint(lower, upper)
            # Fix ordering
            for r in range(n_rows):
                row = current_rows[r]
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        row[i] = row[i-1] + 1
            current_cost = compute_cost(current_rows)
            continue
        
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 30
        
        # Bias towards smaller values
        if random.random() < 0.5:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.3)
        else:
            new_val = random.randint(lower, upper)
        
        if new_val == old_val:
            continue
        
        current_rows[r][i] = new_val
        
        row = current_rows[r]
        if any(row[j] <= row[j-1] for j in range(1, len(row))):
            current_rows[r][i] = old_val
            continue
        
        new_cost = compute_cost(current_rows)
        
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current_cost = new_cost
            new_max = max_entry(current_rows)
            if new_max < best_max_sa:
                best_max_sa = new_max
                best_rows_sa = [rr[:] for rr in current_rows]
                stuck_count = 0
            else:
                stuck_count += 1
        else:
            current_rows[r][i] = old_val
            stuck_count += 1
        
        T *= cooling

    if best_max_sa < best_max:
        best_rows = best_rows_sa
        best_max = best_max_sa

    return best_rows
# EVOLVE-BLOCK-END
