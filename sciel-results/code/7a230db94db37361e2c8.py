import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def row_to_mask(row):
        """Convert a row to a bitmask of its differences."""
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    mask |= (1 << (d - 1))
        return mask

    def is_valid_mask(mask):
        """Check if mask has no duplicate differences (popcount check)."""
        # We need exactly total_diffs bits set
        return bin(mask).count('1') == total_diffs

    def greedy_construct_row(used_mask, rng, max_window):
        """
        Greedily construct a single row given a used_mask of forbidden differences.
        Returns the row if successful, else None.
        """
        row = [0]
        local_mask = 0
        current_max = 0
        
        for _ in range(k): # Need k more elements
            # Find valid candidates in a window
            candidates = []
            start_val = current_max + 1
            # Dynamic window size: start small, expand if stuck
            # For DTS(4,7), scope ~134, 8 elements. Avg gap ~16.
            # Window of 30-50 is usually sufficient.
            end_val = current_max + max_window
            
            for x in range(start_val, end_val + 1):
                valid = True
                new_local_diffs = 0
                for a in row:
                    d = x - a
                    if d <= 0: 
                        continue
                    bit = 1 << (d - 1)
                    # Check against global used mask
                    if (used_mask & bit):
                        valid = False
                        break
                    # Check against local mask (diffs within this row so far)
                    if (local_mask & bit):
                        valid = False
                        break
                    new_local_diffs |= bit
                if valid:
                    candidates.append((x, new_local_diffs))
            
            if not candidates:
                return None
            
            # Weighted selection: prefer smaller values to keep scope low
            # but allow exploration
            weights = [1.0 / (c[0] - current_max + 1) ** 0.5 for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            chosen = candidates[-1]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = c
                    break
            
            next_x, new_local_diffs = chosen
            row.append(next_x)
            local_mask |= new_local_diffs
            current_max = next_x
            
        return row

    def greedy_construct_all(rng, max_window):
        """Construct all rows greedily."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = greedy_construct_row(used_mask, rng, max_window)
            if row is None:
                return None
            rows.append(row)
            used_mask |= row_to_mask(row)
        
        # Final check
        if is_valid_mask(used_mask):
            return rows
        return None

    # Phase 1: Randomized Greedy Initialization
    # Try multiple random seeds and window sizes
    num_initial_attempts = 500
    for i in range(num_initial_attempts):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(i * 12345 + 67890)
        # Vary the window size
        # Smaller windows might fail but produce compact rows if they succeed
        # Larger windows are safer but might produce larger rows
        max_window = random.choice([30, 40, 50, 60, 80, 100])
        
        rows = greedy_construct_all(rng, max_window)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    if best_rows is None:
        # Fallback to deterministic greedy if all random failed (unlikely)
        rng = random.Random(0)
        best_rows = greedy_construct_all(rng, 200)
        if best_rows is None:
            # Last resort: simple sequential
            used_mask = 0
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        bit = 1 << (d - 1)
                        if used_mask & bit:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_mask |= (1 << (x - a - 1))
                        break
                if len(row) < k + 1:
                    return None # Should not happen with large x
                best_rows.append(row)
        
        best_scope = get_scope(best_rows)

    # Phase 2: Local Search
    # We use the bitmask to speed up validity checks.
    # We will try to modify one element in one row at a time.
    
    if time.time() > deadline - 5:
        return best_rows

    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    temperature = max(1.0, (current_scope - lower_bound) / 10.0)
    cooling_rate = 0.999
    
    # Pre-compute masks for current rows
    row_masks = [row_to_mask(r) for r in current_rows]
    global_mask = 0
    for m in row_masks:
        global_mask |= m

    def try_modify(rows, row_masks, global_mask):
        """Try a random modification and return modified rows, masks, and global mask if valid, else None."""
        new_rows = [r[:] for r in rows]
        new_row_masks = [m for m in row_masks]
        new_global_mask = global_mask
        
        row_idx = random.randint(0, len(new_rows) - 1)
        row = new_rows[row_idx]
        old_mask = new_row_masks[row_idx]
        
        # Remove old row's mask from global
        new_global_mask &= ~old_mask
        
        mod_type = random.choice(['shift', 'replace'])
        
        if mod_type == 'shift':
            if len(row) > 1:
                idx = random.randint(1, len(row) - 1)
                delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                new_val = row[idx] + delta
                # Check bounds
                if new_val > row[idx - 1] and (idx == len(row) - 1 or new_val < row[idx + 1]):
                    row[idx] = new_val
                    new_mask = row_to_mask(row)
                    # Check if new mask is disjoint from other rows
                    other_mask = new_global_mask
                    if not (new_mask & other_mask):
                        # Update
                        new_row_masks[row_idx] = new_mask
                        new_global_mask |= new_mask
                        return new_rows, new_row_masks, new_global_mask
        
        elif mod_type == 'replace':
            if len(row) > 1:
                idx = random.randint(1, len(row) - 1)
                # Try values in a range around current value
                # Bias towards smaller values
                for _ in range(10):
                    new_val = row[idx] + random.randint(-10, 10)
                    if new_val <= row[idx - 1] or (idx < len(row) - 1 and new_val >= row[idx + 1]):
                        continue
                    row[idx] = new_val
                    new_mask = row_to_mask(row)
                    other_mask = new_global_mask
                    if not (new_mask & other_mask):
                        new_row_masks[row_idx] = new_mask
                        new_global_mask |= new_mask
                        return new_rows, new_row_masks, new_global_mask
        
        return None

    steps = 0
    while time.time() < deadline - 2:
        modified = try_modify(current_rows, row_masks, global_mask)
        if modified is not None:
            new_rows, new_row_masks, new_global_mask = modified
            new_scope = get_scope(new_rows)
            
            if new_scope < current_scope:
                current_rows = new_rows
                row_masks = new_row_masks
                global_mask = new_global_mask
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                    if best_scope <= lower_bound:
                        return best_rows
            else:
                # Accept with some probability (Simulated Annealing)
                if random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_rows = new_rows
                    row_masks = new_row_masks
                    global_mask = new_global_mask
                    current_scope = new_scope
            
        temperature *= cooling_rate
        steps += 1
        
        # Occasionally restart local search from a fresh greedy solution
        if steps % 100 == 0 and random.random() < 0.05:
            if time.time() < deadline - 10:
                rng = random.Random(int(time.time() * 1000))
                fresh_rows = greedy_construct_all(rng, 50)
                if fresh_rows is not None:
                    fresh_scope = get_scope(fresh_rows)
                    if fresh_scope < current_scope:
                        current_rows = fresh_rows
                        row_masks = [row_to_mask(r) for r in fresh_rows]
                        global_mask = 0
                        for m in row_masks:
                            global_mask |= m
                        current_scope = fresh_scope
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]

    return best_rows
