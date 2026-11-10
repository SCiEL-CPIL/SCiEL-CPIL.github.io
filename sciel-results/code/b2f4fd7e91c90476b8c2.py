import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    num_elements = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
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

    def full_mask(rows):
        """Compute the full difference mask for all rows."""
        mask = 0
        for row in rows:
            mask |= row_to_mask(row)
        return mask

    def is_valid_mask(mask):
        """Check if mask has no duplicate differences (popcount check)."""
        return bin(mask).count('1') == total_diffs

    def greedy_row_builder(used_mask, rng, max_window):
        """
        Correctly builds a row avoiding used_mask and internal duplicates.
        """
        row = [0]
        local_mask = 0
        current_max = 0
        
        for _ in range(num_elements - 1):
            candidates = []
            start_val = current_max + 1
            end_val = current_max + max_window
            
            for x in range(start_val, end_val + 1):
                valid = True
                new_local_diffs = 0
                for a in row:
                    d = x - a
                    if d <= 0: continue
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
            
            # Weighted selection
            # Weight based on value
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
            row = greedy_row_builder(used_mask, rng, max_window)
            if row is None:
                return None
            rows.append(row)
            used_mask |= row_to_mask(row)
        return rows

    # Phase 1: Randomized Greedy Initialization
    # Try multiple random seeds and window sizes
    num_initial_attempts = 200
    for i in range(num_initial_attempts):
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(i * 12345 + 67890)
        # Vary the window size
        # Smaller windows might fail but produce compact rows if they succeed
        # Larger windows are safer but might produce larger rows
        max_window = random.choice([30, 40, 50, 60, 80])
        
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
        best_rows = greedy_construct_all(rng, 100)
        if best_rows is None:
            # Last resort: simple sequential
            used = 0
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < num_elements:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        bit = 1 << (d - 1)
                        if used & bit:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used |= (1 << (x - a - 1))
                best_rows.append(row)
            best_scope = get_scope(best_rows)

    # Phase 2: Local Search with Simulated Annealing
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    current_mask = full_mask(current_rows)

    temperature = max(1.0, (current_scope - lower_bound) / 5.0)
    cooling_rate = 0.999
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline - 2 and no_improve_count < max_no_improve:
        row_idx = random.randint(0, len(current_rows) - 1)
        row = current_rows[row_idx]
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Move 1: Shift one element
            if len(row) <= 1:
                continue
            idx = random.randint(1, len(row) - 1)
            delta = random.randint(-20, 20)
            new_val = row[idx] + delta
            
            if new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue
            
            # Compute old and new masks
            old_mask = 0
            new_mask = 0
            for j in range(idx):
                d_old = row[idx] - row[j]
                d_new = new_val - row[j]
                if d_old > 0: old_mask |= (1 << (d_old - 1))
                if d_new > 0: new_mask |= (1 << (d_new - 1))
            for j in range(idx + 1, len(row)):
                d_old = row[j] - row[idx]
                d_new = row[j] - new_val
                if d_old > 0: old_mask |= (1 << (d_old - 1))
                if d_new > 0: new_mask |= (1 << (d_new - 1))
            
            other_mask = current_mask & ~old_mask
            if new_mask & other_mask:
                continue
            
            # Check for internal duplicates in new_mask?
            # Since we are just shifting one element, and the row was valid before,
            # and we checked against `other_mask` (which includes other rows and parts of this row not involving idx),
            # we must ensure the new differences involving idx are distinct from each other.
            # `new_mask` is a mask, so if there's a duplicate bit, it's just set once.
            # We need to ensure the number of set bits in `new_mask` is exactly `2 * (len(row) - 1)`?
            # No, `new_mask` contains diffs with all other elements.
            # If two new diffs are equal, the bit is set once, but we need them to be distinct.
            # So we must check if the count of bits in `new_mask` equals the number of pairs.
            num_new_pairs = len(row) - 1
            if bin(new_mask).count('1') != num_new_pairs:
                continue
            
            # Valid move
            old_val = current_rows[row_idx][idx]
            current_rows[row_idx][idx] = new_val
            new_global_mask = other_mask | new_mask
            new_scope = max(max(r) for r in current_rows)
            
            if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                current_scope = new_scope
                current_mask = new_global_mask
                no_improve_count = 0
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    if best_scope <= lower_bound:
                        return best_rows
            else:
                current_rows[row_idx][idx] = old_val
                no_improve_count += 1
        else:
            # Move 2: Rebuild the row
            # Remove current row's mask from global
            row_mask = row_to_mask(row)
            other_mask = current_mask & ~row_mask
            
            # Rebuild row using greedy builder
            rng = random.Random()
            # Use a slightly larger window for rebuild to increase chance of success
            new_row = greedy_row_builder(other_mask, rng, 60)
            
            if new_row is not None:
                new_row_mask = row_to_mask(new_row)
                new_global_mask = other_mask | new_row_mask
                new_scope = max(max(r) for r in current_rows)
                # We need to update the row in current_rows to calculate scope correctly
                old_row = current_rows[row_idx]
                current_rows[row_idx] = new_row
                new_scope = max(max(r) for r in current_rows)
                
                if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_scope = new_scope
                    current_mask = new_global_mask
                    no_improve_count = 0
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                else:
                    current_rows[row_idx] = old_row
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1

    # Phase 3: Polish pass with tighter tolerance
    if time.time() < deadline - 5:
        polish_rows = [r[:] for r in best_rows]
        polish_scope = best_scope
        polish_mask = full_mask(polish_rows)
        
        # Use a smaller temperature for finer tuning
        polish_temp = 0.5
        polish_no_improve = 0
        max_polish_no_improve = 2000
        
        while time.time() < deadline - 2 and polish_no_improve < max_polish_no_improve:
            row_idx = random.randint(0, len(polish_rows) - 1)
            row = polish_rows[row_idx]
            
            if len(row) <= 1:
                continue
            idx = random.randint(1, len(row) - 1)
            # Smaller delta for polish
            delta = random.randint(-5, 5)
            new_val = row[idx] + delta
            
            if new_val <= row[idx - 1]:
                continue
            if idx < len(row) - 1 and new_val >= row[idx + 1]:
                continue
            
            # Compute old and new masks
            old_mask = 0
            new_mask = 0
            for j in range(idx):
                d_old = row[idx] - row[j]
                d_new = new_val - row[j]
                if d_old > 0: old_mask |= (1 << (d_old - 1))
                if d_new > 0: new_mask |= (1 << (d_new - 1))
            for j in range(idx + 1, len(row)):
                d_old = row[j] - row[idx]
                d_new = row[j] - new_val
                if d_old > 0: old_mask |= (1 << (d_old - 1))
                if d_new > 0: new_mask |= (1 << (d_new - 1))
            
            other_mask = polish_mask & ~old_mask
            if new_mask & other_mask:
                continue
            
            num_new_pairs = len(row) - 1
            if bin(new_mask).count('1') != num_new_pairs:
                continue
            
            # Valid move
            old_val = polish_rows[row_idx][idx]
            polish_rows[row_idx][idx] = new_val
            new_global_mask = other_mask | new_mask
            new_scope = max(max(r) for r in polish_rows)
            
            if new_scope < polish_scope:
                polish_scope = new_scope
                polish_mask = new_global_mask
                polish_no_improve = 0
                if polish_scope < best_scope:
                    best_scope = polish_scope
                    best_rows = [r[:] for r in polish_rows]
                    if best_scope <= lower_bound:
                        return best_rows
            else:
                polish_rows[row_idx][idx] = old_val
                polish_no_improve += 1

    # Phase 4: Exhaustive local search for the best row
    # This tries to find the minimum possible max element for each row
    # given the constraints from other rows, by trying all valid shifts
    # in a more systematic way.
    
    if time.time() < deadline - 2:
        final_rows = [r[:] for r in best_rows]
        final_scope = best_scope
        final_mask = full_mask(final_rows)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for r_idx in range(n_rows):
                row = final_rows[r_idx]
                # Remove this row's mask
                row_mask = row_to_mask(row)
                other_mask = final_mask & ~row_mask
                
                # Try to shift each element to minimize the row's max
                # We do this by trying to reduce the largest element first
                # Then the second largest, etc.
                
                # For each element from last to first (excluding 0)
                for idx in range(len(row) - 1, 0, -1):
                    # Try to reduce row[idx] as much as possible
                    # while maintaining validity and strict increase
                    min_allowed = row[idx - 1] + 1
                    if idx < len(row) - 1:
                        max_allowed = row[idx + 1] - 1
                    else:
                        max_allowed = row[idx] # We want to reduce, so start from current
                    
                    # Try values from min_allowed up to current value
                    # We want the smallest valid value
                    found = False
                    for new_val in range(min_allowed, row[idx] + 1):
                        # If new_val == row[idx], it's the current state, skip if we haven't found better
                        if new_val == row[idx] and not found:
                            continue
                            
                        # Check validity
                        old_mask = 0
                        new_mask = 0
                        for j in range(idx):
                            d_old = row[idx] - row[j]
                            d_new = new_val - row[j]
                            if d_old > 0: old_mask |= (1 << (d_old - 1))
                            if d_new > 0: new_mask |= (1 << (d_new - 1))
                        for j in range(idx + 1, len(row)):
                            d_old = row[j] - row[idx]
                            d_new = row[j] - new_val
                            if d_old > 0: old_mask |= (1 << (d_old - 1))
                            if d_new > 0: new_mask |= (1 << (d_new - 1))
                        
                        # Check against other rows and rest of this row
                        # Note: old_mask includes diffs involving idx
                        # other_mask has diffs NOT involving idx
                        # We need new_mask to not intersect other_mask
                        # AND new_mask must have correct popcount
                        if new_mask & other_mask:
                            continue
                            
                        num_new_pairs = len(row) - 1
                        if bin(new_mask).count('1') != num_new_pairs:
                            continue
                            
                        # Valid!
                        if new_val < row[idx]:
                            row[idx] = new_val
                            found = True
                            improved = True
                            # Update final_mask
                            # We need to recalculate the mask for this row
                            # Actually, we can update final_mask by removing old row mask and adding new
                            # But we are modifying row in place.
                            # Let's just recalculate final_mask at the end of the loop for simplicity/safety
                            # Or update it incrementally
                            break
                    
                    if found:
                        # Update final_mask
                        # Recalculate is safer
                        pass

                if improved:
                    # Recalculate final_mask
                    final_mask = full_mask(final_rows)
                    new_scope = max(max(r) for r in final_rows)
                    if new_scope < final_scope:
                        final_scope = new_scope
                        if final_scope < best_scope:
                            best_scope = final_scope
                            best_rows = [r[:] for r in final_rows]
                            if best_scope <= lower_bound:
                                return best_rows

    return best_rows
