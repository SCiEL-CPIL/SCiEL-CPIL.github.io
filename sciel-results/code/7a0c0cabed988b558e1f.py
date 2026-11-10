import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 120s limit
    
    num_elements = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def compute_diff_mask(rows):
        """Compute bitmask of all differences."""
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << (d - 1))
        return mask
    
    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if (used_mask & bit) or (new_diffs & bit):
                        valid = False
                        break
                    new_diffs |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_diffs
            rows.append(row)
        return rows
    
    # Phase 1: Pure greedy baseline
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = [r[:] for r in greedy_rows]
        best_scope = greedy_scope
    
    # Phase 2: Simulated Annealing with bitmask operations
    if best_rows is not None and time.time() < deadline - 5:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_mask = compute_diff_mask(current_rows)
        
        # Precompute row difference masks for fast updates
        row_masks = []
        for row in current_rows:
            rm = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    rm |= (1 << (d - 1))
            row_masks.append(rm)
        
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.9995
        no_improve_count = 0
        max_no_improve = 5000
        rng = random.Random(42)
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Try a random modification
            row_idx = rng.randint(0, len(current_rows) - 1)
            row = current_rows[row_idx]
            
            # Move type 1: Shift one element (70% probability)
            if rng.random() < 0.7 and len(row) > 1:
                idx = rng.randint(1, len(row) - 1)
                delta = rng.randint(-30, 30)
                new_val = row[idx] + delta
                
                # Check ordering constraints
                if new_val <= row[idx - 1]:
                    no_improve_count += 1
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    no_improve_count += 1
                    continue
                
                # Compute old and new differences involving this element using bitmask
                old_row_mask = row_masks[row_idx]
                
                # Remove old differences involving element idx
                for j in range(idx):
                    d = row[idx] - row[j]
                    old_row_mask &= ~(1 << (d - 1))
                for j in range(idx + 1, len(row)):
                    d = row[j] - row[idx]
                    old_row_mask &= ~(1 << (d - 1))
                
                # Add new differences involving element idx
                new_row_mask = old_row_mask
                for j in range(idx):
                    d = new_val - row[j]
                    if d <= 0:
                        break
                    new_row_mask |= (1 << (d - 1))
                for j in range(idx + 1, len(row)):
                    d = row[j] - new_val
                    if d <= 0:
                        break
                    new_row_mask |= (1 << (d - 1))
                
                # Check for conflicts with other rows
                other_mask = 0
                for i in range(len(current_rows)):
                    if i != row_idx:
                        other_mask |= row_masks[i]
                
                if new_row_mask & other_mask:
                    no_improve_count += 1
                    continue
                
                # Compute new scope
                new_scope = max(max(r) for r in current_rows if current_rows.index(r) != row_idx)
                new_scope = max(new_scope, new_val)
                
                # Accept or reject based on simulated annealing
                if new_scope <= current_scope or rng.random() < math.exp(-(new_scope - current_scope) / temperature):
                    # Apply the move
                    current_rows[row_idx][idx] = new_val
                    row_masks[row_idx] = new_row_mask
                    current_mask = 0
                    for rm in row_masks:
                        current_mask |= rm
                    current_scope = new_scope
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            else:
                # Move type 2: Replace entire row (30% probability)
                # Compute mask of all other rows
                other_mask = 0
                for i in range(len(current_rows)):
                    if i != row_idx:
                        other_mask |= row_masks[i]
                
                # Try to build a new row avoiding other_mask
                new_row = [0]
                current_max = 0
                success = True
                
                while len(new_row) < num_elements:
                    candidates = []
                    x = current_max + 1
                    max_x = current_max + max(100, k * 10)
                    
                    while x <= max_x:
                        new_diffs = 0
                        valid = True
                        for a in new_row:
                            d = x - a
                            bit = 1 << (d - 1)
                            if (other_mask & bit) or (new_diffs & bit):
                                valid = False
                                break
                            new_diffs |= bit
                        if valid:
                            candidates.append((x, new_diffs))
                        x += 1
                    
                    if not candidates:
                        success = False
                        break
                    
                    # Weight towards smaller values
                    if len(candidates) == 1:
                        next_x, next_diffs = candidates[0]
                    else:
                        weights = [1.0 / (c[0] - current_max + 1) ** 0.5 for c in candidates]
                        total_w = sum(weights)
                        r = rng.random() * total_w
                        cumulative = 0
                        next_x, next_diffs = candidates[-1]
                        for c, w in zip(candidates, weights):
                            cumulative += w
                            if r <= cumulative:
                                next_x, next_diffs = c
                                break
                    
                    new_row.append(next_x)
                    current_max = next_x
                
                if success:
                    # Compute new row mask
                    new_row_mask = 0
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            d = new_row[j] - new_row[i]
                            new_row_mask |= (1 << (d - 1))
                    
                    # Check no conflicts
                    if not (new_row_mask & other_mask):
                        new_scope = max(max(r) for r in current_rows if current_rows.index(r) != row_idx)
                        new_scope = max(new_scope, max(new_row))
                        
                        if new_scope <= current_scope or rng.random() < math.exp(-(new_scope - current_scope) / temperature):
                            current_rows[row_idx] = new_row
                            row_masks[row_idx] = new_row_mask
                            current_mask = 0
                            for rm in row_masks:
                                current_mask |= rm
                            current_scope = new_scope
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            
            # Update best if improved
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
                if best_scope <= lower_bound:
                    return best_rows
            
            # Cool down
            temperature = max(0.1, temperature * cooling_rate)
        
        # Final check
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in current_rows]
    
    # Phase 3: Final greedy refinement on best rows
    if best_rows is not None and time.time() < deadline - 2:
        # Try to reduce scope by rebuilding rows greedily with tighter constraints
        for _ in range(10):
            if time.time() >= deadline - 2:
                break
            
            # Compute mask of all rows except the one we're rebuilding
            for row_idx in range(n_rows):
                other_mask = 0
                for i in range(n_rows):
                    if i != row_idx:
                        row = best_rows[i]
                        for a in range(len(row)):
                            for b in range(a + 1, len(row)):
                                d = row[b] - row[a]
                                other_mask |= (1 << (d - 1))
                
                # Try to rebuild this row with greedy
                new_row = [0]
                current_max = 0
                success = True
                
                while len(new_row) < num_elements:
                    x = current_max + 1
                    max_x = current_max + max(50, k * 5)
                    found = False
                    
                    while x <= max_x:
                        new_diffs = 0
                        valid = True
                        for a in new_row:
                            d = x - a
                            bit = 1 << (d - 1)
                            if (other_mask & bit) or (new_diffs & bit):
                                valid = False
                                break
                            new_diffs |= bit
                        if valid:
                            new_row.append(x)
                            current_max = x
                            found = True
                            break
                        x += 1
                    
                    if not found:
                        success = False
                        break
                
                if success:
                    new_scope = max(max(r) for r in best_rows if best_rows.index(r) != row_idx)
                    new_scope = max(new_scope, max(new_row))
                    if new_scope < best_scope:
                        best_rows[row_idx] = new_row
                        best_scope = new_scope
    
    return best_rows
