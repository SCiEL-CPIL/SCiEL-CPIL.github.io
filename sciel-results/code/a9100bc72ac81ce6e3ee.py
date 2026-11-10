import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using greedy initialization
    and targeted local search to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return deadline - time.time()
    
    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry
    
    def greedy_initial(n_rows, k):
        """Simple greedy initialization - clean version from inspiration."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                if time.time() > deadline - 15:
                    return None
            rows.append(row)
        return rows
    
    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]
        
        # Collect old differences involving position ci
        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])
        
        # Remove old differences
        temp_used = used_diffs - old_diffs
        
        # Check if new_val is valid
        if new_val <= row[ci - 1] if ci > 0 else new_val <= 0:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None
        
        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        
        if not valid:
            return False, None
        
        # Success: return the new used set
        new_used = temp_used | new_diffs
        return True, new_used
    
    def local_search_reduce_max(rows, n_rows, k, max_iters=5000):
        """
        Local search specifically designed to reduce the maximum entry.
        Focus on the row with the largest last element and try to reduce it.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Find the row with the largest last element
            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]
            
            # Try to reduce the last element of the max row
            row = rows[max_row_idx]
            ci = len(row) - 1  # Last element
            
            # Try smaller values for the last element
            found = False
            for new_val in range(max_val - 1, row[ci - 1], -1):
                success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    new_score = max(r[-1] for r in rows)
                    if new_score < best_score:
                        best_score = new_score
                    found = True
                    break
            
            # If we couldn't reduce the last element, try other positions in that row
            if not found:
                for ci in range(len(row) - 2, 0, -1):
                    current = row[ci]
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                        if success:
                            rows[max_row_idx][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            found = True
                            break
                    if found:
                        break
            
            # Also try to reduce elements in other rows to free up differences
            if not found:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    row = rows[ri]
                    for ci in range(len(row) - 1, 0, -1):
                        current = row[ci]
                        # Only try to reduce if it helps
                        for new_val in range(current - 1, row[ci - 1], -1):
                            success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                            if success:
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                break
        
        return rows, best_score
    
    def local_search_general(rows, n_rows, k, max_iters=3000):
        """
        General local search: try to reduce any element to a smaller valid position.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            made_progress = False
            
            # Try to reduce elements in all rows
            for ri in range(n_rows):
                row = rows[ri]
                for ci in range(len(row) - 1, 0, -1):
                    current = row[ci]
                    # Try to move to a smaller value
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                        if success:
                            rows[ri][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            made_progress = True
                            break
                    if made_progress:
                        break
                if made_progress:
                    break
            
            if not made_progress:
                break
        
        return rows, best_score
    
    # Initialize with greedy
    best_rows = greedy_initial(n_rows, k)
    if best_rows is None:
        # Fallback: simple construction
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows
    
    best_score = validate(best_rows)[1]
    
    # First pass: targeted local search to reduce max
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=3000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Second pass: general local search
    improved_rows, improved_score = local_search_general(best_rows, n_rows, k, max_iters=2000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows
    
    # Random restarts with perturbation
    rng = random.Random(42)
    
    for restart in range(20):
        if time_remaining() < 25:
            break
        
        # Perturb the solution
        perturbed = [row[:] for row in best_rows]
        for ri in range(n_rows):
            for ci in range(1, len(perturbed[ri])):
                if rng.random() < 0.2:
                    perturbed[ri][ci] += rng.randint(1, 15)
        
        # Validate perturbed
        valid, score = validate(perturbed)
        if not valid:
            continue
        
        # Local search from perturbed
        improved_rows, improved_score = local_search_reduce_max(perturbed, n_rows, k, max_iters=1500)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
        
        # Another pass
        improved_rows, improved_score = local_search_general(best_rows, n_rows, k, max_iters=1000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows
    
    return best_rows
# EVOLVE-BLOCK-END
