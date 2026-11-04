"""Improved difference triangle set solver using backtracking with bitset tracking."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 145.0
    total_diffs_needed = n_rows * k * (k + 1) // 2

    # Bitset for tracking used differences. We need differences up to ~200 for small cases.
    # Use a list of 64-bit integers as a bitset for fast operations.
    # Max difference we'll realistically need: for k=7, n_rows=4, max entry ~134, so diffs up to ~134.
    # Let's use a generous bound.
    MAX_DIFF = 200
    n_bits = (MAX_DIFF + 63) // 64
    n_bits = max(n_bits, 4)  # at least 4 words

    def make_bitset():
        return [0] * n_bits

    def bitset_set(bs, d):
        """Set bit d in bitset bs. Returns True if it was already set (conflict)."""
        if d > MAX_DIFF:
            return True  # treat as conflict
        idx = d // 64
        mask = 1 << (d % 64)
        if bs[idx] & mask:
            return True
        bs[idx] |= mask
        return False

    def bitset_clear(bs, d):
        """Clear bit d in bitset bs."""
        if d > MAX_DIFF:
            return
        idx = d // 64
        mask = 1 << (d % 64)
        bs[idx] &= ~mask

    def bitset_count(bs):
        """Count set bits in bitset."""
        count = 0
        for word in bs:
            count += bin(word).count('1')
        return count

    def copy_bitset(bs):
        return bs[:]

    def check_conflicts(row, bs):
        """Check if all diffs in row conflict with bs. Returns list of conflicting diffs or None."""
        conflicts = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d > MAX_DIFF:
                    return [d]  # too large
                idx = d // 64
                mask = 1 << (d % 64)
                if bs[idx] & mask:
                    conflicts.append(d)
        return conflicts if conflicts else None

    def add_row_to_bitset(row, bs):
        """Add all diffs from row to bitset. Assumes no conflicts."""
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= MAX_DIFF:
                    idx = d // 64
                    mask = 1 << (d % 64)
                    bs[idx] |= mask

    def remove_row_from_bitset(row, bs):
        """Remove all diffs from row from bitset."""
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= MAX_DIFF:
                    idx = d // 64
                    mask = 1 << (d % 64)
                    bs[idx] &= ~mask

    # Greedy initialization
    def greedy_init():
        bs = make_bitset()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if adding x would cause conflicts
                conflict = False
                for a in row:
                    d = x - a
                    if d > MAX_DIFF:
                        conflict = True
                        break
                    idx = d // 64
                    mask = 1 << (d % 64)
                    if bs[idx] & mask:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        idx = d // 64
                        mask = 1 << (d % 64)
                        bs[idx] |= mask
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max(max(row) for row in best_rows)

    # Backtracking search with restarts
    def backtrack_search(rows_so_far, bs, current_row_idx, max_allowed):
        """Try to complete the solution starting from current_row_idx."""
        if time.time() > deadline:
            return None

        if current_row_idx == n_rows:
            return rows_so_far[:]

        # Try to build row current_row_idx
        # We want to keep entries small to minimize max
        # Start with [0] and try to add elements
        row = [0]
        
        # Use a recursive helper to build the row
        def build_row(pos, max_val):
            if time.time() > deadline:
                return None
            if pos == k + 1:
                return row[:]
            
            start = row[-1] + 1
            # Limit search to reasonable range
            end = min(max_allowed, start + 30)
            
            # Try values in order, preferring smaller ones
            for val in range(start, end + 1):
                # Check conflicts with current bitset
                conflict = False
                for a in row:
                    d = val - a
                    if d > MAX_DIFF:
                        conflict = True
                        break
                    idx = d // 64
                    mask = 1 << (d % 64)
                    if bs[idx] & mask:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Tentatively add
                row.append(val)
                for a in row[:-1]:
                    d = val - a
                    idx = d // 64
                    mask = 1 << (d % 64)
                    bs[idx] |= mask
                
                result = build_row(pos + 1, max_val)
                if result is not None:
                    return result
                
                # Backtrack
                row.pop()
                for a in row:
                    d = val - a
                    idx = d // 64
                    mask = 1 << (d % 64)
                    bs[idx] &= ~mask
            
            return None
        
        result = build_row(1, max_allowed)
        if result is not None:
            # Add this row to bitset permanently
            add_row_to_bitset(result, bs)
            rows_so_far.append(result)
            final = backtrack_search(rows_so_far, bs, current_row_idx + 1, max_allowed)
            if final is not None:
                return final
            # Backtrack
            rows_so_far.pop()
            remove_row_from_bitset(result, bs)
        
        return None

    # Try to improve via backtracking with decreasing max_allowed
    # Start from current best and try to reduce
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try to find a solution with max_allowed = best_max - 1
        target_max = best_max - 1
        if target_max < k + 1:
            break
        
        bs = make_bitset()
        rows_so_far = []
        result = backtrack_search(rows_so_far, bs, 0, target_max)
        
        if result is not None:
            new_max = max(max(row) for row in result)
            if new_max < best_max:
                best_max = new_max
                best_rows = result
        
        # Also try with some randomness in the order
        if attempt % 5 == 4:
            # Randomized restart: shuffle the order of trying values
            pass  # For now, keep it simple

    # Local search to fine-tune
    def local_search(rows, max_iter=50000):
        if time.time() > deadline:
            return rows
        
        bs = make_bitset()
        for row in rows:
            add_row_to_bitset(row, bs)
        
        current_max = max(max(row) for row in rows)
        best_local = [r[:] for r in rows]
        best_local_max = current_max
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random row and position
            r = random.randint(0, n_rows - 1)
            i = random.randint(1, k)
            
            old_val = rows[r][i]
            lower = rows[r][i - 1] + 1
            upper = rows[r][i + 1] - 1 if i < k else current_max
            
            # Try to decrease the value
            if i < k:
                # Try values from lower to old_val - 1
                candidates = list(range(lower, old_val))
            else:
                # For the last element, try to decrease
                candidates = list(range(lower, old_val))
            
            if not candidates:
                continue
            
            random.shuffle(candidates)
            
            improved = False
            for new_val in candidates:
                if new_val >= old_val:
                    continue
                
                # Check if changing this value causes conflicts
                # Remove old diffs
                temp_bs = copy_bitset(bs)
                row = rows[r]
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        d = old_val - row[j]
                    else:
                        d = row[j] - old_val
                    if d <= MAX_DIFF:
                        idx = d // 64
                        mask = 1 << (d % 64)
                        temp_bs[idx] &= ~mask
                
                # Check new diffs
                conflict = False
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        d = new_val - row[j]
                    else:
                        d = row[j] - new_val
                    if d > MAX_DIFF:
                        conflict = True
                        break
                    idx = d // 64
                    mask = 1 << (d % 64)
                    if temp_bs[idx] & mask:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Accept this change
                # Update bitset
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        d = new_val - row[j]
                    else:
                        d = row[j] - new_val
                    if d <= MAX_DIFF:
                        idx = d // 64
                        mask = 1 << (d % 64)
                        temp_bs[idx] |= mask
                
                rows[r][i] = new_val
                bs = temp_bs
                new_max = max(max(row) for row in rows)
                if new_max < best_local_max:
                    best_local_max = new_max
                    best_local = [rr[:] for rr in rows]
                current_max = new_max
                improved = True
                break
            
            if not improved and it % 1000 == 0:
                # Try a larger perturbation
                pass
        
        return best_local

    # Run local search
    best_rows = local_search(best_rows)
    best_max = max(max(row) for row in best_rows)

    return best_rows
