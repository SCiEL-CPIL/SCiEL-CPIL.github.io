import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on scope
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

    def get_forbidden_diffs(rows_except_idx):
        """
        Returns the set of differences used by all rows except the one at index idx.
        """
        forbidden = set()
        for i, row in enumerate(rows_except_idx):
            if i == idx:
                continue
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    forbidden.add(row[b] - row[a])
        return forbidden

    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def can_construct_row(forbidden_diffs, k, max_val):
        """
        Checks if a row of k+1 integers starting at 0, ending <= max_val,
        with strictly increasing integers and all differences not in forbidden_diffs, exists.
        Returns the row if found, else None.
        Uses backtracking.
        """
        row = [0]
        
        def backtrack(idx, current_max):
            if idx == k + 1:
                return row[:]
            
            # Lower bound for next element
            next_min = row[-1] + 1
            
            # We want to keep max_val as small as possible, so we iterate from smallest to largest
            # But we are constrained by max_val
            for candidate in range(next_min, max_val + 1):
                # Check if candidate creates any forbidden diffs
                valid = True
                for prev_val in row:
                    diff = candidate - prev_val
                    if diff in forbidden_diffs:
                        valid = False
                        break
                
                if valid:
                    row.append(candidate)
                    result = backtrack(idx + 1, max_val)
                    if result is not None:
                        return result
                    row.pop()
            return None

        return backtrack(1, max_val)

    def find_best_row_for_index(rows, idx, max_search_limit):
        """
        Tries to find a better row for position idx, given the other rows are fixed.
        Returns a new row if one with smaller max element is found, else None.
        """
        forbidden = get_forbidden_diffs(rows, idx)
        current_max = rows[idx][-1]
        
        # Binary search for the smallest possible max element
        # We know it must be at least current_max if we don't improve, but we want < current_max
        # Actually, we can just try to find *any* valid row with max < current_max
        # If we find one, we can try to minimize it further?
        # Let's just find the minimal possible max element for this configuration.
        
        # Lower bound for max element of this row:
        # It must be at least the max element of the row in the optimal global solution? No.
        # It must be large enough to accommodate k+1 numbers with distinct diffs not in forbidden.
        
        # We can binary search on M
        lo = 1
        hi = current_max - 1  # We only care if we can improve
        
        if hi < lo:
            return None
            
        best_row_found = None
        
        # To save time, we can first check if a row exists with max = hi.
        # If yes, binary search down.
        # If no, we are stuck.
        
        # Optimization: Instead of full binary search, we can just try to find the *first* valid row
        # by increasing M from a lower bound.
        # What is a good lower bound?
        # The row needs k+1 numbers. The smallest possible max is roughly k(k+1)/2? No, diffs must be distinct.
        # A simple lower bound is k.
        
        # Let's try to find the minimum M such that can_construct_row succeeds.
        # We'll do a linear scan from a reasonable lower bound up to current_max - 1.
        # If k=8, and current_max is ~100, scanning 1..99 is fast if backtracking is pruned well.
        
        # Actually, backtracking with `max_val` constraint is efficient if we fail early.
        
        # Let's try binary search to be safe on time.
        # If can_construct_row(forbidden, k, mid) is True, hi = mid
        # Else lo = mid + 1
        
        # But we need the *row* itself.
        # So:
        # 1. Find min_M
        # 2. Get row for min_M
        
        # First, check if even current_max is possible? It is, because we have a current row.
        # But we want strictly smaller.
        
        # Let's just iterate M from 1 to current_max - 1?
        # Worst case 130 iterations. Each backtracking is fast?
        # Let's try.
        
        for M in range(1, current_max):
            row_candidate = can_construct_row(forbidden, k, M)
            if row_candidate is not None:
                return row_candidate
                
        return None

    # Phase 1: Initial Greedy
    current_rows = greedy_construct()
    if is_valid(current_rows):
        best_rows = [r[:] for r in current_rows]
        best_scope = get_scope(current_rows)
    else:
        # Fallback if greedy fails (unlikely)
        return None

    # Phase 2: Row-Level Local Search
    # We will repeatedly try to improve the worst row (largest max element).
    # To avoid getting stuck, we also try random rows.
    
    no_improve_count = 0
    max_no_improve = 50 # Reheat or randomize if stuck
    
    while time.time() < deadline - 5:
        # Determine which row to improve
        # Strategy: Pick the row with the largest max element.
        # If we've been stuck, pick a random row.
        
        if no_improve_count > max_no_improve:
            idx_to_improve = random.randint(0, n_rows - 1)
            no_improve_count = 0
        else:
            # Find index of max element
            max_val = -1
            idx_to_improve = 0
            for i in range(n_rows):
                if current_rows[i][-1] > max_val:
                    max_val = current_rows[i][-1]
                    idx_to_improve = i

        # Try to find a better row for idx_to_improve
        new_row = find_best_row_for_index(current_rows, idx_to_improve, best_scope)
        
        if new_row is not None:
            # Check if it's actually better
            if new_row[-1] < current_rows[idx_to_improve][-1]:
                current_rows[idx_to_improve] = new_row
                current_scope = get_scope(current_rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                    if best_scope <= lower_bound:
                        return best_rows
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            # If we can't improve the worst row, maybe we can improve a different row?
            # The loop will pick the next worst or random one.
            
            # Occasionally, try a "Perturbation":
            # Pick a random row and replace it with a *random* valid row (generated by greedy) 
            # that might be larger, to change the landscape.
            if random.random() < 0.05:
                pert_idx = random.randint(0, n_rows - 1)
                # Generate a random valid row for pert_idx given others
                forbidden = get_forbidden_diffs(current_rows, pert_idx)
                # Use a simple greedy to generate *some* row
                row = [0]
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & forbidden):
                        row.append(x)
                    else:
                        # If greedy fails, try a bit of randomness
                        # For simplicity, if greedy fails, skip perturbation
                        success = False
                        break
                    if x > 200: # Limit growth
                        success = False
                        break
                if success:
                    # Accept it even if it's worse, to escape local minimum
                    current_rows[pert_idx] = row
                    # Do not update best_rows if scope increased, but continue search
                    # Reset no_improve to allow more search in this new landscape
                    no_improve_count = 0

    return best_rows
