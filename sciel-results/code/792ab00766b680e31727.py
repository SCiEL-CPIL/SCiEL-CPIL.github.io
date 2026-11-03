"""Rewritten program for difference triangle set task."""
import time
import random
import sys

def construct(n_rows, k):
    """Construct difference triangle set: n_rows rows of k+1 increasing integers
    starting at 0, with all positive differences distinct across all rows.
    Minimizes the largest entry."""
    
    start_time = time.time()
    time_limit = 115  # seconds, leave buffer for evaluation overhead
    
    # Number of distinct differences needed: n_rows * C(k+1, 2)
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Lower bound on max entry: we need total_diffs distinct positive integers
    # that can all be represented as differences within rows of length k+1 starting at 0.
    # Simple counting bound: the max value must be at least such that the number of
    # distinct differences we can form is sufficient.
    # For a single row with max value m and k+1 entries, max distinct diffs is C(k+1,2).
    # But the values must be chosen from 0..m.
    # The counting bound mentioned: 112 for (4,7).
    # Let's use a practical lower bound.
    
    # Greedy lower bound: need total_diffs distinct differences, each <= max_entry
    # If max_entry = M, the set of all possible differences is {1, ..., M}, size M.
    # But not all are achievable simultaneously due to row structure constraints.
    # Simple bound: M >= total_diffs (trivial, too weak)
    # Better: for k+1 marks in [0,M], the number of distinct differences is at most
    # (k+1)*M/2 roughly... actually C(k+1,2) per row.
    # The counting bound of 112 for (4,7): 4*28=112 differences, and M=112.
    # So the bound is just M >= total_diffs.
    
    lower_bound = total_diffs  # trivial counting bound
    
    # Best known for (4,7) is 114? No, 134. Let me re-read.
    # "The best known set has largest entry 134"
    # "the simple counting bound is 112"
    
    # Strategy: Use a randomized greedy with backtracking for small problems,
    # and for larger problems use a simulated annealing / local search approach.
    
    # For the specific case (4,7), let's try a structured approach.
    
    # Approach 1: Greedy with randomization and multiple restarts
    # Approach 2: Local search starting from greedy solution
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_construct(n_rows, k, seed=0):
        """Greedy construction with random tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
        return rows
    
    def local_search(rows, n_rows, k, time_budget=30, seed=0):
        """Improve a valid solution using local search."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        current_max = max_entry(rows)
        best_rows = [list(r) for r in rows]
        best_max = current_max
        
        deadline = start_time + time_budget
        
        while time.time() < deadline:
            # Pick a random row and a random position to perturb
            r_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, len(rows[r_idx]))  # don't change the 0
            
            old_val = rows[r_idx][pos]
            
            # Try to change this value to something else
            # New value must maintain increasing order
            min_val = rows[r_idx][pos - 1] + 1
            max_val = rows[r_idx][pos + 1] - 1 if pos + 1 < len(rows[r_idx]) else current_max + 5
            
            if max_val < min_val:
                continue
            
            # Try a few candidate values
            candidates = set()
            candidates.add(old_val + rng.randrange(-5, 6))
            candidates.add(old_val - rng.randrange(1, 5))
            candidates.add(old_val + rng.randrange(5, 15))
            candidates.add(rng.randrange(min_val, min(max_val + 1, current_max + 1)))
            
            found_improvement = False
            for new_val in sorted(candidates):
                if new_val < min_val or new_val > max_val:
                    continue
                if new_val == old_val:
                    continue
                
                # Check if changing this value maintains validity
                # Remove old differences involving this position
                # Add new differences and check for conflicts
                
                # Build set of differences not involving this position
                other_diffs = set()
                for r_i, row in enumerate(rows):
                    n = len(row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if r_i == r_idx and (i == pos or j == pos):
                                continue
                            d = row[j] - row[i]
                            other_diffs.add(d)
                
                # Now add new differences for position pos
                new_diffs = set()
                valid = True
                row = rows[r_idx]
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d <= 0:
                        valid = False
                        break
                    if d in other_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Accept the move
                rows[r_idx][pos] = new_val
                new_max = max_entry(rows)
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
                    found_improvement = True
                else:
                    # Simulated annealing: accept worse solutions with some probability
                    # Only if we're still below a threshold
                    if new_max <= best_max + 3:
                        # Accept with some probability
                        if rng.random() < 0.3:
                            pass  # keep the change
                        else:
                            rows[r_idx][pos] = old_val
                            break
                    else:
                        rows[r_idx][pos] = old_val
                        break
                
                break  # only do one change at a time
            
            if found_improvement:
                # Reset some exploration
                pass
        
        return best_rows
    
    # For small problems, try multiple strategies
    
    if n_rows <= 6 and k <= 8:
        # Try multiple greedy seeds and local search
        best_result = None
        best_score = float('inf')
        
        deadline = start_time + time_limit
        
        n_attempts = 20
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            seed = attempt * 7919 + 13
            rows = greedy_construct(n_rows, k, seed)
            if is_valid(rows):
                score = max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_result = rows
            
            # Local search from this solution
            if time.time() > deadline - 10:
                break
            improved = local_search(rows, n_rows, k, time_budget=min(10, deadline - time.time() - 5), seed=seed)
            if is_valid(improved):
                score = max_entry(improved)
                if score < best_score:
                    best_score = score
                    best_result = improved
        
        if best_result is not None:
            return best_result
    
    # Fallback: pure greedy
    rows = greedy_construct(n_rows, k, seed=42)
    return rows
