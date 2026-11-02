"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    """
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Total differences needed: n_rows * C(k+1, 2)
    total_diffs = n_rows * (k + 1) * k // 2
    
    # Lower bound on max value: need total_diffs distinct positive integers
    # each <= max_val, so max_val >= total_diffs
    lb = total_diffs
    
    best_solution = None
    best_max_val = None
    
    # Strategy: randomized greedy with multiple restarts + local improvement
    # Each row is built greedily but with randomization to explore different regions
    
    def greedy_row(used_diffs, k, max_val, rng):
        """Build a single row greedily, trying to keep values small."""
        row = [0]
        current_max = 0
        for _ in range(k):
            found = False
            # Try candidates in order, but with some randomization
            # Start from current_max + 1 up to max_val
            candidates = list(range(current_max + 1, max_val + 1))
            # Shuffle a portion to add randomness while keeping small values preferred
            # Use a weighted approach: earlier candidates more likely
            if len(candidates) > 1:
                # Partial shuffle: swap some adjacent elements
                for i in range(min(len(candidates) - 1, 10)):
                    if rng.random() < 0.3:
                        candidates[i], candidates[i+1] = candidates[i+1], candidates[i]
            
            for x in candidates:
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    current_max = x
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                return None
        return row
    
    def greedy_full(n_rows, k, max_val, rng):
        """Build all rows greedily within max_val."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs, k, max_val, rng)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    def local_improve(rows, k, rng, iterations=50):
        """Try to reduce the max value by rebuilding rows."""
        max_val = max(max(row) for row in rows)
        for _ in range(iterations):
            # Pick a random row to rebuild
            idx = rng.randint(0, len(rows) - 1)
            old_row = rows[idx]
            
            # Compute used diffs without this row
            all_diffs = set()
            for i, r in enumerate(rows):
                if i == idx:
                    continue
                for a, b in combinations(r, 2):
                    all_diffs.add(b - a)
            
            # Try to rebuild this row with smaller max
            target_max = max_val - 1
            row = greedy_row(all_diffs, k, target_max, rng)
            if row is not None:
                new_max = max(row)
                if new_max < max(row):
                    rows[idx] = row
                    max_val = max(max(r) for r in rows)
                    # Continue improving
                    continue
            # If can't improve, try with same max but different structure
            row = greedy_row(all_diffs, k, max_val, rng)
            if row is not None and max(row) <= max_val:
                rows[idx] = row
        
        return rows, max_val
    
    # First, try deterministic greedy to get a baseline
    rng = random.Random(42)
    # Estimate a good max_val: start from lower bound and increase
    # The counting bound gives lb = total_diffs, but we need more room
    # For (7,5): total_diffs = 7*15 = 105, best known is 112
    
    # Try increasing max_val until we find a solution
    max_val_try = lb
    while max_val_try <= lb + 30:  # reasonable range
        sol = greedy_full(n_rows, k, max_val_try, rng)
        if sol is not None:
            best_solution = sol
            best_max_val = max_val_try
            break
        max_val_try += 1
    
    if best_solution is None:
        # Fallback: use a larger max_val
        max_val_try = lb + 30
        while max_val_try <= lb + 100:
            sol = greedy_full(n_rows, k, max_val_try, rng)
            if sol is not None:
                best_solution = sol
                best_max_val = max_val_try
                break
            max_val_try += 1
    
    if best_solution is None:
        # Last resort: very large max_val
        max_val_try = lb + 100
        sol = greedy_full(n_rows, k, max_val_try, rng)
        if sol is not None:
            best_solution = sol
            best_max_val = max_val_try
    
    if best_solution is None:
        # Should not happen, but return something valid
        # Use the original simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Now try to improve with multiple randomized restarts
    current_max = max(max(row) for row in best_solution)
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts * 12345 + 67890)
        
        # Try to build a solution with max_val = current_max
        sol = greedy_full(n_rows, k, current_max, rng)
        if sol is not None:
            new_max = max(max(row) for row in sol)
            if new_max < current_max:
                current_max = new_max
                best_solution = sol
                # Try to improve further
                best_solution, current_max = local_improve(best_solution, k, rng, iterations=100)
        
        # Also try local improvement on current best
        if time.time() < deadline:
            improved, new_max = local_improve([row[:] for row in best_solution], k, rng, iterations=200)
            if new_max < current_max:
                current_max = new_max
                best_solution = improved
    
    return best_solution

# EVOLVE-BLOCK-END
