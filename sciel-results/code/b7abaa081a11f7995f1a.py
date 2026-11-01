"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Strategy: Use a randomized local search / simulated annealing approach
    to minimize the largest entry (scope). Start with a greedy construction
    and then try to reduce the scope by swapping/replacing elements.
    """
    
    def greedy_construct():
        """Build rows greedily, one by one, using smallest available marks."""
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
    
    def get_scope(rows):
        """Return the maximum entry across all rows."""
        return max(max(row) for row in rows)
    
    def get_all_diffs(rows):
        """Return the set of all positive differences across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def try_reduce_scope(rows, target_scope):
        """
        Try to reduce the scope by replacing the largest element in a row
        with a smaller one that doesn't conflict.
        Returns True if a reduction was made.
        """
        # Find the row with the largest element
        max_val = get_scope(rows)
        if max_val <= target_scope:
            return False
        
        # Find which row has the max value
        target_row_idx = None
        for i, row in enumerate(rows):
            if max(row) == max_val:
                target_row_idx = i
                break
        
        if target_row_idx is None:
            return False
        
        row = rows[target_row_idx]
        # Compute current differences for this row
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                row_diffs.add(row[j] - row[i])
        
        # Get all diffs from other rows
        other_diffs = set()
        for i, r in enumerate(rows):
            if i == target_row_idx:
                continue
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    other_diffs.add(r[b] - r[a])
        
        # Try to replace the last element with a smaller one
        last = row[-1]
        # The second-to-last element
        second_last = row[-2] if len(row) >= 2 else 0
        
        # Try values from target_scope down to second_last + 1
        for new_val in range(target_scope, second_last, -1):
            if new_val == last:
                continue
            # Check if new_val > second_last (strictly increasing)
            if new_val <= second_last:
                continue
            # Compute new differences involving new_val
            new_diffs = {new_val - a for a in row[:-1]}
            # Remove old differences involving last
            old_diffs = {last - a for a in row[:-1]}
            # Check for conflicts
            remaining_diffs = (row_diffs - old_diffs) | new_diffs
            if remaining_diffs & other_diffs:
                continue
            # Also check that new_diffs don't conflict with each other
            # (they can't since all are new_val - a for distinct a)
            # Apply the change
            row[-1] = new_val
            return True
        
        return False
    
    def local_search(rows, max_iterations=100000):
        """
        Perform Simulated Annealing to minimize the scope.
        """
        import math
        
        current_scope = get_scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        # Initial temperature
        T = 10.0
        cooling_rate = 0.999
        
        for i in range(max_iterations):
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            pos = random.randint(1, len(row) - 1)
            
            old_val = row[pos]
            # Define a search window around the current value to allow exploration
            # But keep it within bounds
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos + 1 < len(row) else current_scope + 20
            
            if max_val < min_val:
                continue
            
            # Randomly choose a new value
            # Bias towards smaller values if we want to reduce scope, 
            # but allow larger values to escape minima
            new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            row[pos] = new_val
            
            if is_valid(rows):
                new_scope = get_scope(rows)
                delta = new_scope - current_scope
                
                if delta < 0:
                    # Improvement
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in rows]
                else:
                    # Worse or equal: accept with probability exp(-delta/T)
                    if delta == 0:
                        # Accept equal scope moves to explore
                        if random.random() < 0.5:
                            current_scope = new_scope
                        else:
                            row[pos] = old_val
                    else:
                        if random.random() < math.exp(-delta / T):
                            current_scope = new_scope
                        else:
                            row[pos] = old_val
            else:
                # Invalid move: revert
                row[pos] = old_val
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return best_rows, best_scope
    
    # Start with greedy construction
    rows = greedy_construct()
    
    # Try to improve with local search
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)
    
    # Run local search with higher iterations
    improved_rows, improved_scope = local_search(rows, max_iterations=100000)
    if improved_scope < best_scope:
        best_rows = improved_rows
        best_scope = improved_scope
    
    return best_rows
# EVOLVE-BLOCK-END
