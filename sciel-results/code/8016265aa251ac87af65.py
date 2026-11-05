import time
import sys
import random
import math
from heapq import heappush, heappop
from collections import Counter

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 elements.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def gaps_to_row(gaps):
        """Convert a list of k gaps to a row of k+1 elements starting at 0."""
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        return row

    def row_to_gaps(row):
        """Convert a row of k+1 elements to k gaps."""
        return [row[i+1] - row[i] for i in range(k)]

    def get_row_differences(row):
        """Get all positive differences in a row."""
        n = len(row)
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def all_differences(rows):
        diffs = []
        for r in rows:
            diffs.extend(get_row_differences(r))
        return diffs

    def count_conflicts(diffs):
        c = Counter(diffs)
        return sum(cnt - 1 for cnt in c.values() if cnt > 1)

    def objective(rows):
        diffs = all_differences(rows)
        conflicts = count_conflicts(diffs)
        max_val = max(max(r) for r in rows)
        return conflicts * 1000000 + max_val

    # Priority-queue-driven greedy construction
    # Each row is built by selecting k gaps. For each gap selection, we compute
    # the set of new differences it introduces and use a priority queue to
    # choose the gap that minimizes collision with already-used differences.
    
    def greedy_with_lookahead():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            # Build one row: start with element 0, then pick k elements
            row = [0]
            # For each position, we need to choose the next element.
            # The new element x adds differences {x - a for a in row}.
            # We want to minimize collisions with used_diffs.
            
            # Use a priority queue: for candidate next values, score = number of collisions
            # But we can't enumerate all values. Instead, use a smart heuristic:
            # Try small increments first, but skip values that cause many collisions.
            
            current_max = 0
            for pos in range(k):
                best_val = None
                best_score = float('inf')
                # Look ahead: try a range of candidate values
                # The minimum next value is current_max + 1
                # We'll try values from current_max+1 upward, but cap the search
                search_limit = current_max + 50  # Look at next 50 candidates
                
                for candidate in range(current_max + 1, search_limit + 1):
                    new_diffs = {candidate - a for a in row}
                    collisions = len(new_diffs & used_diffs)
                    # Also consider: if this creates internal duplicates within the row
                    # (which shouldn't happen since row is strictly increasing)
                    if collisions < best_score:
                        best_score = collisions
                        best_val = candidate
                        if collisions == 0:
                            break  # Can't do better than 0 collisions
                
                if best_val is None:
                    best_val = current_max + 1
                
                row.append(best_val)
                new_diffs = {best_val - a for a in row[:-1]}
                used_diffs |= new_diffs
                current_max = best_val
            
            rows.append(row)
        return rows

    # Initial solution
    current_rows = greedy_with_lookahead()
    best_rows = [r[:] for r in current_rows]
    best_obj = objective(best_rows)
    current_obj = best_obj

    # Convert to gap representation for local search
    current_gaps = [row_to_gaps(r) for r in current_rows]
    best_gaps = [g[:] for g in current_gaps]

    def gaps_objective(gaps_list):
        """Compute objective from gap representation."""
        rows = [gaps_to_row(g) for g in gaps_list]
        diffs = all_differences(rows)
        conflicts = count_conflicts(diffs)
        max_val = max(gaps_to_row(g)[-1] for g in gaps_list)
        return conflicts * 1000000 + max_val

    # Local search in gap-space
    # Move: pick a random row, pick a random gap position, adjust the gap by ±delta
    # This is a structural change: we're optimizing the gaps directly rather than elements
    
    temp = 500.0
    cooling = 0.995
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset temperature and try harder moves
            temp = max(temp, 300.0)
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 1.3)

        new_gaps = [g[:] for g in current_gaps]
        row_idx = random.randint(0, n_rows - 1)
        gap_idx = random.randint(0, k - 1)

        # Determine valid range for this gap
        # The gap must be >= 1
        # And the resulting row must stay within reasonable bounds
        # For now, allow gap to range from 1 to some upper bound
        old_gap = new_gaps[row_idx][gap_idx]
        
        # Try a random new gap value
        # Bias toward small changes but allow occasional large jumps
        if random.random() < 0.7:
            # Small change
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            new_gap = old_gap + delta
        else:
            # Larger jump
            new_gap = random.randint(1, old_gap + 20)
        
        if new_gap < 1:
            new_gap = 1
        
        new_gaps[row_idx][gap_idx] = new_gap
        
        # Compute new objective
        new_obj = gaps_objective(new_gaps)
        
        delta_obj = new_obj - current_obj
        
        if delta_obj < 0 or random.random() < math.exp(-delta_obj / temp):
            current_gaps = new_gaps
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_gaps = [g[:] for g in current_gaps]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temp *= cooling

    # Final: convert best gaps back to rows
    best_rows = [gaps_to_row(g) for g in best_gaps]
    
    # Verify validity
    diffs = all_differences(best_rows)
    if len(diffs) != len(set(diffs)):
        # Fallback: return greedy solution if it's valid
        greedy_rows = greedy_with_lookahead()
        greedy_diffs = all_differences(greedy_rows)
        if len(greedy_diffs) == len(set(greedy_diffs)):
            return greedy_rows
    
    return best_rows
