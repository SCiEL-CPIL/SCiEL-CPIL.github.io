"""
Difference triangle set (4,7) solver.
Strategy: Iterative improvement with randomized greedy construction and local search.
"""
import time
import random
import sys

# Deadline guard: evaluator kills at 150s, we aim to return well before that.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def _construct_greedy(n_rows, k, random_order=False):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences across all rows must be distinct.
    Returns a list of rows, or None if it fails (shouldn't happen with greedy).
    """
    used_diffs = set()
    rows = []
    
    # If random_order, we might shuffle the order in which we pick the next number?
    # Actually, the greedy step is deterministic given the current state.
    # To introduce randomness, we can randomize the *starting point* of the search for the next number,
    # or use a different heuristic for tie-breaking (though there are no ties in standard greedy).
    # Better: Randomize the order in which we fill the rows? No, rows are independent except for used_diffs.
    # Let's stick to a simple randomized restart strategy where we might skip some small numbers
    # or use a different priority.
    
    # Standard greedy: for each row, start at 0, find smallest x such that {x-a for a in row} disjoint from used_diffs.
    # To make it randomized, we can occasionally pick a larger x than the minimum available.
    
    for r in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            # Find the smallest valid next element
            found = False
            # Optimization: start from current_max + 1
            candidate = current_max + 1
            while True:
                new_diffs = {candidate - a for a in row}
                if not new_diffs & used_diffs:
                    row.append(candidate)
                    used_diffs |= new_diffs
                    current_max = candidate
                    found = True
                    break
                candidate += 1
            
            if not found:
                return None # Should not happen

        rows.append(row)
    
    return rows

def _local_search(rows, n_rows, k, time_limit=5.0):
    """
    Try to improve the rows by swapping elements or moving them to smaller values.
    This is a heuristic local search.
    """
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    best_score = max(max(row) for row in rows)
    current_rows = [list(r) for r in rows]
    
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try to reduce the largest element in the row with the largest max
        # Sort rows by max element descending
        row_indices = sorted(range(n_rows), key=lambda i: -current_rows[i][-1])
        
        for r_idx in row_indices:
            row = current_rows[r_idx]
            if row[-1] <= best_score - 1:
                continue
                
            # Try to move the last element to a smaller value
            # Or try to move any element to a smaller value
            for pos in range(len(row) - 1, 0, -1):
                original_val = row[pos]
                # Try values from row[pos-1] + 1 up to original_val - 1
                for new_val in range(row[pos-1] + 1, original_val):
                    # Check if new_val is valid
                    # Remove old diffs involving row[pos]
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original_val - row[i])
                    
                    # Tentatively remove
                    tentative_used = used_diffs - old_diffs
                    
                    # Check new diffs
                    new_diffs = {new_val - row[i] for i in range(pos)}
                    if new_diffs & tentative_used:
                        continue
                    
                    # Also need to check diffs between new_val and elements after pos?
                    # No, row is sorted. new_val > row[pos-1].
                    # But we need to ensure new_val < row[pos+1] if pos < len(row)-1
                    if pos < len(row) - 1 and new_val >= row[pos+1]:
                        continue
                        
                    # If valid, apply
                    # Update used_diffs
                    used_diffs = tentative_used | new_diffs
                    row[pos] = new_val
                    # Ensure sorted order is maintained? 
                    # Since new_val > row[pos-1] and new_val < row[pos+1] (or is last), it's fine.
                    
                    if row[-1] < best_score:
                        best_score = row[-1]
                    improved = True
                    break # Restart search for this row or next? Let's break inner loop and restart outer
                if improved:
                    break
            if improved:
                break # Restart outer loop to re-evaluate
        
    return current_rows, best_score

def construct(n_rows, k):
    global _START_TIME, _DEADLINE
    
    # Check time
    if time.time() > _DEADLINE - 5.0:
        # Not enough time, return a quick greedy solution
        return _construct_greedy(n_rows, k)
    
    best_rows = None
    best_score = float('inf')
    
    # Initial greedy
    rows = _construct_greedy(n_rows, k)
    if rows is None:
        return None
    best_score = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]
    
    # Local search on the initial solution
    if time.time() < _DEADLINE - 10.0:
        improved_rows, improved_score = _local_search(best_rows, n_rows, k, time_limit=10.0)
        if improved_score < best_score:
            best_rows = improved_rows
            best_score = improved_score
    
    # Try randomized restarts
    # The standard greedy is deterministic. To get different solutions, we can introduce randomness.
    # One way: when choosing the next number, instead of always picking the smallest, 
    # pick from a set of valid candidates with some probability of picking a larger one.
    
    # However, for this specific problem (4,7), the greedy might already be quite good or close to optimal.
    # Let's try a different heuristic: "Backtracking" is too slow.
    # "Simulated Annealing" on the positions?
    
    # Let's try a simple random perturbation + local search loop.
    remaining_time = _DEADLINE - 15.0 - time.time()
    if remaining_time > 1.0:
        attempts = 0
        while time.time() < _DEADLINE - 15.0:
            attempts += 1
            # Create a perturbed version of the best solution
            # Pick a random row and a random position, and try to move it to a random valid spot
            # Or just re-run greedy with a slight bias?
            
            # Let's try a different construction strategy:
            # Instead of filling rows one by one, interleave?
            # Or, use a different initial greedy: "Largest First"?
            # No, the constraint is on differences.
            
            # Let's try "Randomized Greedy":
            # When finding the next element, if there are multiple valid candidates in a window, pick one at random.
            # But greedy always picks the smallest.
            
            # Alternative: Start with the greedy solution, then randomly swap two elements in different rows?
            # This is complex to maintain validity.
            
            # Let's stick to Local Search on the current best, but with more aggressive moves.
            # The previous local search only moved elements to smaller values.
            
            # Let's try a restart with a different random seed for a randomized greedy.
            # How to randomize greedy?
            # Instead of picking the smallest valid x, pick a random valid x from [smallest_valid, smallest_valid + margin].
            
            def randomized_greedy(margin=5):
                used_diffs = set()
                rows = []
                for r in range(n_rows):
                    row = [0]
                    current_max = 0
                    while len(row) < k + 1:
                        # Find smallest valid
                        candidate = current_max + 1
                        valid_candidates = []
                        # We only look a little ahead to keep it fast
                        limit = current_max + 1 + margin
                        while candidate <= limit:
                            new_diffs = {candidate - a for a in row}
                            if not new_diffs & used_diffs:
                                valid_candidates.append(candidate)
                            candidate += 1
                        
                        if not valid_candidates:
                            # Fall back to standard greedy search
                            candidate = current_max + 1
                            while True:
                                new_diffs = {candidate - a for a in row}
                                if not new_diffs & used_diffs:
                                    valid_candidates.append(candidate)
                                    break
                                candidate += 1
                            
                        # Pick randomly from valid candidates
                        chosen = random.choice(valid_candidates)
                        new_diffs = {chosen - a for a in row}
                        row.append(chosen)
                        used_diffs |= new_diffs
                        current_max = chosen
                    
                    rows.append(row)
                return rows
            
            # Try a few randomized greedies
            if attempts % 10 == 0: # Do local search every 10 attempts
                pass
            
            rows = randomized_greedy(margin=10)
            score = max(max(r) for r in rows)
            
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try to improve this new best
                if time.time() < _DEADLINE - 20.0:
                    improved_rows, improved_score = _local_search(best_rows, n_rows, k, time_limit=2.0)
                    if improved_score < best_score:
                        best_rows = improved_rows
                        best_score = improved_score
                        
            # If we've found a very good solution, we can stop early?
            # The counting bound is 112. Best known is 134.
            # If we get close to 112, we might be done.
            if best_score <= 115:
                break

    return best_rows

# For testing
if __name__ == "__main__":
    rows = construct(4, 7)
    if rows:
        print(rows)
        max_val = max(max(r) for r in rows)
        print(f"Max value: {max_val}")
        
        # Verify
        diffs = set()
        valid = True
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        valid = False
                        print(f"Duplicate diff {d}")
                    diffs.add(d)
        print(f"Valid: {valid}")
