import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    num_elements = k + 1
    # For n_rows=7, k=6, num_elements=7
    # Total diffs per row = 7*6/2 = 21
    # Total diffs overall = 7 * 21 = 147
    
    # We know the best known scope is 112.
    # We will try to find a solution with scope M starting from the theoretical lower bound 
    # or just start from 105 and go up. 
    # To be safe and fast, we can start from 105.
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def solve_csp(M):
        """
        Solve the CSP for a given maximum scope M.
        Uses randomized backtracking with MRV heuristic.
        """
        used_diffs = set()
        rows = [None] * n_rows
        
        # Pre-compute possible values for each position in a row?
        # Actually, since rows are independent except for used_diffs, we can solve row by row.
        # But the order of solving rows can matter. 
        # Let's stick to solving rows sequentially, but within each row, use MRV on the positions.
        
        # However, a simpler and effective approach for this specific structure:
        # Since each row starts with 0, the "variables" are the positions 1..k.
        # We can treat the entire set of rows as one big CSP, but solving row by row is simpler.
        
        # Let's use a recursive function that fills rows one by one.
        # To escape local optima, we can randomize the order in which we try candidate values.
        
        def backtrack(row_idx):
            if time.time() > DEADLINE:
                return False
            
            if row_idx == n_rows:
                return True
            
            # We need to find a valid row for this index.
            # We will use a nested backtracking for the row itself.
            # To add randomness, we can shuffle the candidate values at each step.
            
            def build_row(pos, current_row, current_max):
                if time.time() > DEADLINE:
                    return False
                
                if pos == num_elements:
                    # Check if this row is valid against global used_diffs
                    # (We check incrementally in the loop below, so this is just a marker)
                    new_diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            d = current_row[j] - current_row[i]
                            if d in used_diffs:
                                return False
                            new_diffs.add(d)
                            
                    rows[row_idx] = current_row
                    used_diffs.update(new_diffs)
                    
                    if backtrack(row_idx + 1):
                        return True
                    
                    rows[row_idx] = None
                    used_diffs.difference_update(new_diffs)
                    return False
                
                min_val = current_row[-1] + 1 if current_row else 1
                # We must leave room for the remaining elements
                max_val = M - (num_elements - pos - 1)
                
                if min_val > max_val:
                    return False
                
                # Generate candidate values
                candidates = list(range(min_val, max_val + 1))
                
                # Randomize candidates to escape local minima
                random.shuffle(candidates)
                
                for val in candidates:
                    # Check if adding val is valid with respect to current_row and used_diffs
                    valid = True
                    for x in current_row:
                        d = val - x
                        if d in used_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    if build_row(pos + 1, current_row + [val], val):
                        return True
                
                return False
            
            return build_row(1, [0], 0)
        
        # Initial call
        return backtrack(0)

    # Strategy:
    # Try scopes from 105 upwards.
    # Since the search is randomized, we might need multiple attempts for a given M.
    # But for M=112, it should be findable quickly.
    # We'll iterate M from 105 to 112.
    
    for M in range(105, 113):
        if time.time() > DEADLINE:
            break
        
        # Try a few random restarts for this M if the first one fails quickly
        # But usually one good randomized search is enough for M=112.
        # Let's just call solve_csp once. If it fails, we move to next M or restart?
        # Given the tight time limit, let's just do one pass per M. 
        # Actually, randomized backtracking is quite powerful. 
        # If it fails for M=105..111, it's likely because no solution exists (or hard to find).
        # For M=112, it should find one.
        
        # To improve robustness, let's do a few restarts if we have time.
        # But keep it simple first.
        
        if solve_csp(M):
            # Verify solution
            all_diffs = []
            for r in rows:
                all_diffs.extend(get_diffs(r))
            if len(all_diffs) == len(set(all_diffs)):
                return rows
            else:
                # Should not happen, but if it does, continue
                continue

    # Fallback: Greedy construction with larger scope
    # This ensures we return *a* valid solution, even if not optimal.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows
