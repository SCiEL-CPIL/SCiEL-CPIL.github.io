import time
import sys
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    num_marks = k + 1
    diffs_per_row = num_marks * (num_marks - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on M is total_diffs because differences must be distinct positive integers starting from 1?
    # Actually, the differences are just distinct, they don't have to be 1..N.
    # But the smallest possible max difference in a row of size k+1 is k (e.g., 0,1,2..k has diffs 1..k).
    # The union of differences must have size n_rows * diffs_per_row.
    # The max difference in any row is at most M.
    # So the total number of distinct differences is at most M (if we could use all 1..M).
    # Actually, the differences are subsets of {1, ..., M}.
    # So we need M >= total_diffs.
    low_bound = total_diffs
    
    # Upper bound: Greedy construction gives a valid upper bound.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < num_marks:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows

    best_solution = greedy_construct()
    if not best_solution:
        return []
    
    best_M = max(max(r) for r in best_solution)
    
    # Precompute all valid rows (Golomb rulers) up to best_M
    # A valid row is a tuple of length num_marks starting with 0, strictly increasing,
    # with all pairwise differences distinct.
    
    valid_rows = []
    
    def generate_valid_rows(current, max_val, limit):
        if time.time() > deadline:
            return
        if len(current) == num_marks:
            # Check distinct differences
            diffs = set()
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    d = current[j] - current[i]
                    if d in diffs:
                        return
                    diffs.add(d)
            # Store row and its difference mask
            # Mask is an integer where bit d is set if difference d is present.
            # d ranges from 1 to max_val.
            mask = 0
            for d in diffs:
                mask |= (1 << d)
            valid_rows.append((tuple(current), mask, max_val))
            return
        
        start = current[-1] + 1
        # We need num_marks - len(current) more elements.
        # The smallest possible next element is start.
        # The largest possible last element is limit.
        # So we need enough room: current[-1] + (num_marks - len(current)) <= limit
        if current[-1] + (num_marks - len(current)) > limit:
            return
            
        for x in range(start, limit + 1):
            current.append(x)
            generate_valid_rows(current, x, limit)
            current.pop()
            
    # Generate up to best_M
    generate_valid_rows([0], 0, best_M)
    
    # Sort valid rows lexicographically for symmetry breaking
    valid_rows.sort(key=lambda x: x[0])
    
    # Search for M from low_bound to best_M
    # We want to find if we can pick n_rows disjoint rows with max element <= M.
    
    def solve_for_M(M):
        # Filter rows with max_val <= M
        candidates = [(row, mask) for (row, mask, mv) in valid_rows if mv <= M]
        
        if len(candidates) < n_rows:
            return None
        
        # Backtracking search
        # To break symmetry, we enforce that the indices of chosen rows in the sorted `candidates` list are increasing.
        chosen_rows = []
        
        def backtrack(start_idx, used_mask):
            if time.time() > deadline:
                return False
                
            if len(chosen_rows) == n_rows:
                return True
            
            # Try each candidate from start_idx onwards
            for i in range(start_idx, len(candidates)):
                row, mask = candidates[i]
                
                # Check disjointness
                if used_mask & mask:
                    continue
                
                # Choose this row
                chosen_rows.append(row)
                
                if backtrack(i + 1, used_mask | mask):
                    return True
                    
                # Undo
                chosen_rows.pop()
            
            return False
            
        if backtrack(0, 0):
            return chosen_rows
        return None

    found_solution = None
    for M in range(low_bound, best_M + 1):
        if time.time() > deadline:
            break
            
        solution = solve_for_M(M)
        if solution:
            found_solution = solution
            best_M = M
            break
            
    if found_solution:
        return [list(r) for r in found_solution]
    else:
        return best_solution

# Test locally
if __name__ == "__main__":
    # n_rows=7, k=5
    res = construct(7, 5)
    if res:
        max_val = max(max(r) for r in res)
        print(f"Found solution with max value: {max_val}")
        # Validate
        used = set()
        valid = True
        for r in res:
            if r[0] != 0:
                valid = False
                break
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        valid = False
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No solution found")
