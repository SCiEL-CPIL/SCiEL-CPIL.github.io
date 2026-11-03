import time
import random
import math
from collections import deque

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    # Precompute all pairs (i, j) with i < j
    pairs = []
    for i in range(row_len):
        for j in range(i + 1, row_len):
            pairs.append((i, j))
    
    # 1. Generate candidate rows for scope S
    # We want rows of length k+1 starting at 0, strictly increasing, max <= S.
    # A row is valid if its internal differences are distinct (which is always true for distinct elements? No, differences must be distinct globally. But within a row, if a,b,c are distinct, a-c = (a-b)+(b-c). So a-c cannot equal a-b or b-c unless one is 0. Since strictly increasing, differences are positive.
    # Wait, within a single row, are differences distinct?
    # a_j - a_i. If i<j<k, a_k - a_i = (a_j - a_i) + (a_k - a_j). Since all terms positive, a_k - a_i is strictly larger than both. So within a single row, all differences are distinct.
    # So we only need to check conflicts BETWEEN rows.
    
    def generate_candidates(S):
        # Generate all valid rows with max entry <= S
        # To keep the list manageable, we can use a greedy or random sampling if S is large,
        # but for S ~ 100-120 and k=6, the number of subsets of {1..S} of size 6 is C(S,6).
        # C(120, 6) is ~ 40 million. Too many to generate all.
        # We need a smarter way to find valid sets of rows.
        
        # Alternative: Backtracking search for the set of rows directly.
        pass

    # Let's use a different approach: 
    # We search for a set of `n_rows` rows.
    # We can iterate over possible "last elements" or just use a global search.
    
    # Given the constraint on time and the difficulty of generating all candidates, 
    # let's use a hybrid: 
    # 1. Start with a greedy construction to get a valid upper bound.
    # 2. Use a local search / simulated annealing to reduce the max value.
    # 3. To make it "structurally different", we will use a **Constraint Satisfaction** approach 
    #    with **Backtracking** and **Forward Checking** to find a solution with a specific max value M.
    #    We will binary search on M? Or just decrement M from the greedy solution?
    
    # Let's try to solve for a specific M by finding n_rows disjoint sets of differences.
    
    # Step 1: Greedy initial solution
    def greedy_initial():
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    current_matrix = greedy_initial()
    best_max = max(max(r) for r in current_matrix)
    
    # Step 2: Try to reduce the scope.
    # We will attempt to find a valid configuration with max value `target_max`.
    # We use a backtracking search that assigns rows one by one.
    # To make it efficient, we need good heuristics for row generation.
    
    # Since generating all rows is hard, we will use a "repair" strategy or a smarter backtracking.
    # Actually, for small k (6) and n_rows (7), the number of differences per row is 21.
    # Total differences = 147.
    # The set of differences must be a subset of {1, ..., M}.
    
    # Let's try to construct the rows one by one using a recursive function that picks a row,
    # removes its differences from the available pool, and recurses.
    
    # To avoid combinatorial explosion, we will limit the search depth and use randomization.
    
    def find_solution_for_scope(M, max_time=5.0):
        """
        Try to find a set of n_rows valid rows with max entry <= M.
        Returns the list of rows or None.
        """
        start_time = time.time()
        
        # Available differences are 1..M
        # We need to pick n_rows disjoint sets of 21 differences each, such that each set
        # can be realized as the difference set of a row of length k+1.
        
        # This is equivalent to:
        # 1. Pick a row.
        # 2. Check if its differences are available.
        # 3. Remove them.
        # 4. Repeat.
        
        # How to pick a row?
        # We can iterate through possible rows? No, too many.
        # We can generate a random row with max <= M and check if it's valid against used diffs.
        
        used_diffs = set()
        rows = []
        
        # We need to be able to backtrack. 
        # A simple random greedy might fail. 
        # Let's use a limited backtracking search.
        
        def backtrack(depth, current_rows, current_used):
            if time.time() - start_time > max_time:
                return None
            
            if depth == n_rows:
                return [r[:] for r in current_rows]
            
            # Generate a few candidate rows
            # To improve success rate, we can bias towards rows with smaller max values?
            # Or just random.
            
            # Heuristic: Try to complete the row such that it uses "small" differences if possible?
            # Actually, large differences are harder to fit in later rows if M is small.
            # So maybe we should use large differences early?
            
            # Let's try random generation of a row.
            # A row is defined by its elements a_1 < ... < a_k (a_0=0).
            # We can sample these elements from 1..M.
            
            # To ensure the row is "valid" internally (which it always is), we just need distinct elements.
            # The constraint is against `current_used`.
            
            # We will try a few random candidates.
            # If we fail, we might need to backtrack.
            
            # For simplicity and speed, let's try a "random restart" greedy within the time limit.
            # But the prompt asks for a structural change. 
            # The "A* / Shortest Path" idea was suggested. 
            # Let's implement a search that treats the "used differences" as the state.
            # But the state space is huge.
            
            # Let's stick to a robust local search that is different from the previous one.
            # The previous one was SA on the matrix.
            # This one will be **Iterated Local Search (ILS)** with **Row Swapping**.
            
            # Actually, let's implement the backtracking with randomization properly.
            
            # We'll generate a candidate row by picking k random numbers from 1..M, sorting them,
            # and checking if the differences are in current_used.
            
            # Optimization: Pre-check if the differences of the candidate row are disjoint from current_used.
            
            # To make backtracking effective, we should order the candidates.
            # But generating candidates is random.
            
            # Let's try up to 100 random candidates at each depth.
            # If none work, return None (failure for this branch).
            # The outer loop will restart with different random seeds.
            
            attempts = 0
            while attempts < 100:
                if time.time() - start_time > max_time:
                    return None
                
                # Generate a random row
                # Sample k distinct integers from 1..M
                # Ensure they are > 0 and < M (since max is M, and we need room for differences? No, max entry is M).
                # The elements must be in 1..M.
                
                # To bias towards valid solutions, we can try to reuse the structure of previous rows?
                # No, let's just sample.
                
                # One issue: if M is close to the theoretical minimum, random sampling is very inefficient.
                # We need a smarter row generator.
                
                # Idea: Generate the row by picking the differences?
                # No, the row is defined by positions.
                
                # Let's just try the random sampling. If M is not too tight, it might work.
                
                # Sample k elements
                # To ensure strict increase and start at 0, we pick a_1..a_k from 1..M.
                # Sort them.
                
                # We can use `random.sample(range(1, M+1), k)`
                
                if M < k:
                    return None
                    
                candidates = random.sample(range(1, M + 1), k)
                candidates.sort()
                row = [0] + candidates
                
                # Check differences
                row_diffs = set()
                valid = True
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        d = row[j] - row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check against used
                if row_diffs & current_used:
                    continue
                
                # Accept this row
                current_rows.append(row)
                current_used |= row_diffs
                
                result = backtrack(depth + 1, current_rows, current_used)
                if result is not None:
                    return result
                
                # Backtrack
                current_used -= row_diffs
                current_rows.pop()
                
                attempts += 1
            
            return None
        
        return backtrack(0, [], set())

    # Step 3: Iteratively try to reduce the scope
    # Start from the greedy solution's max.
    current_max = best_max
    
    # We will try to find a solution for M = current_max - 1, current_max - 2, etc.
    # To avoid getting stuck in a local minimum of the greedy, we also run the local search (SA)
    # but with a different move set: **Row Rebuild**.
    
    # Let's combine:
    # 1. Try to find a solution with M < current_max using the backtracking search.
    # 2. If found, update current_max.
    # 3. If not, run a local search to see if we can improve the current matrix.
    
    # Given the time limit, we will alternate between these.
    
    # Actually, the backtracking search is likely to fail for small M because of the "hole" problem.
    # Let's use the local search as the primary engine, but with a **different move**:
    # Instead of changing one element, we **rebuild a row** from scratch, ensuring it doesn't conflict with others.
    
    def rebuild_row(matrix, row_idx, M):
        """
        Try to replace matrix[row_idx] with a new valid row that has max <= M
        and doesn't conflict with other rows.
        Returns the new row or None.
        """
        # Collect used differences from all other rows
        used_diffs = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            row = matrix[i]
            for a in range(row_len):
                for b in range(a + 1, row_len):
                    used_diffs.add(row[b] - row[a])
        
        # Now find a row of length row_len, max <= M, disjoint from used_diffs.
        # We can use a small backtracking search for just this row.
        
        def find_single_row(depth, current_row, current_diffs):
            if depth == row_len:
                return current_row[:]
            
            if depth == 0:
                # Start with 0
                return find_single_row(1, [0], set())
            
            # Pick next element
            # It must be > current_row[-1] and <= M.
            # Also, the new differences must not be in current_diffs or used_diffs.
            
            last = current_row[-1]
            # To speed up, we can iterate from last+1 to M
            # But M can be 100+.
            
            # Let's try a few random candidates first, then sequential if needed?
            # For a single row, sequential is fine if M is small.
            
            # We'll try a random subset of candidates to avoid bias.
            # Or just iterate.
            
            # Iterate from last+1 to M
            # Optimization: If M is large, this is slow.
            # But we only do this for one row at a time.
            
            # Let's try up to 50 random candidates from (last+1, M]
            # If that fails, try sequential?
            
            # Actually, for a single row, the constraints are local.
            # We can just iterate.
            
            # To make it faster, we can pre-check:
            # The new element x creates differences x - a_i for all i < depth.
            # These must be disjoint from current_diffs and used_diffs.
            
            # Let's just iterate.
            for x in range(last + 1, M + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & (current_diffs | used_diffs):
                    continue
                
                current_row.append(x)
                current_diffs |= new_diffs
                
                result = find_single_row(depth + 1, current_row, current_diffs)
                if result is not None:
                    return result
                
                current_row.pop()
                current_diffs -= new_diffs
            
            return None
        
        return find_single_row(1, [0], set())

    # Main Loop:
    # 1. Start with greedy.
    # 2. While time remains:
    #    a. Try to rebuild a random row with a slightly smaller max constraint?
    #       No, the constraint is global.
    #    b. Try to reduce the max entry of a specific row by rebuilding it with M = current_max - 1?
    #       If we can rebuild a row with a smaller max, the global max might decrease.
    
    # Let's refine the loop:
    # We maintain `current_matrix`.
    # We identify the row with the largest max entry.
    # We try to rebuild that row with a max entry of `current_max - 1`.
    # If successful, we update the matrix.
    # If we can do this for all rows, we might reduce the global max.
    
    # This is a "greedy reduction" approach.
    
    # To escape local minima, we can also allow "expanding" a row if it helps?
    # No, we want to minimize.
    
    # Let's add a randomization: Pick a random row, try to rebuild it with M = current_max - delta.
    
    # We also need to handle the case where no row can be rebuilt.
    # Then we might need to change the structure of other rows to free up differences.
    # This is where a full SA or ILS comes in.
    
    # Let's implement a simple ILS:
    # 1. Current solution.
    # 2. Perturb: Rebuild 1-2 random rows with M = current_max (or slightly larger to allow flexibility?).
    #    Actually, if we rebuild with M = current_max, we might get a different valid row.
    # 3. Local Search: Try to reduce the max of any row by rebuilding it with M-1, M-2, etc.
    # 4. Accept if improved.
    
    # This is structurally different from the element-wise SA.
    
    # Let's implement this.
    
    def get_max_entry(matrix):
        return max(max(r) for r in matrix)

    current_max = get_max_entry(current_matrix)
    best_matrix = [r[:] for r in current_matrix]
    best_max = current_max
    
    rng = random.Random(42)
    
    while time.time() < DEADLINE:
        # Try to improve
        improved = False
        
        # Identify the row with the max entry
        max_row_idx = -1
        max_val = -1
        for i in range(n_rows):
            if current_matrix[i][-1] > max_val:
                max_val = current_matrix[i][-1]
                max_row_idx = i
        
        # Try to rebuild the max row with a smaller max
        # We try M from max_val - 1 down to max_val - 10 (or 0)
        # If we succeed, we accept.
        
        # To avoid getting stuck, we can also try rebuilding other rows.
        
        # Let's try a few random rows
        for _ in range(3):
            r_idx = rng.randint(0, n_rows - 1)
            # Target max for this row
            # We want to reduce the global max, so we focus on rows with high max.
            # But reducing a low-max row might free up differences to reduce a high-max row.
            # So let's try the max row first, then random.
            
            target_max = current_matrix[r_idx][-1] - 1
            if target_max < k: # Must have at least k+1 distinct integers
                continue
            
            new_row = rebuild_row(current_matrix, r_idx, target_max)
            if new_row is not None:
                # Check if this reduces the global max
                old_global_max = get_max_entry(current_matrix)
                current_matrix[r_idx] = new_row
                new_global_max = get_max_entry(current_matrix)
                
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_matrix = [r[:] for r in current_matrix]
                
                improved = True
                break # Restart the loop with the new state
        
        if not improved:
            # Try to perturb to escape local minimum
            # Rebuild a random row with the SAME max (or slightly larger) to get a different configuration
            r_idx = rng.randint(0, n_rows - 1)
            # Allow slightly larger max to enable escape?
            # If we allow larger max, we might increase the global max, which is bad for the score.
            # But it might allow us to find a configuration that CAN be reduced later.
            
            # Let's try rebuilding with the current max of that row.
            target_max = current_matrix[r_idx][-1]
            
            # To ensure we get a DIFFERENT row, we can reject the original row.
            # But `rebuild_row` doesn't know the original row.
            # We can modify it to exclude the original set of differences?
            # Or just try a few times.
            
            # For now, just try to rebuild. If it returns the same row, we might loop.
            # Let's add a check.
            
            old_row = current_matrix[r_idx][:]
            for _ in range(5):
                new_row = rebuild_row(current_matrix, r_idx, target_max)
                if new_row is not None and new_row != old_row:
                    current_matrix[r_idx] = new_row
                    break
            else:
                # If we couldn't find a different row, we are stuck.
                # We might need to do a more drastic perturbation.
                # For now, just wait.
                pass
        
        # Update current_max
        current_max = get_max_entry(current_matrix)
        
        # Safety check
        if time.time() > DEADLINE - 5:
            break

    return best_matrix
