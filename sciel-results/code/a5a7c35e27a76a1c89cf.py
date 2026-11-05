import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Constraint-driven Constructive Search (CSP) with Backtracking.
    We build the rows one by one. 
    The "state" is the set of used differences.
    For each new row, we search for a valid configuration (set of k+1 numbers)
    such that:
    1. It starts at 0 and is strictly increasing.
    2. Its internal differences do not overlap with the global set of used differences.
    3. To minimize scope, we prioritize rows with smaller maximum elements.
    
    To make this efficient within the time limit:
    1. We pre-compute a list of "candidate rows" for small scopes.
       A candidate row is a tuple of (row_tuple, diff_set, max_val).
    2. We sort candidates by max_val (scope contribution).
    3. We use backtracking to select n_rows non-conflicting candidates.
    4. If we get stuck, we increase the maximum allowed scope for the next row 
       or restart with a different ordering/randomization.
       
    Since the optimal scope is ~112 and n_rows=7, k=6, the number of possible rows
    with max_val <= 112 is manageable if we prune aggressively.
    
    Optimization:
    - Instead of generating all rows up to M, we generate them on the fly or 
      pre-generate a large pool.
    - Key insight: The set of differences for a row is determined by the row.
      Two rows conflict if their diff sets intersect.
    - We want to find a set of n_rows rows with pairwise disjoint diff sets 
      minimizing the maximum element across all rows.
    
    Implementation Details:
    - Pre-compute all valid rows with max element up to a limit (e.g., 150).
    - Group rows by their max element.
    - Use a backtracking search to pick n_rows rows from this pool such that 
      their diff sets are disjoint.
    - To handle the "minimize largest entry" objective, we iterate over possible 
      global max M from lower bound upwards. For a fixed M, we only consider rows 
      with max element <= M.
    """
    start_time = time.time()
    DEADLINE = 145.0

    num_elements = k + 1
    total_diffs_per_row = num_elements * (num_elements - 1) // 2

    # Pre-compute all possible rows with max element up to MAX_M
    # For k=6, n_rows=7, optimal M is 112. Let's go up to 130 to be safe.
    MAX_M = 130
    
    # row_data: list of (max_val, row_tuple, frozenset_of_diffs)
    # We only care about rows that start with 0.
    all_candidate_rows = []
    
    # Generate all subsets of {1, ..., MAX_M-1} of size k-1? 
    # No, that's too many. 
    # Better: Generate rows by choosing the last element m, then the k-1 intermediate elements.
    # Actually, for a fixed max element m, the row is (0, x1, ..., x_{k-1}, m).
    # The number of such rows for a fixed m is C(m-1, k-1).
    # For m=112, k-1=5, C(111, 5) is approx 111^5 / 120 ~ 1.6e8. This is too many to pre-compute all.
    
    # Alternative Strategy: On-the-fly generation with Backtracking.
    # We will use a recursive search to build the list of rows.
    # State: current_used_diffs (set), rows_selected (list)
    # Action: Choose a new row.
    # To find a new row, we need to search for a set of points {0, x1, ..., x_{k-1}, m} 
    # such that its diff set is disjoint from current_used_diffs.
    
    # To speed up the search for a single row given a "blocked" set of diffs:
    # We can use a backtracking search over the positions.
    # Since we want to minimize the global max, we should try to find rows with small m first.
    
    def find_row_with_mask(blocked_diffs, max_m_allowed):
        """
        Find a row [0, x1, ..., x_{k-1}, m] with m <= max_m_allowed
        such that no difference in the row is in blocked_diffs.
        Returns the row tuple and its diff set, or None.
        """
        # We search for m from 1 to max_m_allowed.
        # For each m, we search for x1...x_{k-1}.
        # This is a CSP.
        
        # To make it faster, we can randomize the order of m or x's if we are in a heuristic phase.
        # But for determinism and correctness, let's do a structured search.
        
        # Optimization: The differences involving 0 are just the elements themselves.
        # So x1, ..., x_{k-1}, m must NOT be in blocked_diffs.
        # Also, differences between xi and xj must not be in blocked_diffs.
        
        # Let's try a backtracking approach for the elements x1...x_{k-1} given a fixed m.
        
        def backtrack_elements(pos, current_elements, current_diffs):
            if pos == k - 1:
                # Check if m is valid
                if m in blocked_diffs:
                    return None
                # Check diffs between m and existing elements
                for e in current_elements:
                    if m - e in blocked_diffs:
                        return None
                # Check if m is in current_diffs? No, m is the max, so m is a new diff (m-0).
                # Wait, current_diffs contains diffs among current_elements and 0.
                # The new diffs are m - e for all e in [0] + current_elements.
                # We already checked m-0 (m) and m-e.
                # Are there any other constraints?
                # The diff set of the full row is current_diffs U {m - e for e in [0]+current_elements}.
                # We need this full set to be disjoint from blocked_diffs.
                # We have ensured:
                # 1. e not in blocked_diffs (checked when adding e)
                # 2. e_i - e_j not in blocked_diffs (checked when adding e)
                # 3. m not in blocked_diffs
                # 4. m - e not in blocked_diffs
                # So the union is disjoint.
                
                full_row = (0,) + tuple(sorted(current_elements)) + (m,)
                # Re-calculate full diff set for return
                full_diffs = set()
                for i in range(len(full_row)):
                    for j in range(i+1, len(full_row)):
                        full_diffs.add(full_row[j] - full_row[i])
                return (full_row, frozenset(full_diffs))

            # Choose next element x
            # x must be > last element (or 0 if first)
            # x < m
            # x not in blocked_diffs
            # x - last_element not in blocked_diffs
            # Also, for any previously chosen element y, x - y not in blocked_diffs.
            
            start_val = current_elements[-1] + 1 if current_elements else 1
            end_val = m - 1
            
            # To improve search, we can iterate x in a specific order.
            # Smaller x's might be better for packing? Or larger?
            # Let's try a mix.
            
            for x in range(start_val, end_val + 1):
                if x in blocked_diffs:
                    continue
                
                valid = True
                # Check diffs with all existing elements
                for e in current_elements:
                    if x - e in blocked_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Also, the diff x itself (x-0) is checked by "x in blocked_diffs".
                
                current_elements.append(x)
                # We don't strictly need to maintain current_diffs set for pruning if we check on the fly,
                # but it helps if we want to avoid re-checking.
                # However, checking on the fly is O(k) which is small.
                
                res = backtrack_elements(pos + 1, current_elements, current_diffs)
                if res:
                    return res
                current_elements.pop()
            
            return None
        
        # We iterate m from 1 to max_m_allowed.
        # To allow for heuristic diversity, we can randomize the order of m in some runs,
        # but for the first pass, let's go from smallest to largest to find tight solutions.
        
        for m in range(1, max_m_allowed + 1):
            if m in blocked_diffs:
                continue
            res = backtrack_elements(0, [], set())
            if res:
                return res
        
        return None

    def solve(max_scope):
        """
        Try to find n_rows rows with max element <= max_scope.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []
        
        def backtrack(idx):
            if idx == n_rows:
                return True
            
            # Find a row for this index
            # To allow for different solutions, we can try multiple candidates.
            # However, find_row_with_mask returns the first valid one found.
            # To explore the space, we might need to modify find_row_with_mask to yield multiple solutions
            # or use a different heuristic.
            
            # For now, let's just find one. If it fails, we need to backtrack further.
            # But since we build rows sequentially, if we pick a "bad" row early, we might get stuck later.
            # This is a greedy approach with backtracking on the row selection.
            
            # To make it more robust, we should try to find a row that leaves the most options for future rows.
            # This is hard to estimate.
            
            # Alternative: Randomized restarts.
            # We will wrap this in a loop with randomization.
            
            res = find_row_with_mask(used_diffs, max_scope)
            if res is None:
                return False
            
            row, diffs = res
            
            # Heuristic: Check if this choice is likely to lead to a dead end?
            # Simple heuristic: If the row uses very small numbers, it blocks many small diffs.
            # If it uses large numbers, it blocks large diffs.
            
            # Accept the row
            rows.append(row)
            used_diffs.update(diffs)
            
            if backtrack(idx + 1):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.difference_update(diffs)
            
            # To try a different row, we would need find_row_with_mask to support "next solution".
            # For simplicity in this first version, we rely on randomized restarts of the whole process
            # or a more sophisticated search.
            
            # Let's add a simple randomization: 
            # In find_row_with_mask, we can shuffle the order of m or x.
            # Since we can't easily modify the inner function state for backtracking, 
            # we will implement a "randomized" version of the whole solve.
            
            return False
        
        return backtrack(0)

    # Since the simple backtracking might get stuck in local minima (bad first row choice),
    # we will use a randomized approach:
    # 1. Try deterministic search (smallest m first).
    # 2. If that fails, try randomized order of m.
    
    best_rows = None
    
    # Lower bound for M
    # Counting bound: 105.
    # We try M from 105 to MAX_M.
    
    for M in range(105, MAX_M + 1):
        if time.time() > DEADLINE:
            break
            
        # Try deterministic first
        res = solve(M)
        if res:
            best_rows = rows[:] # 'rows' is populated by solve
            break
        
        # If deterministic fails, try a few randomized versions?
        # The current solve() doesn't support randomization easily.
        # Let's modify find_row_with_mask to accept a random seed or order.
        
        # Actually, let's implement a proper randomized search in the main loop.
        # We'll create a helper that runs the backtracking with a specific random seed.
        
        # To save time, we only do this if deterministic fails.
        # But deterministic is O(1) call. The cost is inside.
        
        # Let's try 10 randomized restarts for each M if deterministic fails.
        # We need to inject randomness into find_row_with_mask.
        
        def solve_randomized(M, seed):
            random.seed(seed)
            used_diffs = set()
            rows = []
            
            def find_row_randomized(blocked_diffs, max_m_allowed):
                # Shuffle the range of m?
                # Or shuffle the range of x?
                
                # Let's shuffle the candidate m values.
                m_candidates = list(range(1, max_m_allowed + 1))
                random.shuffle(m_candidates)
                
                def backtrack_elements(pos, current_elements):
                    if pos == k - 1:
                        if m in blocked_diffs:
                            return None
                        for e in current_elements:
                            if m - e in blocked_diffs:
                                return None
                        full_row = (0,) + tuple(sorted(current_elements)) + (m,)
                        full_diffs = set()
                        for i in range(len(full_row)):
                            for j in range(i+1, len(full_row)):
                                full_diffs.add(full_row[j] - full_row[i])
                        return (full_row, frozenset(full_diffs))
                    
                    start_val = current_elements[-1] + 1 if current_elements else 1
                    end_val = m - 1
                    
                    # Shuffle x candidates too?
                    x_candidates = list(range(start_val, end_val + 1))
                    random.shuffle(x_candidates)
                    
                    for x in x_candidates:
                        if x in blocked_diffs:
                            continue
                        valid = True
                        for e in current_elements:
                            if x - e in blocked_diffs:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        current_elements.append(x)
                        res = backtrack_elements(pos + 1, current_elements)
                        if res:
                            return res
                        current_elements.pop()
                    return None
                
                for m in m_candidates:
                    if m in blocked_diffs:
                        continue
                    res = backtrack_elements(0, [])
                    if res:
                        return res
                return None
            
            def backtrack(idx):
                if idx == n_rows:
                    return True
                
                res = find_row_randomized(used_diffs, M)
                if res is None:
                    return False
                
                row, diffs = res
                rows.append(row)
                used_diffs.update(diffs)
                
                if backtrack(idx + 1):
                    return True
                
                rows.pop()
                used_diffs.difference_update(diffs)
                return False
            
            return backtrack(0)

        # We didn't find it deterministically. Try randomized.
        # But wait, the deterministic `solve` above uses a nested `rows` variable which is shared.
        # Let's clean up the code structure to avoid side effects.
        
        # Re-implementing a clean solver function for both deterministic and randomized.
        
        def run_solver(M, use_randomization=False, seed=0):
            if use_randomization:
                random.seed(seed)
            
            used_diffs = set()
            found_rows = []
            
            def _find_row(blocked, max_m, rand):
                if rand:
                    m_list = list(range(1, max_m + 1))
                    random.shuffle(m_list)
                else:
                    m_list = range(1, max_m + 1)
                
                for m in m_list:
                    if m in blocked:
                        continue
                    
                    def _bt(pos, elems):
                        if pos == k - 1:
                            if m in blocked:
                                return None
                            for e in elems:
                                if m - e in blocked:
                                    return None
                            row = (0,) + tuple(sorted(elems)) + (m,)
                            diffs = set()
                            for i in range(len(row)):
                                for j in range(i+1, len(row)):
                                    diffs.add(row[j] - row[i])
                            return (row, frozenset(diffs))
                        
                        start_v = elems[-1] + 1 if elems else 1
                        end_v = m - 1
                        if rand:
                            x_list = list(range(start_v, end_v + 1))
                            random.shuffle(x_list)
                        else:
                            x_list = range(start_v, end_v + 1)
                        
                        for x in x_list:
                            if x in blocked:
                                continue
                            ok = True
                            for e in elems:
                                if x - e in blocked:
                                    ok = False
                                    break
                            if not ok:
                                continue
                            
                            elems.append(x)
                            res = _bt(pos + 1, elems)
                            if res:
                                return res
                            elems.pop()
                        return None
                    
                    res = _bt(0, [])
                    if res:
                        return res
                return None

            def _backtrack(idx):
                if idx == n_rows:
                    return True
                
                res = _find_row(used_diffs, M, use_randomization)
                if res is None:
                    return False
                
                row, diffs = res
                found_rows.append(row)
                used_diffs.update(diffs)
                
                if _backtrack(idx + 1):
                    return True
                
                found_rows.pop()
                used_diffs.difference_update(diffs)
                return False
            
            if _backtrack(0):
                return found_rows
            return None

        # Try deterministic
        res = run_solver(M, use_randomization=False)
        if res:
            best_rows = res
            break
            
        # Try randomized if time permits
        if time.time() < DEADLINE - 10:
            for s in range(20):
                if time.time() > DEADLINE:
                    break
                res = run_solver(M, use_randomization=True, seed=42+s)
                if res:
                    best_rows = res
                    break
            if best_rows:
                break

    if best_rows:
        # Convert tuples to lists
        return [list(r) for r in best_rows]
    
    # Fallback: Greedy
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

if __name__ == "__main__":
    res = construct(7, 6)
    print(f"Scope: {max(max(r) for r in res)}")
    all_diffs = []
    for r in res:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
    print(f"Valid: {len(all_diffs) == len(set(all_diffs))}")
    print(res)
