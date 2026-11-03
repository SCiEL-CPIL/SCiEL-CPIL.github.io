import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    # Precompute all pairs (i, j) for indices in a row
    # These pairs define the differences that must be unique
    pairs = []
    for i in range(row_len):
        for j in range(i + 1, row_len):
            pairs.append((i, j))
    
    n_pairs = len(pairs)
    total_diffs_needed = n_rows * n_pairs

    # We need to assign values to rows such that:
    # 1. Each row is strictly increasing, starts at 0.
    # 2. All differences a[j] - a[i] for i<j in any row are distinct across all rows.
    
    # Strategy:
    # We will build the rows one by one.
    # To minimize the max entry, we should try to keep values small.
    # However, the order in which we build rows matters.
    # We can use a backtracking search where we choose the "next" row to fill.
    # Actually, since rows are symmetric in the constraint (disjoint diffs), 
    # we can fix an order of construction but randomize the "difficulty" or use MRV.
    
    # A better approach for minimizing max:
    # Binary search on M (the maximum allowed value).
    # Check if there exists a valid set of n_rows rows with max entry <= M.
    # If yes, try smaller M. If no, increase M.
    
    # The counting bound is 105. Best known is 112.
    # Let's start checking from a lower bound.
    # Lower bound: We need total_diffs_needed distinct differences.
    # The smallest possible differences are 1, 2, ..., total_diffs_needed.
    # If we use differences 1..D, the max value in a row with n elements is at least n-1 (if diff 1 is used).
    # But the differences are constrained by the row structure.
    
    # Let's implement a CSP solver for a fixed M.
    # Variables: The entries of the rows (excluding the first 0).
    # But this is a huge state space.
    
    # Alternative CSP: 
    # The "differences" are the key. 
    # Let's select a set of n_rows * n_pairs distinct differences.
    # Then, can we partition them into n_rows sets of n_pairs differences 
    # such that each set can be realized by a row of length row_len?
    
    # This is also complex.
    
    # Let's stick to a row-by-row backtracking with strong pruning.
    # State:
    # - used_diffs: set of all differences used so far.
    # - rows: list of completed rows.
    # - current_row: the row currently being built (partial).
    # - M: the current upper bound on the maximum entry.
    
    # To make it efficient:
    # 1. Precompute for a given M, all valid rows (lists of length row_len, starting 0, ending <= M) 
    #    that have unique internal differences? No, internal diffs are always unique in a strictly increasing row?
    #    Wait. In a row [a0, a1, ..., ak], diffs are aj - ai.
    #    Are they always unique? No. E.g., [0, 1, 3, 4]. Diffs: 1, 3, 4, 2, 3, 1. 
    #    3 appears twice (3-0 and 4-1). 1 appears twice (1-0 and 4-3).
    #    So a row itself must be a Sidon set (B2 sequence) to have unique internal differences.
    #    Actually, the problem states "all differences between two entries of the same row are distinct over all rows".
    #    This implies differences within a single row must be distinct from each other?
    #    "distinct over all rows" usually means the set of all differences from all rows has no duplicates.
    #    Therefore, differences within a single row MUST be distinct.
    #    So each row must be a Sidon set.
    
    # This is a crucial insight!
    # Step 1: Generate all Sidon sets of length row_len with max entry <= M.
    # Step 2: Find a subset of n_rows of these sets such that their difference sets are disjoint.
    
    # This reduces the problem to:
    # 1. Enumerate candidate rows (Sidon sets) up to M.
    # 2. Solve a Set Packing / Independent Set problem on the candidates, where edges represent conflicting differences.
    
    # Given n_rows=7, k=6 (row_len=7).
    # Number of Sidon sets of length 7 with max <= 112 is likely manageable.
    
    def is_sidon(row):
        diffs = set()
        for i in range(row_len):
            for j in range(i + 1, row_len):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def generate_sidon_rows(M):
        """
        Generate all strictly increasing sequences of length row_len starting at 0,
        with max <= M, that are Sidon sets.
        """
        candidates = []
        # Backtracking to generate rows
        current = [0]
        
        def backtrack(last_val, depth):
            if time.time() > DEADLINE:
                return
            if depth == row_len:
                # Check if Sidon
                if is_sidon(current):
                    candidates.append(current[:])
                return
            
            # Next value must be > last_val and <= M
            # Also, to be a Sidon set, we need to check partial constraints?
            # It's faster to just generate and filter, or prune early.
            # Pruning: If we add a new value v, the new diffs are v - current[i].
            # These must not conflict with existing diffs in the partial row.
            
            existing_diffs = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    existing_diffs.add(current[j] - current[i])
            
            start = last_val + 1
            # Optimization: The smallest possible next value is last_val + 1.
            # The largest is M.
            # But we can prune if the remaining slots cannot be filled?
            # For row_len=7, M=112, the space is not too huge.
            
            for v in range(start, M + 1):
                # Check new diffs
                new_diffs = {v - x for x in current}
                if new_diffs & existing_diffs:
                    continue
                current.append(v)
                backtrack(v, depth + 1)
                current.pop()
                
        backtrack(0, 1)
        return candidates

    # We need to find 7 disjoint rows from candidates.
    # This is a Set Packing problem.
    # We can use a greedy approach with backtracking.
    
    def solve_set_packing(candidates, n_rows_needed, used_diffs, current_solution, depth):
        if time.time() > DEADLINE:
            return None
        
        if depth == n_rows_needed:
            return current_solution[:]
        
        # Heuristic: Try candidates that use "small" differences first? 
        # Or those that conflict with the fewest other candidates?
        # For now, iterate. To avoid symmetry, we can enforce an order on the rows?
        # Since rows are indistinguishable, we can require row[0] <= row[1] ... in terms of some metric?
        # Or just pick the first valid one.
        
        # To improve performance, we can sort candidates by their max value (smallest first)
        # This encourages smaller max values.
        
        for i, row in enumerate(candidates):
            # Check if this row conflicts with used_diffs
            row_diffs = set()
            valid = True
            for a in range(row_len):
                for b in range(a + 1, row_len):
                    d = row[b] - row[a]
                    if d in used_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                new_used_diffs = used_diffs | row_diffs
                result = solve_set_packing(candidates[i+1:], n_rows_needed, new_used_diffs, current_solution + [row], depth + 1)
                if result is not None:
                    return result
        
        return None

    # Binary search on M
    # Lower bound: 
    # We need 7 * 21 = 147 distinct differences.
    # The differences are integers >= 1.
    # The smallest 147 integers are 1..147.
    # If we use differences 1..147, what is the min max value?
    # This is a lower bound on the max difference, not necessarily the max entry.
    # But max entry >= max difference.
    # So M >= 147? No, differences can be larger than the max entry? No, max diff <= max entry.
    # So max entry >= max difference used.
    # The max difference used must be at least the 147th smallest available difference?
    # No, we just need 147 distinct differences. They can be anything.
    # But to minimize the max entry, we want small differences.
    # The counting bound 105 comes from: 
    # If max entry is M, the number of possible differences is M(M+1)/2? No.
    # The number of distinct differences available in [1, M] is M.
    # Wait, differences are in [1, M]. There are M possible values.
    # We need 147 distinct values. So M >= 147?
    # The prompt says "simple counting bound is 105".
    # Let's re-read carefully.
    # "all differences between two entries of the same row are distinct over all rows"
    # Number of differences per row: k(k+1)/2 = 6*7/2 = 21.
    # Total differences: 7 * 21 = 147.
    # The differences are integers.
    # The maximum possible difference in a row with max entry M is M (M-0).
    # So all differences must be in {1, 2, ..., M}.
    # Thus, we need 147 distinct integers from {1, ..., M}.
    # So M >= 147.
    # Why is the bound 105?
    # Ah, maybe I am misinterpreting "counting bound".
    # Or maybe the differences don't have to be distinct from 1..M?
    # "distinct over all rows" means no difference value appears twice in the entire set of all differences.
    # So yes, 147 distinct integers.
    # If the max entry is M, the max difference is M.
    # So we need 147 distinct integers in [1, M].
    # So M >= 147.
    # But the prompt says "best known set has largest entry 112".
    # This contradicts M >= 147.
    # Let's check the number of pairs again.
    # k=6. Row has 7 entries.
    # Pairs (i, j) with i < j.
    # Number of pairs = 7 * 6 / 2 = 21.
    # 7 rows * 21 pairs = 147 differences.
    # If max entry is 112, differences are in [1, 112].
    # 112 < 147. Pigeonhole principle says it's impossible.
    
    # Is it possible that differences are not required to be distinct within a row?
    # "all differences between two entries of the same row are distinct over all rows"
    # This phrasing is ambiguous.
    # Interpretation A: The set of all differences from all rows is a set of distinct integers. (My previous interpretation)
    # Interpretation B: For any two rows, their difference sets are disjoint. (Same as A)
    # Interpretation C: "distinct over all rows" modifies "differences between two entries of the same row".
    # Maybe it means: Within a single row, differences are distinct? AND across rows, they are distinct?
    # If Interpretation A is correct, M >= 147.
    # But the prompt cites a paper with M=112.
    # Let's check the paper title or context if possible? No.
    # Let's re-read the prompt's "counting bound is 105".
    # If the bound is 105, then 105 >= 147 is false.
    # So my calculation of 147 distinct differences must be wrong, or the constraint is different.
    
    # Let's look at k=6.
    # Maybe "differences between two entries" means something else?
    # No, a_j - a_i.
    
    # Is it possible that the rows are not required to have distinct differences WITHIN the row?
    # "all differences ... are distinct over all rows"
    # If I have row 1: [0, 1, 3]. Diffs: 1, 3, 2.
    # Row 2: [0, 1, 3]. Diffs: 1, 3, 2.
    # If "distinct over all rows" means the global set of differences has no duplicates, then Row 1 and Row 2 cannot be identical.
    # But if Row 1 is [0, 1, 3] and Row 2 is [0, 2, 4]?
    # Row 1 diffs: {1, 2, 3}.
    # Row 2 diffs: {2, 4, 1}.
    # Global diffs: {1, 2, 3, 4}. No duplicates?
    # 1 is in both. So they conflict.
    # So the global set of differences must be unique.
    # This implies 147 unique differences.
    # This implies M >= 147.
    
    # Why does the prompt say 112?
    # Maybe k is not 6? "Difference triangle set (7,5)"
    # The title says (7,5).
    # "Choose 7 rows of 6 increasing integers"
    # If it's (7,5), maybe there are 5 rows?
    # "Choose 7 rows" -> n_rows=7.
    # "of 6 increasing integers" -> k=5? (6 integers means k=5 if 0-indexed? Or k=6 if 1-indexed?)
    # The interface says `construct(n_rows, k)`.
    # "Each row is a list of k + 1 integers".
    # If the title is (7,5), maybe n_rows=7, k=5?
    # If k=5, row_len=6.
    # Pairs per row: 6*5/2 = 15.
    # Total diffs: 7 * 15 = 105.
    # Then M >= 105.
    # This matches the "simple counting bound is 105"!
    # So the parameters are likely n_rows=7, k=5.
    # The prompt text says "Choose 7 rows of 6 increasing integers".
    # 6 integers means k+1=6 => k=5.
    # So I should assume k=5 if the evaluator passes k=5.
    # But the evaluator passes `k`. I must handle the general `k`.
    
    # So, if k=5, row_len=6.
    # Total diffs = 105.
    # M >= 105.
    # Best known 112.
    
    # My CSP approach:
    # 1. Generate Sidon rows of length row_len with max <= M.
    # 2. Find 7 disjoint rows.
    
    # For k=5, row_len=6.
    # Sidon sets of length 6.
    # This is much smaller than length 7.
    
    # Let's implement the binary search on M.
    
    def check_validity(solution):
        """Verify the solution is valid and return max entry."""
        used_diffs = set()
        max_val = 0
        for row in solution:
            if row[0] != 0:
                return False, 0
            if not all(row[i] < row[i+1] for i in range(len(row)-1)):
                return False, 0
            max_val = max(max_val, row[-1])
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False, 0
                    used_diffs.add(d)
        return True, max_val

    # Initial greedy to get a valid solution quickly
    def greedy_solution():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows

    best_sol = greedy_solution()
    valid, best_max = check_validity(best_sol)
    if not valid:
        # Fallback: just return the greedy, even if invalid? No, must be valid.
        # Greedy should be valid by construction.
        pass

    # Binary search for minimal M
    # Lower bound: total_diffs = n_rows * (row_len * (row_len - 1) // 2)
    # M_low = total_diffs
    # M_high = best_max
    
    total_diffs = n_rows * (row_len * (row_len - 1) // 2)
    M_low = total_diffs
    M_high = best_max
    
    # If M_low > M_high, something is wrong with my logic or the problem.
    # But M_high is from a valid solution, so M_high >= M_low must hold if M_low is a valid lower bound.
    # Is M_low a valid lower bound?
    # Yes, we need total_diffs distinct integers in [1, M]. So M >= total_diffs.
    
    # We want to find the smallest M such that a solution exists.
    # We know a solution exists for M_high.
    # We check M values in [M_low, M_high].
    
    # To speed up, we can iterate M from M_low upwards?
    # Or binary search?
    # The search space for M is small (105 to 112).
    # Let's just iterate M from M_low to M_high and return the first valid one.
    # This ensures we find the minimum.
    
    found_solution = None
    
    for M in range(M_low, M_high + 1):
        if time.time() > DEADLINE:
            break
            
        candidates = generate_sidon_rows(M)
        
        if len(candidates) < n_rows:
            continue
            
        # Solve Set Packing
        # To make it faster, sort candidates by max value (smallest first)
        candidates.sort(key=lambda x: x[-1])
        
        # Use a recursive solver with pruning
        # We can also add a heuristic: 
        # If we pick a row, it consumes specific diffs.
        # We want to leave "flexible" diffs for later?
        
        solution = solve_set_packing(candidates, n_rows, set(), [], 0)
        
        if solution is not None:
            # Verify
            valid, m_val = check_validity(solution)
            if valid and m_val <= M:
                found_solution = solution
                # Since we iterate M upwards, this is the minimum M
                break

    if found_solution is not None:
        return found_solution
    else:
        return best_sol

# Test locally? No, just output.

