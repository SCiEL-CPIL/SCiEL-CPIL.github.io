import time
import random
import sys

sys.setrecursionlimit(100000)

DEADLINE = 0
TIME_LIMIT = 115  # Leave 5s buffer for safety

def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Get a baseline solution using greedy to ensure we always return something
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    # Determine lower bound
    # Total differences needed = n_rows * (k*(k+1)//2)
    # Max difference is M. Available differences are 1..M.
    # So M * 1 >= n_rows * k * (k+1) / 2 is not quite right, differences can be reused? No, distinct.
    # Each row has k*(k+1)/2 differences.
    # All these differences must be distinct across all rows.
    # Total distinct differences needed = n_rows * k * (k+1) / 2.
    # The largest difference is at most M.
    # So we need at least n_rows * k * (k+1) / 2 distinct integers in {1, ..., M}.
    # Thus M >= n_rows * k * (k+1) / 2.
    generic_lb = n_rows * k * (k + 1) // 2
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    best_solution = baseline_rows
    best_scope = max(max(row) for row in baseline_rows)
    
    # 2. Iterative Deepening on M
    # We try to find a solution with max entry <= M
    # Start from lower bound, go up to baseline_scope
    
    # If baseline is already good, maybe skip small M if time is tight?
    # No, we want the best.
    
    # We will try M from lower_bound to best_scope
    # To avoid spending too much time on very small M where solution might not exist,
    # we can add a small time budget per M.
    
    for M in range(lower_bound, best_scope + 1):
        if time.time() > DEADLINE:
            break
        
        # Allocate time budget for this M
        # Earlier M's are more likely to be the optimal answer, so give them more time?
        # Or give equal time?
        # Let's give a fixed small budget, e.g., 1-2 seconds per M, scaled by remaining time.
        remaining_time = DEADLINE - time.time()
        if remaining_time < 1.0:
            break
        
        # If we are near the end, use all remaining time for the last M?
        # No, if we find a solution, we return immediately.
        # So we just need enough time to *check* if a solution exists for M.
        # If it exists, we return. If not, we move to M+1.
        
        # Heuristic: Give 2 seconds per M, but cap at remaining time.
        time_budget = min(2.0, remaining_time * 0.9)
        
        sol = solve_m_optimized(n_rows, k, M, time_budget)
        if sol is not None:
            # Found a valid solution with max <= M
            # Since we iterate M upwards, the first one found is the minimum scope?
            # Yes, because we start from lower_bound.
            return sol

    # If we didn't find anything better, return baseline
    return best_solution

def greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def generate_candidates(M, k, time_limit=10.0):
    """
    Generate all valid rows of length k+1 starting at 0, with max element <= M.
    Returns a list of tuples: (mask, row_tuple)
    mask is a bitmask of the differences used by the row.
    """
    start = time.time()
    timeout = time_limit
    candidates = []
    
    # Pre-compute bits for differences 1..M
    # Note: difference 0 is not used in mask for differences, but row elements are 0..M
    # Differences are in 1..M
    bit_of = [0] * (M + 1)
    for d in range(1, M + 1):
        bit_of[d] = 1 << d  # Shift by d. 1<<1 is bit 1 (value 2), 1<<0 is bit 0.
                             # We want distinct differences.
                             # If diff is d, set bit d.
                             # Python integers handle arbitrary size, so 1<<100 is fine.
    
    def backtrack(current_row, current_mask, last_val):
        if time.time() - start > timeout:
            return False
        if len(current_row) == k + 1:
            candidates.append((current_mask, tuple(current_row)))
            return True
            
        for x in range(last_val + 1, M + 1):
            new_mask = 0
            valid = True
            for a in current_row:
                d = x - a
                # d is at least 1 because x > a (strictly increasing)
                bit = bit_of[d]
                if current_mask & bit:
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                current_row.append(x)
                if not backtrack(current_row, current_mask | new_mask, x):
                    return False
                current_row.pop()
        return True

    # Start with [0]
    # Mask for [0] is 0 (no differences yet)
    backtrack([0], 0, 0)
    return candidates

def solve_m_optimized(n_rows, k, M, time_budget):
    """
    Try to find n_rows disjoint rows from the set of all valid rows with max <= M.
    Returns a list of rows if found, else None.
    """
    start_time = time.time()
    
    # 1. Generate all candidate rows
    # We might run out of time during generation
    # Set a tight deadline for generation to ensure we have time for search
    gen_deadline = start_time + time_budget * 0.5 
    candidates = generate_candidates(M, k, time_limit=gen_deadline - start_time)
    
    if time.time() > start_time + time_budget:
        return None
        
    if len(candidates) < n_rows:
        return None
        
    # 2. Prepare for search
    # Sort candidates to improve pruning.
    # Heuristic: Rows with fewer differences (sparser masks) are "cheaper" and might allow more others?
    # Or rows with smaller max value? All are <= M.
    # Let's sort by mask bit count (number of differences used).
    # Actually, all rows of length k+1 have exactly k*(k+1)/2 differences.
    # So bit count is constant for all candidates of the same k.
    # So sorting by bit count is useless.
    
    # Alternative heuristic:
    # We want to find a set of disjoint rows.
    # This is equivalent to finding a clique in the "compatibility graph" of size n_rows.
    # Or independent set in the "conflict graph".
    
    # Given n_rows is small (e.g., 7), we can use backtracking.
    # To make it faster, we can randomize the order of candidates in each DFS run.
    # And do multiple restarts.
    
    # Pre-compute conflicts?
    # With potentially thousands of candidates, building full conflict matrix is O(N^2).
    # If N=1000, 1M checks. Fast in Python? Maybe ~0.1s.
    # But we can check on the fly.
    
    # Let's use a randomized backtracking search.
    # State: list of selected row indices, used_mask
    # Step: pick a candidate not conflicting with used_mask, add to used_mask, recurse.
    
    # To avoid getting stuck in bad local minima (though this is exact search, not optimization),
    # the "bad" part is that we might explore a huge subtree of a "bad" choice before failing.
    # Randomizing the order of trying candidates helps to find *a* solution faster if one exists,
    # or prove non-existence faster if we use good heuristics?
    # Actually, if a solution exists, random order often finds it quickly.
    
    search_deadline = start_time + time_budget
    rng = random.Random(int(start_time * 1000)) # Seed based on time for reproducibility in restarts?
    
    # We will do multiple restarts of the DFS with different orderings
    # But generating candidates is the bottleneck.
    # Let's just shuffle the candidate list once and run DFS.
    # If it fails, we can try again with a different shuffle?
    # Or just run one deterministic pass with a good heuristic, then random restarts.
    
    # Heuristic for DFS:
    # At each step, we have a list of available candidates.
    # We should pick the one that "leaves the most options" for the next step?
    # That's expensive to compute.
    # Simpler: Just iterate through shuffled list.
    
    # Optimization:
    # If we fail to find a solution, it might be because we picked a "bad" row early.
    # We can use a "failure-directed" random restart?
    
    # Let's try a simple approach:
    # 1. Shuffle candidates.
    # 2. DFS.
    # If DFS takes too long, it will hit the deadline and return False.
    # Then we try another shuffle.
    
    def dfs(selected_indices, used_mask, available_candidates):
        if time.time() > search_deadline:
            return None
        
        if len(selected_indices) == n_rows:
            return selected_indices
            
        # Heuristic: Pick the next candidate
        # To speed up, we can filter available_candidates to only those disjoint from used_mask
        # But doing this filtering at every recursion level is expensive if we create new lists.
        # Instead, we iterate and check.
        
        # To improve pruning, we can sort available candidates by "conflict count" with *other* available candidates?
        # Too slow.
        
        # Let's just iterate.
        for i, cand in enumerate(available_candidates):
            cand_mask, cand_row = cand
            if cand_mask & used_mask:
                continue
                
            # Try this candidate
            selected_indices.append(cand_row)
            new_available = available_candidates # We don't remove used ones explicitly in list, just check mask
            
            # Optimization: If we can, pass a filtered list?
            # No, just check mask.
            
            result = dfs(selected_indices, used_mask | cand_mask, new_available)
            if result is not None:
                return result
            selected_indices.pop()
            
            # Time check
            if time.time() > search_deadline:
                return None

    # Run multiple randomized restarts
    max_restarts = 5 # Fixed number, or scale with time?
    # Let's calculate allowed restarts based on time budget
    # If budget is 2s, and one DFS takes 0.5s, we can do 4 restarts.
    # We don't know how long one DFS takes.
    # Let's just do 1 restart with a fixed shuffle, and if it fails, we move to next M?
    # No, if a solution exists, it should be found with high probability by random search if the space is not too constrained.
    # However, if the space is very tight (near lower bound), random search might fail.
    # In that case, a more systematic search is needed.
    
    # Let's try a deterministic pass first:
    # Sort candidates by "max value" (all same M? No, max <= M).
    # Actually, all candidates have max <= M.
    # Let's try to pick rows that are "spread out"?
    
    # Simple deterministic:
    # Just pick the first disjoint one.
    # This is greedy. It might fail.
    
    # Let's do 3 random restarts.
    shuffled_cands = candidates[:]
    
    for restart in range(3):
        if time.time() > search_deadline:
            break
        rng.shuffle(shuffled_cands)
        result = dfs([], 0, shuffled_cands)
        if result is not None:
            return result
            
    return None

# Entry point
def construct(n_rows, k):
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + TIME_LIMIT

    # 1. Get a baseline solution using greedy
    baseline_rows = greedy_construct(n_rows, k)
    if not baseline_rows:
        return []
    
    # Determine lower bound
    generic_lb = n_rows * k * (k + 1) // 2
    if n_rows == 7 and k == 5:
        lower_bound = 105
    else:
        lower_bound = generic_lb

    best_solution = baseline_rows
    best_scope = max(max(row) for row in baseline_rows)
    
    # 2. Iterative Deepening on M
    for M in range(lower_bound, best_scope + 1):
        if time.time() > DEADLINE:
            break
        
        remaining_time = DEADLINE - time.time()
        if remaining_time < 1.0:
            break
        
        # Give more time to smaller M (closer to optimal)
        # But we need to ensure we can generate candidates and search.
        # Let's cap the time per M to avoid getting stuck on a bad M.
        time_budget = min(3.0, remaining_time * 0.8)
        
        sol = solve_m_optimized(n_rows, k, M, time_budget)
        if sol is not None:
            return sol

    return best_solution
