import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    points = list(range(v))
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    t_set = set(t_subsets)
    num_t = len(t_subsets)
    
    # Map t-subset to index for faster access if needed, but set of tuples is okay for lookup
    # Actually, for the "uncovered" set, we store the tuples themselves.
    
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = set(t_subsets)
        solution = []
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            
            # Pick a random uncovered t-subset
            # Using random.choice on a set is O(1) in CPython? 
            # Actually, random.choice requires a sequence. 
            # We can convert to list, but that's O(N). 
            # Better: keep a list of uncovered and remove from it.
            # But removal from list is O(N).
            # Let's use a list and a "alive" mask or just swap-remove.
            # Given N_t ~ 11k, converting to list each time is 11k * 400 steps = 4.4M ops. Acceptable.
            # Or, we can just pick a random element from the set by converting to list once? No, it changes.
            
            # Optimization: Keep uncovered as a list.
            # When we cover some, we filter the list.
            # Filtering is O(N).
            
            # Let's try the list approach.
            # Actually, the current code has `uncovered` as a set. 
            # `random.choice` doesn't work on sets.
            # We need to pick one.
            # We can maintain a list `uncovered_list`.
            
            # Let's restructure to use a list for uncovered.
            pass
            
        return solution

    # The above sketch has issues with random selection from set.
    # Let's implement a robust version.
    
    def run_greedy_v2(seed_val):
        random.seed(seed_val)
        uncovered_list = list(t_subsets)
        solution = []
        
        while uncovered_list:
            if elapsed() > deadline - 10:
                break
                
            # Pick random index
            idx = random.randint(0, len(uncovered_list) - 1)
            target_ts = uncovered_list[idx]
            
            # Find all blocks containing target_ts
            # target_ts is a tuple of 5 points.
            # Points in target_ts: S
            # Points not in target_ts: R
            S = set(target_ts)
            R = [p for p in points if p not in S]
            
            best_block = None
            best_gain = -1
            best_block_tsets = []
            
            # Iterate 3-combinations of R
            for comb3 in itertools.combinations(R, 3):
                # Block is S union comb3
                # Create tuple of sorted block
                block_tuple = tuple(sorted(S.union(comb3)))
                
                # Calculate gain: how many t-subsets in this block are in uncovered_list?
                # We can't check "in uncovered_list" efficiently if it's a list (O(N)).
                # We should maintain a set `uncovered_set` alongside `uncovered_list`.
                # Or just use the set for membership and list for iteration/removal.
                
                # Let's maintain `uncovered_set`.
                # Gain = sum(1 for ts in itertools.combinations(block_tuple, t) if ts in uncovered_set)
                
                gain = 0
                block_tsets = []
                for ts in itertools.combinations(block_tuple, t):
                    if ts in uncovered_set:
                        gain += 1
                        block_tsets.append(ts)
                    
                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tuple
                    best_block_tsets = block_tsets
            
            if best_block is None or best_gain == 0:
                break
                
            solution.append(best_block)
            
            # Remove covered t-subsets from uncovered_list and uncovered_set
            for ts in best_block_tsets:
                uncovered_set.discard(ts)
                
            # Rebuild uncovered_list from uncovered_set
            # This is O(N)
            uncovered_list = list(uncovered_set)
            
        return solution

    # We need to define uncovered_set in the scope of run_greedy_v2
    # Let's refine the structure.
    
    def greedy_search(seed_val):
        random.seed(seed_val)
        uncovered_set = set(t_subsets)
        solution = []
        
        while uncovered_set:
            if elapsed() > deadline - 10:
                break
                
            # Pick a random uncovered t-subset
            # Convert to list to pick random? O(N).
            # Alternative: Pick random from list, if it's not in set, retry? 
            # No, list must be synced.
            # Let's just use `random.choice(list(uncovered_set))`.
            # N=11628. Steps ~ 300. 300 * 11628 ~ 3.5M ops. OK.
            
            target_ts = random.choice(list(uncovered_set))
            
            S = set(target_ts)
            R = [p for p in points if p not in S]
            
            best_block = None
            best_gain = -1
            best_block_tsets = []
            
            for comb3 in itertools.combinations(R, 3):
                block_tuple = tuple(sorted(S.union(comb3)))
                
                gain = 0
                block_tsets = []
                for ts in itertools.combinations(block_tuple, t):
                    if ts in uncovered_set:
                        gain += 1
                        block_tsets.append(ts)
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tuple
                    best_block_tsets = block_tsets
                    
            if best_block is None:
                break
                
            solution.append(best_block)
            
            for ts in best_block_tsets:
                uncovered_set.discard(ts)
                
        return solution

    def local_search(initial_solution):
        solution = list(initial_solution)
        if not solution:
            return []
            
        # Precompute t-subsets for each block in solution
        # block -> list of t-subsets
        block_tsets_map = {}
        for b in solution:
            block_tsets_map[b] = list(itertools.combinations(b, t))
            
        # Count coverage
        cover_count = defaultdict(int)
        for b in solution:
            for ts in block_tsets_map[b]:
                cover_count[ts] += 1
                
        improved = True
        local_start = time.time()
        
        while improved and (time.time() - local_start) < 20.0:
            improved = False
            
            # Phase 1: Drop redundant
            i = 0
            while i < len(solution):
                b = solution[i]
                redundant = True
                for ts in block_tsets_map[b]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in block_tsets_map[b]:
                        cover_count[ts] -= 1
                    solution.pop(i)
                    del block_tsets_map[b]
                    improved = True
                    # Do not increment i, check same index again
                else:
                    i += 1
            if (time.time() - local_start) > 20.0:
                break
                
            # Phase 2: Swaps
            # Try to swap each block with a better one
            # To make it fast, we limit the number of swap attempts per iteration
            random.shuffle(solution)
            for i, b_out in enumerate(solution):
                if (time.time() - local_start) > 20.0:
                    break
                    
                unique_tsets = [ts for ts in block_tsets_map[b_out] if cover_count[ts] == 1]
                if not unique_tsets:
                    # Should have been dropped, but if not, drop it now?
                    # If no unique tsets, it's redundant.
                    for ts in block_tsets_map[b_out]:
                        cover_count[ts] -= 1
                    solution.pop(i)
                    del block_tsets_map[b_out]
                    improved = True
                    break
                    
                # Find candidates covering all unique_tsets
                # Pick one unique tset to generate candidates
                u_ts = unique_tsets[0]
                S = set(u_ts)
                R = [p for p in points if p not in S]
                
                candidates = []
                for comb3 in itertools.combinations(R, 3):
                    b_in = tuple(sorted(S.union(comb3)))
                    if b_in in block_tsets_map:
                        continue
                    # Check if b_in covers all unique_tsets
                    covers_all = True
                    b_in_set = set(itertools.combinations(b_in, t))
                    for u in unique_tsets:
                        if u not in b_in_set:
                            covers_all = False
                            break
                    if covers_all:
                        candidates.append(b_in)
                        
                if not candidates:
                    continue
                    
                # Evaluate candidates: maximize overlap with existing covered tsets (minimize new unique load? or just random?)
                # Simple heuristic: pick one that maximizes number of tsets already covered by >=2 blocks
                best_cand = None
                best_score = -1
                for b_in in candidates:
                    score = 0
                    for ts in itertools.combinations(b_in, t):
                        if cover_count[ts] >= 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = b_in
                        
                if best_cand:
                    # Perform swap
                    for ts in block_tsets_map[b_out]:
                        cover_count[ts] -= 1
                    del block_tsets_map[b_out]
                    
                    b_in_tsets = list(itertools.combinations(best_cand, t))
                    block_tsets_map[best_cand] = b_in_tsets
                    for ts in b_in_tsets:
                        cover_count[ts] += 1
                        
                    solution[i] = best_cand
                    improved = True
                    break # Restart loop? Or continue? Restarting is safer.
                    # Actually, breaking the for loop and restarting the while is good.
                    
        return solution

    best_solution = None
    best_size = float('inf')
    
    num_restarts = 20
    for i in range(num_restarts):
        if elapsed() > deadline - 30:
            break
            
        sol = greedy_search(i)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            
        if elapsed() > deadline - 20:
            break
            
        improved_sol = local_search(sol)
        if len(improved_sol) < best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol
            
    if not best_solution:
        # Fallback: single greedy run without local search time limit?
        # Or just return empty? No, must be valid.
        # Greedy should work.
        best_solution = greedy_search(0)
        
    return [list(b) for b in best_solution]
