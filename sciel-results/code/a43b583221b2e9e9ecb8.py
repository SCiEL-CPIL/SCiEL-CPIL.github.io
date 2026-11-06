import time
import random

N = 9
TIME_LIMIT = 115.0  # Safety margin before 120s

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Use a set for O(1) lookup of used vertices
    # We will perform a backtracking search
    
    def is_valid_extension(path, used, v_new):
        # v_new must not be in used
        if v_new in used:
            return False
        # v_new must not be distance 1 from any v_i where i < len(path) - 1
        # The last vertex is path[-1], which is the parent.
        # We check against path[0] to path[-2]
        for i in range(len(path) - 1):
            if (v_new ^ path[i]).bit_count() == 1:
                return False
        return True

    def get_candidates(path, used):
        cur = path[-1]
        candidates = []
        for b in range(N):
            v_new = cur ^ (1 << b)
            if is_valid_extension(path, used, v_new):
                candidates.append((b, v_new))
        return candidates

    def heuristic_score(path, used, v_new):
        # Count how many valid neighbors v_new has
        # A neighbor v_next of v_new is valid if it's not used and not dist 1 from path[0..-2]
        # Note: v_new itself is not in used yet when we score it.
        # We check neighbors of v_new.
        score = 0
        for b in range(N):
            v_next = v_new ^ (1 << b)
            if v_next in used:
                continue
            # Check if v_next is distance 1 from any path[i] for i < len(path)-1
            # If so, it would be an invalid extension *after* adding v_new? 
            # Actually, the condition is: non-consecutive vertices in the *final* path must be dist >= 2.
            # If we add v_new, then v_next (if added later) would be at dist 2 from path[-1] (which is v_new's parent? No, v_new is parent of v_next).
            # Wait. If path is ... v_{k-1}, v_k. We add v_{k+1}.
            # v_{k+1} must be dist >= 2 from v_0 ... v_{k-1}.
            # If we consider v_next as a potential v_{k+2}, it must be dist >= 2 from v_0 ... v_{k-1}.
            # It is dist 1 from v_{k+1} (consecutive, OK).
            # So we just need to check if v_next is dist 1 from any v_i in path[0..k-1].
            valid = True
            for i in range(len(path) - 1):
                if (v_next ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                score += 1
        return score

    # Backtracking search
    path = [0]
    used = {0}
    max_depth_reached = 1
    
    # We use a stack for backtracking
    # State: (path, used, backtrack_depth_allowed)
    # To keep it simple and fast, we'll do a iterative deepening or just a greedy-with-backtrack loop.
    
    # Strategy: 
    # 1. Try to extend greedily.
    # 2. If stuck, backtrack 1 step, try next candidate.
    # 3. If all candidates at a level fail, backtrack further.
    # 4. If we reach a dead end at the root, restart with a random initial bit? 
    #    Or just let it run and try different random seeds for tie-breaking.
    
    # Let's implement a recursive backtracking with a time check.
    # However, recursion depth can be up to 512 (vertices), which is fine for Python if we increase limit or use iterative.
    # But 512 is small. The branching factor is small (<=9).
    # The main cost is the `is_valid_extension` check which is O(L) where L is path length.
    # L can be ~200. 200 * 9 * 200 (backtrack steps?) might be too slow if we do full search.
    # We need to limit the search.
    
    # Better approach: Randomized Greedy with local repair?
    # Or, a greedy search that prioritizes "promising" moves.
    
    # Let's try a greedy algorithm with a lookahead heuristic, and if it gets stuck, 
    # we backtrack a few steps (e.g., 5) and try a different move.
    
    def greedy_with_backtrack():
        nonlocal best_path, best_len
        path = [0]
        used = {0}
        
        # Stack of (current_path, current_used, next_candidate_index, backtrack_limit)
        # Actually, simpler: 
        # We maintain the path.
        # At each step, we look at candidates.
        # We sort candidates by heuristic score (descending).
        # We pick the best.
        # If we get stuck, we pop the last vertex, mark it as "blocked" for this branch? 
        # No, standard backtracking:
        
        # Let's use a stack of states.
        # State: (path_tuple, used_frozenset, candidates_list, current_index)
        # This is memory heavy.
        
        # Alternative: Iterative Deepening DFS? No, depth is not fixed.
        
        # Let's stick to a simple backtracking that limits the number of backtracks per extension.
        
        def search(path, used):
            nonlocal best_path, best_len
            if time.time() - start_time > TIME_LIMIT:
                return False # Stop
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
            cur = path[-1]
            candidates = []
            for b in range(N):
                v_new = cur ^ (1 << b)
                if is_valid_extension(path, used, v_new):
                    candidates.append((b, v_new))
                    
            if not candidates:
                return False # Dead end, backtrack
            
            # Sort by heuristic
            # To add randomness to escape local optima, we can shuffle or pick randomly from top k
            # Let's sort by score, but if scores are equal, randomize order.
            scored_candidates = []
            for b, v in candidates:
                score = heuristic_score(path, used, v)
                scored_candidates.append((score, random.random(), b, v))
            
            # Sort by score desc, then random
            scored_candidates.sort(key=lambda x: (-x[0], x[1]))
            
            # Try the best candidate
            # To be more robust, we could try a few, but backtracking handles the "try next" logic.
            # We just return True if we successfully extended.
            # The caller (backtrack logic) will handle popping if this leads to a dead end later.
            
            # Actually, a simple DFS that explores the best branch first is good.
            # But if the best branch leads to a dead end, we need to try the second best.
            
            # So we need to iterate through candidates.
            for i, (score, _, b, v) in enumerate(scored_candidates):
                path.append(v)
                used.add(v)
                
                if search(path, used):
                    return True # Found a longer path or finished search (shouldn't happen unless we hit limit)
                
                # Backtrack
                path.pop()
                used.remove(v)
                
                # Optional: If we've tried many, and time is running out, stop?
                # The time check at the top handles this.
                
            return False # All candidates failed to extend further (dead end)
            
        search(path, used)
        return best_path

    # The above recursive search might be too slow if it explores too much.
    # The heuristic `heuristic_score` is O(L * N). L ~ 200, N=9. ~1800 ops per candidate.
    # Candidates ~ 9. ~16000 ops per step.
    # If we do 200 steps, that's 3.2M ops just for scoring.
    # Backtracking adds overhead.
    # Python is slow. 3.2M ops is ~0.1s. 
    # If we backtrack a lot, it could be slow.
    
    # Let's optimize `is_valid_extension` and `heuristic_score`.
    # We can maintain a set of "blocked" vertices?
    # A vertex is blocked for extension if it is in `used` OR it is distance 1 from any `path[i]` for `i < len(path)-1`.
    # Let's maintain `blocked` set.
    # When we add `v_new` to path:
    #   1. Add `v_new` to `used`.
    #   2. The new `v_new` blocks its neighbors? No.
    #   3. The condition is: `v_candidate` is valid if `v_candidate` not in `used` AND `v_candidate` not in `dist1_from_path_excluding_last`.
    #   Let `S` be the set of vertices at distance 1 from any `v_i` in `path[0..len(path)-2]`.
    #   Then valid candidates for `path[-1]` are neighbors of `path[-1]` that are NOT in `used` AND NOT in `S`.
    
    # Let's re-implement with this optimization.
    
    def optimized_search():
        nonlocal best_path, best_len
        path = [0]
        used = {0}
        # S: set of vertices at dist 1 from path[0..-2]
        # Initially path=[0], path[0..-2] is empty. S = empty.
        S = set()
        
        def add_vertex(v):
            nonlocal S
            path.append(v)
            used.add(v)
            # Update S: 
            # The new vertex v is at dist 1 from path[-2] (if exists).
            # But S is defined as dist 1 from path[0..-2].
            # When we append v, the new "excluding last" set is path[0..-2] (which was path[0..-1] before).
            # So we need to add neighbors of the *new* second-to-last vertex?
            # No.
            # Let's trace:
            # Path: [v0, v1]. Last is v1. Excluding last: [v0]. S = neighbors(v0).
            # Add v2. Path: [v0, v1, v2]. Last is v2. Excluding last: [v0, v1].
            # New S should be neighbors(v0) U neighbors(v1).
            # Old S was neighbors(v0).
            # So we add neighbors(v1) to S.
            # Generally, when adding v_new to path, we add neighbors(path[-2]) to S.
            
            if len(path) >= 2:
                prev = path[-2]
                for b in range(N):
                    nb = prev ^ (1 << b)
                    S.add(nb)
            
        def remove_vertex(v):
            nonlocal S
            # v is path[-1]. We are popping it.
            # The new "excluding last" set is path[0..-3].
            # The old "excluding last" set was path[0..-2].
            # So we need to remove neighbors(path[-2]) from S?
            # Wait.
            # Before pop: Path [v0, ..., vk]. Last vk. Excl last: v0..v(k-1). S = N(v0)..N(v(k-1)).
            # After pop: Path [v0, ..., v(k-1)]. Last v(k-1). Excl last: v0..v(k-2). S' = N(v0)..N(v(k-2)).
            # So we must remove N(v(k-1)) from S.
            # v(k-1) is the new last element after pop, which is path[-2] BEFORE pop.
            
            if len(path) >= 2:
                prev = path[-2]
                for b in range(N):
                    nb = prev ^ (1 << b)
                    S.discard(nb)
            
            path.pop()
            used.remove(v)
            
        def get_valid_candidates(cur):
            candidates = []
            for b in range(N):
                v_new = cur ^ (1 << b)
                if v_new in used or v_new in S:
                    continue
                candidates.append(v_new)
            return candidates
            
        def heuristic_score(v_new):
            # Count valid neighbors of v_new
            # A neighbor v_next of v_new is valid if v_next not in used and v_next not in S (current S).
            # Note: S does not include neighbors of v_new yet, because v_new is not in "excl last" set.
            score = 0
            for b in range(N):
                v_next = v_new ^ (1 << b)
                if v_next in used or v_next in S:
                    continue
                score += 1
            return score

        def search(path, used, S):
            nonlocal best_path, best_len
            if time.time() - start_time > TIME_LIMIT:
                return False
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
                
            cur = path[-1]
            candidates = get_valid_candidates(cur)
            
            if not candidates:
                return False
                
            # Score and sort
            scored = []
            for v in candidates:
                s = heuristic_score(v)
                scored.append((s, random.random(), v))
            
            scored.sort(key=lambda x: (-x[0], x[1]))
            
            for _, _, v in scored:
                add_vertex(v)
                
                if search(path, used, S):
                    return True
                
                remove_vertex(v)
                
            return False
            
        search(path, used, S)
        return best_path

    # Run the search
    # We might want to run multiple independent searches with different random seeds or initial states?
    # The randomization in sorting handles some of that.
    # Let's just run the optimized search.
    
    res = optimized_search()
    
    # If the search didn't find anything better than the seed (unlikely), return seed?
    # The seed is 104 edges. Our search should find at least that.
    # If res is empty or short, fall back to simple greedy?
    # But optimized_search starts with 0 and should work.
    
    # One potential issue: if the random seed leads to a bad local optimum quickly and then backtracking is too slow to escape.
    # The time limit guards against infinite loops.
    
    return res

# Keep the original function name
construct = construct
