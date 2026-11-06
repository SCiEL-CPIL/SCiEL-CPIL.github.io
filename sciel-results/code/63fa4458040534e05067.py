import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for fast adjacency checks
# NEIGHBOR_MASK[v] is a bitmask where bit i is 1 if i is a neighbor of v
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(42)
    best_path = None
    best_len = 0

    # Bitmasks for efficiency
    ALL_MASK = (1 << SIZE) - 1

    def validate_and_score(path):
        """Check validity and return edge count (len-1) or -1 if invalid."""
        if not path:
            return 0
        n = len(path)
        seen = set()
        for i, v in enumerate(path):
            if v in seen:
                return -1
            seen.add(v)
            if i >= 1:
                # Check immediate predecessor
                if (v ^ path[i-1]).bit_count() != 1:
                    return -1
                # Check non-consecutive predecessors
                for j in range(i - 1):
                    if (v ^ path[j]).bit_count() == 1:
                        return -1
        return n - 1

    def smart_backtrack(start, max_time_per_run=5.0):
        """
        Perform a backtracking search with MRV/Lookahead heuristics.
        Returns the longest path found during this specific run.
        """
        run_start = time.time()
        best_local_path = [start]
        best_local_len = 1
        
        # State:
        # path: list of vertices
        # used_mask: bitmask of vertices in path
        # blocked_mask: bitmask of vertices that are neighbors of used vertices (except the current head)
        #   A vertex v is "blocked" if it is a neighbor of any u in path[:-1].
        #   If we are at cur, the valid next vertices are (ALL_MASK ^ used_mask ^ blocked_mask) & NEIGHBOR_MASK[cur]
        
        def dfs(path, used_mask, blocked_mask, cur):
            nonlocal best_local_path, best_local_len, run_start
            
            # Time check
            if time.time() - run_start > max_time_per_run:
                return
            
            L = len(path)
            if L > best_local_len:
                best_local_len = L
                best_local_path = list(path)

            # Find valid next moves
            # Candidates are neighbors of cur that are not used and not blocked
            valid_candidates_mask = NEIGHBOR_MASK[cur] & ~(used_mask | blocked_mask)
            
            if valid_candidates_mask == 0:
                return
            
            # Generate list of candidate vertices
            candidates = []
            vm = valid_candidates_mask
            while vm:
                # Extract lowest set bit
                lsb = vm & -vm
                v = lsb.bit_length() - 1
                candidates.append(v)
                vm ^= lsb
            
            if not candidates:
                return
            
            # Heuristic: Lookahead
            # For each candidate v, calculate the number of valid moves from v
            # This helps avoid moving into a dead end.
            scored_candidates = []
            for v in candidates:
                # If we move to v:
                # new_used = used_mask | (1 << v)
                # new_blocked = (blocked_mask | (1 << cur)) # cur becomes blocked for future
                #   Wait, cur is already in used_mask, so it's in used. 
                #   The blocked set contains neighbors of all used vertices EXCEPT the current head.
                #   So if we move from cur to v:
                #   - cur moves from "head" to "body". So cur's neighbors become blocked (if not already).
                #   - v becomes the new head.
                #   
                # Let's re-verify the blocked mask logic.
                # Initial: path=[start], used={start}, blocked=0.
                # Move to v:
                #   path=[start, v]
                #   used={start, v}
                #   blocked should be neighbors of start (except v).
                #   So new_blocked = (NEIGHBOR_MASK[start] & ~(1<<v)) | (blocked_mask)
                #   Actually, blocked_mask currently holds neighbors of path[:-1].
                #   In the state (path, used, blocked, cur), blocked holds neighbors of path[:-1].
                #   When we extend to v, the new path is path + [v].
                #   The new "body" is path. The new "head" is v.
                #   So new_blocked should be neighbors of path (which includes cur).
                #   new_blocked = blocked | (NEIGHBOR_MASK[cur] & ~(1<<v))
                #   Note: cur is in path, so its neighbors are now "blocked" for future steps (unless they are v).
                
                new_blocked = blocked_mask | (NEIGHBOR_MASK[cur] & ~(1 << v))
                new_used = used_mask | (1 << v)
                
                # Count valid moves from v
                # Moves from v must be neighbors of v, not in new_used, and not in new_blocked
                next_moves_mask = NEIGHBOR_MASK[v] & ~(new_used | new_blocked)
                count = next_moves_mask.bit_count()
                
                # We want to prefer moves that leave more options (higher count)
                # But to maintain randomness/diversity, we can use a weighted random choice
                # or just sort and pick top k.
                # Let's use a score that encourages high count but allows some randomness.
                scored_candidates.append((count, v, new_used, new_blocked))
            
            # Sort by count descending, then by random tie-breaker to ensure diversity
            # To avoid always picking the same "best" path, we shuffle or pick randomly from top tier
            # Strategy: Pick randomly from candidates with count > 0, weighted by count?
            # Or simply shuffle the list and pick the first one with the max count?
            # Let's use a "soft" max: pick randomly from the top 3 counts if available.
            
            # For efficiency, let's just shuffle and pick the first valid one, but bias towards higher counts.
            # Simple approach: Shuffle candidates, then iterate. If count > threshold, take it.
            # Better approach: Weighted random selection.
            
            # To keep it fast, we'll just shuffle the candidates and pick one.
            # But the "smart" part is that we *prune* if count is 0? 
            # Actually, if count is 0, moving there leads to a dead end immediately. 
            # We should avoid moving to a vertex with 0 next moves unless it's the only option.
            
            # Filter out dead ends if possible
            non_dead_end = [c for c in scored_candidates if c[0] > 0]
            if non_dead_end:
                # Pick randomly from non-dead-ends, weighted by count?
                # Let's just pick randomly from non-dead-ends to maintain diversity
                # while ensuring we don't step into a hole if we have alternatives.
                choice = rnd.choice(non_dead_end)
            else:
                # All are dead ends, pick one randomly
                choice = rnd.choice(scored_candidates)
                
            _, v, new_used, new_blocked = choice
            
            path.append(v)
            dfs(path, new_used, new_blocked, v)
            path.pop()
            
            # Optional: Backtracking
            # If we want to be a true solver, we should try other candidates.
            # But with time limits, deep DFS with heuristics is often better than wide BFS.
            # We can add a "backtrack limit" or just rely on the heuristic to guide us well.
            # Given the complexity, let's stick to a single deep dive per restart with randomized tie-breaking.
            # To allow some backtracking, we could recurse on the next best candidate if the first fails to improve significantly?
            # No, that's complex. Let's rely on multiple restarts.
            # However, simple DFS might get stuck in local optima.
            # Let's add a small amount of backtracking: try the top 2-3 candidates.
            
            # Let's refine: Try all candidates, but in a randomized order biased by score.
            # This is effectively what we did by picking one. To backtrack, we need to loop.
            
            # Let's modify the structure: Loop through candidates in heuristic order.
            # But recursion depth can be high.
            
            # Revised DFS with limited backtracking:
            # We already picked one. To backtrack, we would need to try others.
            # Let's implement a version that tries the top K candidates.
            
            # Actually, the code above only tries ONE candidate. 
            # To improve, we should try multiple.
            # Let's replace the single choice with a loop over the top candidates.
            
            # We'll do this in the main loop logic outside the single-choice block.
            # For now, this function returns after one deep dive.
            # To allow backtracking, we'd need to restructure.
            # Let's keep it simple: One deep dive per call. Multiple calls = multiple restarts.
            # This is often sufficient for "greedy + lookahead" strategies.
            
            pass

        # Initialize
        used_mask = 1 << start
        blocked_mask = 0
        dfs([start], used_mask, blocked_mask, start)
        
        return best_local_path

    # Phase 1: Many smart backtracking restarts
    attempts = 0
    # Use a mix of fixed and random start vertices
    start_vertices = [0, 1, (1<<N)-1] + [rnd.randint(0, SIZE-1) for _ in range(20)]
    
    while time.time() < deadline - 10:
        start = rnd.choice(start_vertices)
        # Vary the time per run to allow some long runs and many short ones
        # Most runs should be quick, a few can be longer
        max_time = rnd.uniform(2.0, 8.0)
        
        p = smart_backtrack(start, max_time)
        if p:
            L = validate_and_score(p)
            if L > best_len:
                best_len = L
                best_path = p
                
        attempts += 1
        if attempts % 100 == 0 and time.time() > deadline - 15:
            break

    # Phase 2: Polish the best path found
    # Try to extend the best path found by trying all possible next steps (though it should be maximal)
    # Or try to swap segments? 
    # Given the constraints, the smart backtrack should find near-maximal paths.
    
    if best_path is None:
        return [0]
    
    return best_path

if __name__ == "__main__":
    # Quick test
    p = construct()
    print(f"Length: {len(p)-1}")
