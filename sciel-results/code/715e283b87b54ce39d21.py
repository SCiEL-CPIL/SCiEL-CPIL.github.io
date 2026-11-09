"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from collections import deque

N = 9
TIME_LIMIT = 115  # seconds, leave buffer before 120s

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a long induced path (snake) in Q_9 using efficient constraint propagation.
    
    We model the problem as finding a path in the hypercube such that no two
    non-consecutive vertices are adjacent. 
    We maintain a 'blocked' set of vertices that are forbidden from being the 
    next vertex in the path because they are adjacent to a non-consecutive 
    vertex in the current path.
    
    Specifically, if current path is P = [v_0, ..., v_k], the next vertex v_{k+1}
    must be:
    1. A neighbor of v_k.
    2. Not in P (already used).
    3. Not adjacent to any v_i where i <= k-2.
    
    We maintain a set `blocked` which is the union of neighbors of {v_0, ..., v_{k-2}}.
    When we extend by v_{k+1}, the new blocked set for the next step (v_{k+2})
    will be `blocked` U neighbors(v_{k-1}).
    """
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute neighbor sets as frozensets or lists for quick intersection/union?
    # Lists are fine, we just need to add to a set.
    
    def backtrack(path, used, blocked, start_time):
        """
        Recursive backtracking search.
        path: list of vertices in current path
        used: set of vertices in path
        blocked: set of vertices that are adjacent to path[:-2] (forbidden for next step)
                 Note: path[-1] is the current head. path[-2] is the one before.
                 The next vertex must not be in blocked.
                 The next vertex must not be in used.
                 The next vertex must be a neighbor of path[-1].
        """
        nonlocal best_path, best_len
        
        if time.time() - start_time > TIME_LIMIT - 2:
            return
        
        cur = path[-1]
        
        # Find valid candidates
        # Candidates are neighbors of cur that are not in used and not in blocked.
        candidates = []
        for nb in neighbors[cur]:
            if nb in used:
                continue
            if nb in blocked:
                continue
            candidates.append(nb)
        
        if not candidates:
            # Dead end
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            return
        
        # Heuristic: Choose candidate that leaves the most options
        # Score: number of neighbors of candidate that are not in used and not in (blocked U neighbors(path[-2]))
        # The new blocked set will be blocked U neighbors(path[-2])
        # Let's precompute new_blocked_base = blocked U neighbors(path[-2]) if len(path) >= 2
        # Actually, the constraint for v_{k+2} is that it cannot be adjacent to v_{k-1} (which is path[-2] if path is 0-indexed and cur is path[-1]).
        # Wait, if path is [v0, v1, vk], cur=vk.
        # Next is vk+1.
        # vk+1 cannot be adjacent to v0...vk-2.
        # Current `blocked` contains neighbors of v0...vk-2.
        # So vk+1 cannot be in `blocked`.
        # After adding vk+1, the new path is [v0...vk, vk+1].
        # Next step vk+2 cannot be adjacent to v0...vk-1.
        # So new_blocked = blocked U neighbors(vk-1).
        # vk-1 is path[-2] if len(path) > 1.
        
        new_blocked_base = set(blocked)
        if len(path) >= 2:
            prev_prev = path[-2]
            new_blocked_base.update(neighbors[prev_prev])
        
        # Score candidates
        def score(v):
            # Count available neighbors for the next step
            # Next step candidates for v will be neighbors[v] not in (used U {v}) and not in (new_blocked_base U neighbors(v))
            # Wait, if we pick v, the new head is v.
            # The next-next vertex will be constrained by neighbors of v?
            # No, the constraint for the vertex *after* v (let's call it w) is:
            # w is neighbor of v.
            # w cannot be adjacent to path[:-2] of the new path.
            # New path = path + [v].
            # New path[:-2] = path[:-1] = [v0...vk-1].
            # So w cannot be adjacent to v0...vk-1.
            # This is exactly `new_blocked_base`.
            # So w must not be in `new_blocked_base`.
            # Also w must not be in `used` U {v}.
            
            avail = 0
            for nb in neighbors[v]:
                if nb in used or nb == v:
                    continue
                if nb in new_blocked_base:
                    continue
                avail += 1
            return avail
        
        # Sort candidates by score
        candidates.sort(key=score, reverse=True)
        
        # Try top candidates
        # To add some randomness and avoid getting stuck in local optima, 
        # we can shuffle the top few or use a random seed.
        # However, for a deterministic improvement, we stick to greedy first.
        # If we have time, we can do randomized restarts.
        
        for v in candidates:
            if time.time() - start_time > TIME_LIMIT - 2:
                return
            
            new_path = path + [v]
            new_used = used | {v}
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = new_path[:]
            
            # Recurse
            backtrack(new_path, new_used, new_blocked_base, start_time)
            
            # If we found a very long path, we might want to break early?
            # No, we want the longest.
            
            # Pruning: If the best possible length from this state cannot exceed best_len, return.
            # Max possible extension is roughly (2^N - len(path)) but that's a loose bound.
            # A better bound: The number of vertices not in used and not in blocked.
            # But we are constrained to be a path.
            # Let's just rely on time limit and heuristic ordering.

    # Strategy 1: Start from 0
    start_time = time.time()
    initial_path = [0]
    initial_used = {0}
    initial_blocked = set() # No vertices to block yet, as path has length 1. 
    # Wait, if path is [v0], next is v1. v1 cannot be adjacent to v_{-1}? No.
    # Condition: non-consecutive vertices differ in >= 2 bits.
    # For path [v0, v1], v0 and v1 are consecutive. No constraint on distance (except they must be adjacent).
    # For path [v0, v1, v2], v2 cannot be adjacent to v0.
    # So for the first step (adding v1 to [v0]), there are no "blocked" vertices from previous non-consecutive ones.
    # So initial_blocked is empty.
    
    backtrack(initial_path, initial_used, initial_blocked, start_time)
    
    # Strategy 2: Randomized restarts with different starting vertices
    # The greedy choice might get stuck. We can try a few random starts.
    # But the backtracking search above is exhaustive for the first start (0).
    # It might take too long to explore the whole space.
    # We should limit the depth or use a heuristic to prune more aggressively.
    # The current backtracking explores ALL valid extensions.
    # In Q9, the branching factor is small (max 9), but the depth is ~100-200.
    # 9^100 is huge. The constraints prune it significantly.
    # However, if it's still too slow, we might need to cut off the search.
    # Let's add a depth limit or time check inside the loop more aggressively.
    # The `if time.time() ...` check is at the start of the function.
    
    # If the first run didn't find a long enough snake, try a few random starts.
    # But we must ensure we don't exceed time.
    if time.time() - start_time < TIME_LIMIT - 10:
        random.seed(42)
        for _ in range(5):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            start_v = random.randint(0, (1 << N) - 1)
            p = [start_v]
            u = {start_v}
            b = set()
            backtrack(p, u, b, start_time)
            
    return best_path
# EVOLVE-BLOCK-END
