"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 9
TOTAL = 1 << N

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a snake in Q_9 using randomized backtracking with incremental block counting.
    This approach is distinct from the greedy lookahead method.
    """
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    best_path = [0]
    best_len = 1
    
    # Precompute neighbors for all vertices
    # neighbors[v] is a list of vertices adjacent to v
    neighbors = [[] for _ in range(TOTAL)]
    for v in range(TOTAL):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    def solve(rng, max_depth=None):
        nonlocal best_path, best_len, start_time
        
        path = [0]
        used = [False] * TOTAL
        used[0] = True
        
        # block_count[u] is the number of vertices in path[:-1] adjacent to u
        # Initially path is [0], so path[:-1] is empty. block_count is all 0.
        block_count = [0] * TOTAL
        
        def backtrack(depth):
            nonlocal best_path, best_len
            
            if time.time() - start_time > time_limit:
                return False # Signal timeout
            
            if depth > best_len:
                best_len = depth
                best_path = list(path)
                
            # Find valid candidates
            cur = path[-1]
            candidates = []
            for u in neighbors[cur]:
                if not used[u] and block_count[u] == 0:
                    candidates.append(u)
                    
            if not candidates:
                return False
                
            # Shuffle candidates for randomization
            rng.shuffle(candidates)
            
            for u in candidates:
                # Try adding u
                used[u] = True
                path.append(u)
                
                # Update block counts
                # u is now the last vertex.
                # The previous last vertex (path[-2]) is now in the "non-last" set.
                # So, for all neighbors w of path[-2], if w != u, increment block_count[w].
                # Wait, the logic is:
                # When we move from v_prev to v_cur, v_prev becomes a "non-last" vertex for the next step.
                # So v_prev blocks its neighbors (except v_cur).
                
                v_prev = path[-2]
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] += 1
                        
                # Recurse
                if backtrack(depth + 1):
                    return True # Timeout or signal to stop? 
                                 # For now, we just want to find best, so we continue searching if time permits.
                                 # But if we find a very long one, we might want to keep going.
                                 # Let's just return True if we want to stop early, but we want max length.
                                 # So we should NOT return True on success, unless we hit a global limit.
                                 # We return False to continue backtracking, unless time is up.
                
                # Backtrack
                path.pop()
                used[u] = False
                for w in neighbors[v_prev]:
                    if w != u:
                        block_count[w] -= 1

            return False # No more candidates or timeout handled inside

        # Run multiple trials with different random seeds
        # Since backtracking is deterministic given the shuffle, we need different shuffles.
        for trial in range(1000):
            if time.time() - start_time > time_limit:
                break
            rng = random.Random(trial)
            # We can also try different start vertices, but symmetry suggests 0 is fine.
            # However, breaking symmetry might help find different structures.
            # For now, start at 0.
            backtrack(1)
            
        return best_path

    return solve(random.Random(0))
# EVOLVE-BLOCK-END
