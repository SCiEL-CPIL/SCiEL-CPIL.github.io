"""Combined program: Iterative Warnsdorff with boolean arrays and local search."""
import time
import random
import sys

N = 9
TIME_LIMIT = 118.0  # Safety margin below 120s evaluator limit
SEED_LENGTH = 104
NUM_VERTICES = 1 << N

# Pre-compute neighbors for all vertices in Q_9
NEIGHBORS = [[] for _ in range(NUM_VERTICES)]
for v in range(NUM_VERTICES):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

# EVOLVE-BLOCK-START
def _warnsdorff_iterative(rng, time_budget, max_backtrack=3):
    """
    Iterative Warnsdorff DFS with limited backtracking using boolean arrays.
    Returns the longest path found.
    """
    best_path = [0]
    best_len = 1
    
    # State: path, used (bool array), exclusion (bool array)
    path = [0]
    used = bytearray(NUM_VERTICES)
    exclusion = bytearray(NUM_VERTICES)
    used[0] = 1
    
    # Stack for backtracking: each entry is (vertex, candidates_list, candidate_index)
    # We use a simpler approach: try up to max_backtrack alternatives at each level
    
    # Iterative DFS with explicit stack
    # Stack elements: (cur_vertex, candidate_list, next_index)
    stack = []
    
    def _score_candidate(v, used_arr, excl_arr):
        """Count valid moves from v."""
        count = 0
        for w in NEIGHBORS[v]:
            if not used_arr[w] and not excl_arr[w]:
                count += 1
        return count
    
    def _get_candidates(cur, used_arr, excl_arr):
        """Get valid candidates from cur, sorted by Warnsdorff score."""
        cands = []
        for v in NEIGHBORS[cur]:
            if not used_arr[v] and not excl_arr[v]:
                score = _score_candidate(v, used_arr, excl_arr)
                cands.append((score, v))
        cands.sort(key=lambda x: x[0])
        return cands
    
    # Main iterative loop
    cur = path[-1]
    backtrack_depth = 0
    
    while True:
        if time.time() > time_budget:
            break
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Get candidates from current vertex
        cands = _get_candidates(cur, used, exclusion)
        
        if not cands:
            # Dead end - try to backtrack
            if backtrack_depth < max_backtrack and len(path) > 1:
                # Undo last move
                last_v = path.pop()
                prev_v = path[-1]
                used[last_v] = 0
                
                # We need to remove last_v's neighbors from exclusion that were added when we moved to last_v
                # This is tricky with boolean arrays. Let's use a simpler approach:
                # Instead of complex backtracking, just restart with different randomization.
                # For now, break out and let the outer loop restart.
                break
            else:
                break
        
        # Pick the best candidate (with some randomness for diversity)
        if len(cands) == 1:
            chosen = cands[0][1]
        else:
            # Pick from top few with random weighting
            top_k = min(3, len(cands))
            # Weight: lower score = higher priority
            weights = [1.0 / (cands[i][0] + 1) for i in range(top_k)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0
            chosen_idx = 0
            for i in range(top_k):
                cumsum += weights[i]
                if r <= cumsum:
                    chosen_idx = i
                    break
            chosen = cands[chosen_idx][1]
        
        # Move to chosen
        old_cur = cur
        path.append(chosen)
        used[chosen] = 1
        
        # Add neighbors of old_cur to exclusion (old_cur becomes non-terminal)
        for nb in NEIGHBORS[old_cur]:
            exclusion[nb] = 1
        
        cur = chosen
    
    return best_path

def _warnsdorff_with_lookahead(rng, time_budget):
    """
    Greedy construction with 2-step lookahead for better move selection.
    """
    path = [0]
    used = bytearray(NUM_VERTICES)
    exclusion = bytearray(NUM_VERTICES)
    used[0] = 1
    
    def _score_move(v, used_arr, excl_arr, cur):
        """
        Score a move to v from cur.
        Lookahead: after moving to v, cur's neighbors become excluded.
        Count how many moves are available from v after that exclusion.
        """
        # Simulate: add v to used, add cur's neighbors to exclusion
        # But we don't want to modify the actual arrays, so we do a virtual check
        score = 0
        for w in NEIGHBORS[v]:
            if used_arr[w] or excl_arr[w]:
                continue
            # w is valid in current state
            # After moving to v, cur's neighbors are added to exclusion
            # Check if w is a neighbor of cur
            is_nb_of_cur = False
            for cnb in NEIGHBORS[cur]:
                if cnb == w:
                    is_nb_of_cur = True
                    break
            if not is_nb_of_cur:
                # w remains valid after the move
                # Count its own extensions (1-level lookahead)
                ext = 0
                for x in NEIGHBORS[w]:
                    if not used_arr[x] and not excl_arr[x]:
                        # Check if x is neighbor of cur (would be excluded)
                        is_nb_of_cur2 = False
                        for cnb in NEIGHBORS[cur]:
                            if cnb == x:
                                is_nb_of_cur2 = True
                                break
                        if not is_nb_of_cur2:
                            ext += 1
                score += 1 + ext * 0.5  # Weight immediate availability + extension
        return score
    
    while time.time() < time_budget:
        cur = path[-1]
        candidates = []
        for v in NEIGHBORS[cur]:
            if not used[v] and not exclusion[v]:
                score = _score_move(v, used, exclusion, cur)
                candidates.append((score, v))
        
        if not candidates:
            break
        
        # Pick the best candidate
        candidates.sort(key=lambda x: -x[0])
        # Add slight randomness
        top_k = min(3, len(candidates))
        if top_k == 1:
            chosen = candidates[0][1]
        else:
            # Weighted random from top
            weights = [1.0 / (i + 1) for i in range(top_k)]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0
            chosen_idx = 0
            for i in range(top_k):
                cumsum += weights[i]
                if r <= cumsum:
                    chosen_idx = i
                    break
            chosen = candidates[chosen_idx][1]
        
        old_cur = cur
        path.append(chosen)
        used[chosen] = 1
        for nb in NEIGHBORS[old_cur]:
            exclusion[nb] = 1
    
    return path

def _try_extend(path, rng, time_budget):
    """
    Try to extend an existing path by finding a valid continuation.
    """
    if not path:
        return path
    
    used = bytearray(NUM_VERTICES)
    exclusion = bytearray(NUM_VERTICES)
    for v in path:
        used[v] = 1
    
    # The last vertex's neighbors are not excluded (it's still terminal)
    # All other vertices' neighbors should be in exclusion
    for i in range(len(path) - 1):
        for nb in NEIGHBORS[path[i]]:
            exclusion[nb] = 1
    
    cur = path[-1]
    new_path = list(path)
    
    while time.time() < time_budget:
        candidates = []
        for v in NEIGHBORS[cur]:
            if not used[v] and not exclusion[v]:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Use Warnsdorff to pick
        scored = []
        for v in candidates:
            score = 0
            for w in NEIGHBORS[v]:
                if not used[w] and not exclusion[w]:
                    score += 1
            scored.append((score, v))
        scored.sort(key=lambda x: x[0])
        
        # Pick with randomness
        top_k = min(2, len(scored))
        if top_k == 1:
            chosen = scored[0][1]
        else:
            r = rng.random()
            if r < 0.7:
                chosen = scored[0][1]
            else:
                chosen = scored[1][1]
        
        old_cur = cur
        new_path.append(chosen)
        used[chosen] = 1
        for nb in NEIGHBORS[old_cur]:
            exclusion[nb] = 1
        cur = chosen
    
    return new_path

def _validate_path(path):
    """Validate that the path is a valid induced path in Q_N."""
    if not path:
        return False
    if len(path) == 1:
        return 0 <= path[0] < NUM_VERTICES
    
    used = set(path)
    if len(used) != len(path):
        return False  # Duplicates
    
    for v in path:
        if v < 0 or v >= NUM_VERTICES:
            return False
    
    # Check consecutive differ by exactly 1 bit
    for i in range(len(path) - 1):
        diff = path[i] ^ path[i + 1]
        if diff == 0 or (diff & (diff - 1)) != 0:
            return False  # Not exactly 1 bit
    
    # Check non-consecutive differ by at least 2 bits
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            diff = path[i] ^ path[j]
            if diff == 0 or (diff & (diff - 1)) == 0:
                return False  # Differ by exactly 1 bit
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    best_path = [0]
    best_len = 1
    
    # Phase 1: Quick initial greedy to get a baseline
    rng_init = random.Random(42)
    p_init = _warnsdorff_with_lookahead(rng_init, start_time + 3.0)
    if len(p_init) > best_len:
        best_len = len(p_init)
        best_path = p_init
    
    # Phase 2: Multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        rng = random.Random(restart_count * 7919 + 12345)
        
        remaining = deadline - time.time()
        
        # Alternate between different strategies
        if restart_count % 3 == 1:
            # Pure Warnsdorff iterative
            run_deadline = min(deadline - 5.0, time.time() + min(10.0, remaining * 0.3))
            p = _warnsdorff_iterative(rng, run_deadline)
        elif restart_count % 3 == 2:
            # Lookahead greedy
            run_deadline = min(deadline - 5.0, time.time() + min(10.0, remaining * 0.3))
            p = _warnsdorff_with_lookahead(rng, run_deadline)
        else:
            # Start from a random vertex
            start_v = rng.randint(0, NUM_VERTICES - 1)
            path = [start_v]
            used = bytearray(NUM_VERTICES)
            exclusion = bytearray(NUM_VERTICES)
            used[start_v] = 1
            run_deadline = min(deadline - 5.0, time.time() + min(10.0, remaining * 0.3))
            
            # Do a quick greedy from random start
            cur = start_v
            while time.time() < run_deadline:
                candidates = []
                for v in NEIGHBORS[cur]:
                    if not used[v] and not exclusion[v]:
                        candidates.append(v)
                if not candidates:
                    break
                scored = []
                for v in candidates:
                    score = 0
                    for w in NEIGHBORS[v]:
                        if not used[w] and not exclusion[w]:
                            score += 1
                    scored.append((score, v))
                scored.sort(key=lambda x: x[0])
                chosen = scored[0][1] if rng.random() < 0.8 else scored[min(1, len(scored)-1)][1]
                old_cur = cur
                path.append(chosen)
                used[chosen] = 1
                for nb in NEIGHBORS[old_cur]:
                    exclusion[nb] = 1
                cur = chosen
            p = path
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 3: Try to extend the best path found
    if time.time() < deadline - 2.0:
        rng_ext = random.Random(99999)
        extended = _try_extend(best_path, rng_ext, deadline - 2.0)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Final validation
    if not _validate_path(best_path):
        # Fallback to a known valid path
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
