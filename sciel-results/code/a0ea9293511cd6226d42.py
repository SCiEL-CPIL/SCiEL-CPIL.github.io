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
def _csp_forward_checking(rng, time_budget, max_depth=200):
    """
    CSP-inspired DFS with forward checking and constraint counting.
    Uses a smarter heuristic that considers the impact of each move on
    the remaining constraint satisfaction problem.
    """
    best_path = [0]
    best_len = 1
    
    path = [0]
    used = bytearray(NUM_VERTICES)
    blocked = bytearray(NUM_VERTICES)  # Vertices that cannot be used (neighbors of non-terminal vertices)
    used[0] = 1
    
    def _count_available(v, used_arr, blocked_arr):
        """Count valid neighbors of v that are not used or blocked."""
        count = 0
        for w in NEIGHBORS[v]:
            if not used_arr[w] and not blocked_arr[w]:
                count += 1
        return count
    
    def _forward_check_score(v, cur, used_arr, blocked_arr):
        """
        Score a candidate v given current vertex cur.
        After moving to v, cur's neighbors become blocked.
        We estimate the impact on the constraint problem.
        """
        # Count available moves from v after the move
        # cur's neighbors will be blocked, so we need to account for that
        available_after = 0
        for w in NEIGHBORS[v]:
            if used_arr[w]:
                continue
            # Check if w would be blocked (i.e., is a neighbor of cur)
            would_be_blocked = False
            for cnb in NEIGHBORS[cur]:
                if cnb == w:
                    would_be_blocked = True
                    break
            if not would_be_blocked and not blocked_arr[w]:
                available_after += 1
        
        # Also consider: how many vertices become blocked that were previously available?
        # This is a proxy for constraint propagation
        new_blocks = 0
        for nb in NEIGHBORS[cur]:
            if not used_arr[nb] and not blocked_arr[nb]:
                new_blocks += 1
        
        # Prefer moves that leave more options and block fewer vertices
        score = available_after * 2.0 - new_blocks * 0.5
        return score
    
    cur = path[-1]
    steps = 0
    max_steps = 50000  # Hard limit to avoid infinite loops
    
    while steps < max_steps:
        steps += 1
        if time.time() > time_budget:
            break
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # Get candidates
        candidates = []
        for v in NEIGHBORS[cur]:
            if not used[v] and not blocked[v]:
                score = _forward_check_score(v, cur, used, blocked)
                candidates.append((score, v))
        
        if not candidates:
            break
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: -x[0])
        
        # Pick with randomness for diversity
        if len(candidates) == 1:
            chosen = candidates[0][1]
        else:
            top_k = min(4, len(candidates))
            # Exponential weighting favoring top candidates
            weights = [1.0 / (i + 1) ** 1.5 for i in range(top_k)]
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
        
        # Move to chosen
        old_cur = cur
        path.append(chosen)
        used[chosen] = 1
        
        # Block neighbors of old_cur (it becomes non-terminal)
        for nb in NEIGHBORS[old_cur]:
            blocked[nb] = 1
        
        cur = chosen
    
    return best_path

def _bidirectional_splice(path1, path2, rng, time_budget):
    """
    Try to splice two partial paths together.
    Look for a valid connection between the end of path1 and start of path2,
    or between any two non-adjacent vertices.
    """
    if not path1 or not path2:
        return path1 if path1 else path2
    
    # Build used and blocked sets for path1
    used = bytearray(NUM_VERTICES)
    blocked = bytearray(NUM_VERTICES)
    for v in path1:
        used[v] = 1
    for i in range(len(path1) - 1):
        for nb in NEIGHBORS[path1[i]]:
            blocked[nb] = 1
    
    # Try to extend path1 with vertices from path2
    # Check if we can connect path1[-1] to some vertex in path2
    best_splice = None
    best_splice_len = len(path1)
    
    end1 = path1[-1]
    
    # Try each vertex in path2 as potential continuation
    for idx, v in enumerate(path2):
        if used[v] or blocked[v]:
            continue
        # Check if v can be added to path1
        # v must be adjacent to end1
        if v not in NEIGHBORS[end1]:
            continue
        
        # Simulate adding v
        # Check if the rest of path2 can be appended
        valid = True
        sim_used = bytearray(used)
        sim_blocked = bytearray(blocked)
        sim_used[v] = 1
        for nb in NEIGHBORS[end1]:
            sim_blocked[nb] = 1
        
        # Try to append path2[idx+1:]
        appended = 0
        cur = v
        for w in path2[idx+1:]:
            if sim_used[w] or sim_blocked[w]:
                valid = False
                break
            # Check adjacency
            if w not in NEIGHBORS[cur]:
                valid = False
                break
            sim_used[w] = 1
            for nb in NEIGHBORS[cur]:
                sim_blocked[nb] = 1
            cur = w
            appended += 1
        
        if valid and appended > 0:
            total_len = len(path1) + 1 + appended
            if total_len > best_splice_len:
                best_splice_len = total_len
                # Build the spliced path
                spliced = list(path1) + [v] + path2[idx+1:idx+1+appended]
                best_splice = spliced
    
    if best_splice:
        return best_splice
    return path1

def _local_search_improve(path, rng, time_budget):
    """
    Try to improve a path by local search: swap or insert vertices.
    """
    if not path or len(path) < 3:
        return path
    
    best_path = list(path)
    best_len = len(path)
    
    # Build state
    used = bytearray(NUM_VERTICES)
    blocked = bytearray(NUM_VERTICES)
    for v in path:
        used[v] = 1
    for i in range(len(path) - 1):
        for nb in NEIGHBORS[path[i]]:
            blocked[nb] = 1
    
    # Try to extend from the end
    cur = path[-1]
    attempts = 0
    max_attempts = 10000
    
    while attempts < max_attempts:
        attempts += 1
        if time.time() > time_budget:
            break
        
        # Try to extend
        candidates = []
        for v in NEIGHBORS[cur]:
            if not used[v] and not blocked[v]:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Try each candidate with some probability
        chosen = rng.choice(candidates)
        old_cur = cur
        
        # Test if this extension is valid
        test_used = bytearray(used)
        test_blocked = bytearray(blocked)
        test_used[chosen] = 1
        for nb in NEIGHBORS[old_cur]:
            test_blocked[nb] = 1
        
        # Check if we can continue further
        next_candidates = []
        for w in NEIGHBORS[chosen]:
            if not test_used[w] and not test_blocked[w]:
                next_candidates.append(w)
        
        if next_candidates or rng.random() < 0.1:
            # Accept this move
            used[chosen] = 1
            for nb in NEIGHBORS[old_cur]:
                blocked[nb] = 1
            cur = chosen
            best_path.append(chosen)
            best_len = len(best_path)
    
    return best_path

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
    
    # Phase 1: CSP forward checking to get a good initial solution
    rng_init = random.Random(42)
    p_init = _csp_forward_checking(rng_init, start_time + 5.0)
    if len(p_init) > best_len:
        best_len = len(p_init)
        best_path = p_init
    
    # Phase 2: Multiple restarts with CSP and splicing
    restart_count = 0
    partial_paths = []
    
    while time.time() < deadline - 15.0:
        restart_count += 1
        rng = random.Random(restart_count * 7919 + 12345)
        
        remaining = deadline - time.time()
        
        if restart_count % 4 == 1:
            # CSP forward checking from vertex 0
            run_deadline = min(deadline - 10.0, time.time() + min(15.0, remaining * 0.4))
            p = _csp_forward_checking(rng, run_deadline)
        elif restart_count % 4 == 2:
            # CSP from random start
            start_v = rng.randint(0, NUM_VERTICES - 1)
            path = [start_v]
            used = bytearray(NUM_VERTICES)
            blocked = bytearray(NUM_VERTICES)
            used[start_v] = 1
            run_deadline = min(deadline - 10.0, time.time() + min(15.0, remaining * 0.4))
            
            # Quick CSP-style greedy
            cur = start_v
            while time.time() < run_deadline:
                candidates = []
                for v in NEIGHBORS[cur]:
                    if not used[v] and not blocked[v]:
                        # Score based on forward checking
                        avail = 0
                        for w in NEIGHBORS[v]:
                            if not used[w] and not blocked[w]:
                                # Check if w would be blocked by cur
                                would_block = False
                                for cnb in NEIGHBORS[cur]:
                                    if cnb == w:
                                        would_block = True
                                        break
                                if not would_block:
                                    avail += 1
                        candidates.append((avail, v))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: -x[0])
                top_k = min(3, len(candidates))
                weights = [1.0 / (i + 1) ** 1.2 for i in range(top_k)]
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
                    blocked[nb] = 1
                cur = chosen
            p = path
        elif restart_count % 4 == 3:
            # Generate a partial path for splicing
            run_deadline = min(deadline - 10.0, time.time() + min(8.0, remaining * 0.3))
            p = _csp_forward_checking(rng, run_deadline)
            if len(p) > 5:
                partial_paths.append(p)
                if len(partial_paths) > 20:
                    partial_paths.pop(0)
        else:
            # Try splicing
            if len(partial_paths) >= 2:
                p1 = partial_paths[rng.randint(0, len(partial_paths) - 1)]
                p2 = partial_paths[rng.randint(0, len(partial_paths) - 1)]
                run_deadline = min(deadline - 10.0, time.time() + 5.0)
                p = _bidirectional_splice(p1, p2, rng, run_deadline)
            else:
                # Fallback to CSP
                run_deadline = min(deadline - 10.0, time.time() + min(10.0, remaining * 0.3))
                p = _csp_forward_checking(rng, run_deadline)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 3: Local search improvement on the best path
    if time.time() < deadline - 5.0:
        rng_ext = random.Random(99999)
        extended = _local_search_improve(best_path, rng_ext, deadline - 5.0)
        if len(extended) > best_len:
            best_len = len(extended)
            best_path = extended
    
    # Final validation
    if not _validate_path(best_path):
        # Fallback to a known valid path
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
