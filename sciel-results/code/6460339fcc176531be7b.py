"""
Snake-in-the-box in Q_9 using local search, backtracking, and heuristic ordering.
Strategy:
1. Generate a strong base snake using a deterministic greedy with lookahead (minimize future options).
2. Use a priority-based extension (Warnsdorff's rule) to grow the snake.
3. Apply local search: if stuck, try to "unstick" by reversing the last few steps or using a small DFS to find an alternative exit.
4. Exploit symmetry: the hypercube is symmetric under bit permutation and complementation. We can normalize the state to reduce search space or just use it to generate diverse starts.
"""

import time
import random
import heapq

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115  # 5s safety margin
    
    def get_neighbors(v):
        """Yield all neighbors of v in Q_N."""
        for b in range(N):
            yield v ^ (1 << b)

    def is_induced_extension(path, v, used_set):
        """
        Check if v can be appended to path as the next vertex.
        path: list of vertices
        v: candidate next vertex
        used_set: set of used vertices
        """
        if v in used_set:
            return False
        # Consecutive constraint is satisfied by construction (v is neighbor of path[-1])
        # Non-consecutive constraint: v must not be adjacent to any path[i] for i < len(path)-1
        # i.e., dist(v, path[i]) != 1 for all i < len(path)-1
        # dist(v, path[i]) == 1 iff (v ^ path[i]).bit_count() == 1
        # We only need to check path[0:-1] because path[-1] is the parent.
        # Optimization: check against the last few vertices first? No, distance 1 is rare.
        # Actually, the most likely conflicts are with vertices close in the sequence.
        # But we must check all.
        # Note: path[-1] is the current end. v is a neighbor of path[-1].
        # So dist(v, path[-1]) = 1. This is allowed for consecutive.
        # We need dist(v, path[i]) >= 2 for i in 0..len(path)-2.
        
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def warnsdorff_score(v, path, used_set):
        """
        Score a candidate vertex v for Warnsdorff's rule.
        Lower score is better (prefer vertices that leave fewer options for the next step).
        """
        # Count how many valid neighbors v has that are not used and not conflicting with path[:-1]
        # This is expensive, so we approximate or do a quick check.
        # For Q_9, degree is 9. We can just count available neighbors.
        count = 0
        for u in get_neighbors(v):
            if u in used_set:
                continue
            # Check if u would be a valid extension from v in the future?
            # For Warnsdorff, we just look at the degree in the remaining graph.
            # But the "remaining graph" is constrained by the induced path condition.
            # A vertex u is a valid next step from v if is_induced_extension(path + [v], u, used_set | {v}) is true.
            # This is costly. Let's use a simpler heuristic: number of unused neighbors.
            # This tends to force the path into corners or tight areas, which can be good or bad.
            # Actually, for snakes, we want to avoid trapping ourselves.
            # So we prefer vertices with FEWER valid future moves (to commit to a direction) 
            # OR MORE? 
            # In Hamiltonian paths on grids, Warnsdorff says choose the one with fewest onward moves.
            # Let's try that.
            count += 1
        return count

    def greedy_extend_warnsdorff(path, used_set, max_time=5.0):
        """
        Extend path using Warnsdorff-like heuristic.
        """
        start_time = time.time()
        path = list(path)
        used_set = set(used_set)
        
        while True:
            if time.time() - start_time > max_time:
                break
            cur = path[-1]
            candidates = []
            
            for v in get_neighbors(cur):
                if v in used_set:
                    continue
                if is_induced_extension(path, v, used_set):
                    # Calculate heuristic score
                    # Heuristic: number of valid neighbors of v (excluding path conflicts for those neighbors too? No, just unused neighbors)
                    # Let's use a slightly stronger heuristic: 
                    # Score = number of neighbors of v that are unused AND not adjacent to path[:-1]
                    # This is O(N^2) per candidate. N=9, so 81 ops. 9 candidates. ~729 ops per step.
                    # 1000 steps = 729k ops. Very fast.
                    valid_next_count = 0
                    for u in get_neighbors(v):
                        if u in used_set:
                            continue
                        if u == cur:
                            continue
                        # Check if u is valid extension from v
                        # u must not be adjacent to path[:-1]
                        # Note: u is neighbor of v. v is neighbor of cur.
                        # We need to check u against path[:-1].
                        conflict = False
                        for p in path[:-1]:
                            if (u ^ p).bit_count() == 1:
                                conflict = True
                                break
                        if not conflict:
                            valid_next_count += 1
                    
                    candidates.append((valid_next_count, v))
            
            if not candidates:
                break
            
            # Sort by score (ascending: fewest options first)
            candidates.sort()
            next_v = candidates[0][1]
            path.append(next_v)
            used_set.add(next_v)
            
        return path, used_set

    def local_search_extend(path, used_set, max_iterations=1000, max_time=5.0):
        """
        Try to extend the path by looking ahead and backtracking if stuck.
        """
        start_time = time.time()
        path = list(path)
        used_set = set(used_set)
        
        # We will use a simple DFS with a depth limit for the "stuck" part.
        # Actually, let's just try random extensions with restarts.
        # If Warnsdorff gets stuck, we try to "wiggle" the tail.
        
        # First, try Warnsdorff
        path, used_set = greedy_extend_warnsdorff(path, used_set, max_time=2.0)
        
        # Now try to extend further with randomization
        rng = random.Random(123)
        for _ in range(max_iterations):
            if time.time() - start_time > max_time:
                break
            
            cur = path[-1]
            candidates = []
            for v in get_neighbors(cur):
                if v in used_set:
                    continue
                if is_induced_extension(path, v, used_set):
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Pick randomly
            v = rng.choice(candidates)
            path.append(v)
            used_set.add(v)
            
        return path, used_set

    best_path = []
    best_len = 0
    
    # Generate diverse starting points
    # Symmetry: we can fix the first vertex to 0 without loss of generality for the length,
    # but the shape matters. Let's use random starts.
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(i * 1000 + 42)
        start = rng.randint(0, MAX_VERTS - 1)
        path = [start]
        used_set = {start}
        
        # Phase 1: Greedy Warnsdorff
        path, used_set = greedy_extend_warnsdorff(path, used_set, max_time=1.0)
        
        # Phase 2: Local Search / Random Extension
        path, used_set = local_search_extend(path, used_set, max_iterations=500, max_time=2.0)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
        # Print progress for debugging (optional, remove in final if noisy)
        # print(f"Start {i}: len={len(path)}")

    # Phase 3: Intensive improvement on the best path
    if best_path and time.time() < deadline - 5:
        # Try to extend the best path further with more aggressive search
        # We can try to reverse the path and extend from the other end?
        # The snake is a path, so extending from the head is equivalent to extending from the tail if we reverse.
        # But the "used" set is the same.
        
        # Let's try a few more random extensions on the best path
        rng = random.Random(999)
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            
            trial_path, trial_used = list(best_path), set(best_path)
            trial_path, trial_used = local_search_extend(trial_path, trial_used, max_iterations=1000, max_time=1.0)
            
            if len(trial_path) > best_len:
                best_len = len(trial_path)
                best_path = list(trial_path)

    # Fallback if nothing found
    if not best_path:
        path = [0]
        used_set = {0}
        while True:
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used_set:
                    continue
                if is_induced_extension(path, v, used_set):
                    path.append(v)
                    used_set.add(v)
                    found = True
                    break
            if not found:
                break
        best_path = path

    return best_path
# EVOLVE-BLOCK-END
