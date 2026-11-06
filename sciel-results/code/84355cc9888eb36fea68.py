"""Constraint-based backtracking solver for snake-in-the-box in Q_9."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# Precompute neighbors for each vertex
_NEIGHBORS = [None] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _NEIGHBORS[_v] = [_v ^ (1 << b) for b in range(N)]

# Precompute adjacency masks: _ADJ_MASK[v] has bit i set iff v and i are adjacent
_ADJ_MASK = [0] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _m = 0
    for _nb in _NEIGHBORS[_v]:
        _m |= (1 << _nb)
    _ADJ_MASK[_v] = _m


def _greedy_seed(bit_order=None):
    """Generate a greedy snake. If bit_order is None, use natural order."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [0]
    used = {0}
    used_mask = 1
    while True:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # v must not be adjacent to any used vertex except cur
            conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
            if conflict:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        used_mask |= (1 << nxt)
    return path


def _dfs_solve(path, used_mask, deadline, max_depth, node_budget):
    """
    Depth-first search with forward checking and MRV heuristic.
    Returns (best_path_found, nodes_visited).
    """
    best_path = list(path)
    nodes_visited = [0]
    
    def _dfs(current_path, current_used_mask):
        nodes_visited[0] += 1
        if nodes_visited[0] > node_budget:
            return
        if time.time() > deadline:
            return
        if len(current_path) >= max_depth:
            if len(current_path) > len(best_path):
                best_path[:] = current_path[:]
            return
        
        cur = current_path[-1]
        
        # Find all valid candidates (unvisited neighbors not adjacent to other path vertices)
        candidates = []
        for v in _NEIGHBORS[cur]:
            if current_used_mask & (1 << v):
                continue
            conflict = _ADJ_MASK[v] & (current_used_mask ^ (1 << v) ^ (1 << cur))
            if conflict:
                continue
            candidates.append(v)
        
        if not candidates:
            # Dead end: record if better
            if len(current_path) > len(best_path):
                best_path[:] = current_path[:]
            return
        
        # MRV: sort by number of future options (ascending)
        def future_options(v):
            new_mask = current_used_mask | (1 << v)
            cnt = 0
            for nv in _NEIGHBORS[v]:
                if new_mask & (1 << nv):
                    continue
                conflict = _ADJ_MASK[nv] & (new_mask ^ (1 << nv) ^ (1 << v))
                if not conflict:
                    cnt += 1
            return cnt
        
        candidates.sort(key=future_options)
        
        for v in candidates:
            new_path = current_path + [v]
            new_mask = current_used_mask | (1 << v)
            _dfs(new_path, new_mask)
            if time.time() > deadline or nodes_visited[0] > node_budget:
                return
    
    _dfs(path, used_mask)
    return best_path, nodes_visited[0]


def _iterative_deepening_dfs(start_path, start_used_mask, deadline, initial_depth, max_extra=30, node_budget_per_iter=50000):
    """
    Iterative deepening: try to extend beyond current length by 1, 2, 3, ...
    Returns best path found.
    """
    best_path = list(start_path)
    current_depth = len(start_path)
    
    for extra in range(1, max_extra + 1):
        target_depth = current_depth + extra
        remaining_time = deadline - time.time()
        if remaining_time < 0.5:
            break
        
        # Scale node budget based on remaining time
        scale = min(1.0, remaining_time / 30.0)
        budget = int(node_budget_per_iter * scale)
        if budget < 5000:
            budget = 5000
        
        improved, _ = _dfs_solve(best_path, 0, deadline, target_depth, budget)
        # Recompute used mask for improved path
        improved_mask = 0
        for v in improved:
            improved_mask |= (1 << v)
        
        if len(improved) > len(best_path):
            best_path = improved
            current_depth = len(best_path)
            # Reset for next iteration with new start
            start_used_mask = improved_mask
        else:
            # No improvement at this depth; try a few more with slightly larger budget
            # but if we've tried 3 extras without success, stop
            if extra >= 3:
                break
    
    return best_path


def _randomized_dfs(start_path, start_used_mask, deadline, target_len, node_budget):
    """
    DFS with randomized candidate ordering for diversity.
    """
    best_path = list(start_path)
    nodes_visited = [0]
    
    def _dfs(current_path, current_used_mask):
        nodes_visited[0] += 1
        if nodes_visited[0] > node_budget:
            return
        if time.time() > deadline:
            return
        if len(current_path) >= target_len:
            best_path[:] = current_path[:]
            return
        
        cur = current_path[-1]
        
        candidates = []
        for v in _NEIGHBORS[cur]:
            if current_used_mask & (1 << v):
                continue
            conflict = _ADJ_MASK[v] & (current_used_mask ^ (1 << v) ^ (1 << cur))
            if conflict:
                continue
            candidates.append(v)
        
        if not candidates:
            if len(current_path) > len(best_path):
                best_path[:] = current_path[:]
            return
        
        # Shuffle for diversity
        random.shuffle(candidates)
        
        for v in candidates:
            new_path = current_path + [v]
            new_mask = current_used_mask | (1 << v)
            _dfs(new_path, new_mask)
            if time.time() > deadline or nodes_visited[0] > node_budget:
                return
    
    _dfs(start_path, start_used_mask)
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Phase 1: Generate greedy seeds with various bit orders, keep best
    best_path = None
    best_len = 0
    best_mask = 0
    
    for i in range(20):
        if time.time() > deadline - 90.0:
            break
        if i == 0:
            path = _greedy_seed()
        else:
            bo = list(range(N))
            random.shuffle(bo)
            path = _greedy_seed(bo)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
            best_mask = 0
            for v in path:
                best_mask |= (1 << v)
    
    # Phase 2: Iterative deepening DFS to extend the best path
    if time.time() < deadline - 5.0:
        improved = _iterative_deepening_dfs(best_path, best_mask, deadline, 
                                             initial_depth=best_len, max_extra=25,
                                             node_budget_per_iter=40000)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
            best_mask = 0
            for v in best_path:
                best_mask |= (1 << v)
    
    # Phase 3: Multiple randomized DFS attempts from truncated prefixes
    remaining = deadline - time.time()
    if remaining > 10.0:
        num_attempts = max(1, int(remaining / 15.0))
        for attempt in range(num_attempts):
            if time.time() > deadline - 2.0:
                break
            
            # Truncate at a random position and try to re-extend
            if best_len <= 2:
                break
            pos = random.randint(max(1, best_len // 3), best_len - 1)
            sub_path = best_path[:pos + 1]
            sub_mask = 0
            for v in sub_path:
                sub_mask |= (1 << v)
            
            target = best_len + random.randint(0, 5)
            budget = int(min(remaining / max(1, num_attempts - attempt), 30.0) * 20000)
            budget = max(budget, 10000)
            
            improved = _randomized_dfs(sub_path, sub_mask, deadline, target, budget)
            if len(improved) > best_len:
                best_path = improved
                best_len = len(improved)
                best_mask = 0
                for v in best_path:
                    best_mask |= (1 << v)
    
    # Phase 4: Final greedy extension from the best path
    if time.time() < deadline - 1.0:
        cur = best_path[-1]
        used_mask = best_mask
        while True:
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
                if conflict:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            best_path.append(nxt)
            used_mask |= (1 << nxt)
            best_mask = used_mask
            cur = nxt
            best_len = len(best_path)
    
    return best_path
# EVOLVE-BLOCK-END
