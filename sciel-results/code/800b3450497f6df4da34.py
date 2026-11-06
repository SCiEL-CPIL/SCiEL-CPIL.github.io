import random
import time

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors as bitmasks
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

# Precompute list of neighbors for fast iteration
NEIGHBOR_LIST = [[] for _ in range(SIZE)]
for v in range(SIZE):
    NEIGHBOR_LIST[v] = [v ^ (1 << b) for b in range(N)]

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(20260715)
    
    best_path = None
    best_len = 0

    def validate_and_score(path):
        if not path:
            return 0
        n = len(path)
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return -1
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1
            if i >= 2:
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def backtracking_search(start_v, target_len, timeout):
        """
        Build path backward from start_v.
        We maintain:
        - path: list of vertices in order (end to start)
        - used_mask: bitmask of used vertices
        - last_v: the current end of the path (we extend from here)
        """
        path = [start_v]
        used_mask = 1 << start_v
        last_v = start_v
        found = False
        
        # Recursive DFS with backtracking
        # To avoid recursion depth issues, we use a stack-based approach 
        # or rely on Python's recursion limit (set high enough for ~200 depth)
        # However, explicit stack is safer and faster for this specific "first success" logic.
        
        # State: (path, used_mask, last_v)
        # We want to find ANY path of length target_len.
        
        def dfs(path, used_mask, last_v):
            nonlocal found
            if time.time() > timeout:
                found = True
                return
            
            if len(path) == target_len:
                found = True
                return
            
            cur = last_v
            candidates = []
            for v in NEIGHBOR_LIST[cur]:
                if used_mask & (1 << v):
                    continue
                # Safety check 1: v must not have any other used neighbors (except cur)
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                
                # Safety check 2: After using v, can the rest of the graph still support a path?
                # Specifically, v must have at least 2 free neighbors (one to connect to next, one to keep options open? 
                # Actually, for a simple path extension, v just needs at least 1 free neighbor to continue, 
                # UNLESS it's the final vertex. But we don't know if it's final.
                # A stronger heuristic: The number of free neighbors of v (excluding cur) must be >= 1 
                # if we plan to extend further. 
                # But a better "safety" for induced paths: 
                # If we pick v, all other neighbors of v become "blocked" from being used later 
                # because they would be distance 2 from cur (via v) but distance 1 from v.
                # Wait, the constraint is: non-consecutive vertices differ by >= 2 bits.
                # If we add v, any neighbor w of v (w != cur) cannot be added later 
                # because dist(w, v) = 1 and they would be non-consecutive (unless w is added immediately after v, 
                # but then w is consecutive to v). 
                # So if we add v, all neighbors of v (except cur) become permanently unavailable 
                # UNLESS they are added immediately after v.
                # This means: if we pick v, we MUST pick one of v's other free neighbors next, 
                # OR v must be the last vertex.
                # Since we don't know if v is last, we should check: 
                # Does v have at least 1 free neighbor (other than cur)?
                # If not, v can only be the last vertex.
                
                free_neighbors_count = (NEIGHBOR_MASK[v] & ~used_mask & ~(1 << cur)).bit_count()
                
                # Heuristic: If free_neighbors_count == 0, v can only be the end of the path.
                # If we still need more vertices, this is a dead end.
                if free_neighbors_count == 0 and len(path) < target_len - 1:
                    continue
                
                # Another safety: If we pick v, the set of vertices that are neighbors to v (and not cur) 
                # are "poisoned" for future use unless they are the immediate next step.
                # This makes the search very constrained.
                
                candidates.append((v, free_neighbors_count))
            
            # Sort candidates by heuristic: prefer vertices with more free neighbors? 
            # Or fewer? Usually in snake-in-box, you want to preserve options.
            # But here, picking a vertex "poisons" its neighbors. 
            # So picking a vertex with FEWER free neighbors might be safer? 
            # Actually, if a vertex has many free neighbors, picking it poisons many vertices.
            # So we might prefer vertices with FEW free neighbors to minimize poisoning?
            # Let's try sorting by free_neighbors_count ascending.
            candidates.sort(key=lambda x: x[1])
            
            for v, _ in candidates:
                if found:
                    return
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                dfs(new_path, new_used, v)
                if found:
                    return
                # Backtrack
                # No need to explicitly remove from used_mask as we pass copies
        
        dfs(path, used_mask, last_v)
        
        if found and len(path) == target_len:
            return list(reversed(path))
        return None

    # Phase 1: Greedy baseline (fast)
    def greedy(priority):
        path = [0]
        used = {0}
        used_mask = 1
        for _ in range(300):
            if time.time() > deadline:
                break
            cur = path[-1]
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
            used_mask |= (1 << nxt)
        return path

    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 2: Backtracking search with increasing target length
    # Start from a length slightly above the best found
    # We fix the start vertex to 0 for simplicity, or try a few starts
    
    # To make backtracking effective, we need a good heuristic for ordering.
    # The "safety count" (free neighbors) is key.
    
    # We will try to find a path of length best_len + 1, then +2, etc.
    # But backtracking can be expensive. We limit time per attempt.
    
    # Try different start vertices? 0 is symmetric, so 0 is fine.
    # But maybe starting from a "dense" area helps? 0 is fine.
    
    current_target = best_len + 1
    max_target = 250  # Upper bound guess
    
    while current_target <= max_target:
        if time.time() > deadline - 5:
            break
        
        # Give backtracking a time budget
        # If we find a path quickly, great. If not, move on.
        # We try a few random restarts of the backtracking logic with different heuristics?
        # For now, just one attempt with a tight time limit.
        
        attempt_deadline = time.time() + 2.0 # 2 seconds per length increment attempt
        path = backtracking_search(0, current_target, attempt_deadline)
        
        if path:
            # Validate
            vL = validate_and_score(path)
            if vL == current_target - 1:
                best_len = current_target - 1
                best_path = path
                current_target += 1
                # Reset time budget for next increment?
                # Keep going
            else:
                # Invalid? Shouldn't happen if backtracking is correct, but check.
                break
        else:
            # Failed to find path of this length
            # Maybe try a different start vertex? Or different heuristic?
            # For now, if we fail at length L, we assume L is the max for this start/heuristic.
            # But it might just be a dead end. 
            # Let's try a few more random starts or heuristics before giving up on this length.
            
            success = False
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Try a random start vertex
                start_v = rnd.randint(0, SIZE - 1)
                attempt_deadline = time.time() + 1.0
                path = backtracking_search(start_v, current_target, attempt_deadline)
                if path:
                    vL = validate_and_score(path)
                    if vL == current_target - 1:
                        best_len = current_target - 1
                        best_path = path
                        current_target += 1
                        success = True
                        break
            
            if not success:
                # Give up on this length? Or increment anyway?
                # If we can't find length L, we likely can't find L+1 either.
                break

    # Phase 3: If time remains, try to extend the best path found using local search/greedy
    # Or just run more greedy restarts if backtracking didn't improve much.
    if best_len < 150 and time.time() < deadline - 10:
        while time.time() < deadline - 5:
            priority = list(range(N))
            rnd.shuffle(priority)
            p = greedy(priority)
            L = validate_and_score(p)
            if L > best_len:
                best_len = L
                best_path = p

    if best_path is None:
        return [0]
    return best_path
