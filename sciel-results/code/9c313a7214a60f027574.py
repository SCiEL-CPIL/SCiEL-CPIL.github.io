import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0
    
    def get_valid_moves(used, end_vertex):
        """
        Returns list of valid candidates to extend from end_vertex.
        A candidate v is valid if:
        1. v is adjacent to end_vertex
        2. v is not used
        3. v is not adjacent to any other used vertex (except end_vertex)
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            # Check v is not adjacent to any other used vertex
            valid = True
            for u in neighbors[v]:
                if u == end_vertex:
                    continue
                if used[u]:
                    valid = False
                    break
            if valid:
                moves.append(v)
        return moves

    def score_move(used, v, end_vertex, other_end_vertex):
        """
        Score a move. Lower is better (more constrained).
        Primary: number of unvisited neighbors of v (Warnsdorff).
        Secondary: penalty if this move would reduce the other end's options.
        """
        # Count unvisited neighbors of v
        unvisited_count = 0
        for u in neighbors[v]:
            if not used[u]:
                unvisited_count += 1
        
        # Check impact on other end: if we add v, does it block the other end?
        # v is now "used", so any neighbor of other_end that is v or adjacent to v
        # becomes potentially blocked.
        other_blocked = 0
        if other_end_vertex is not None:
            for u in neighbors[other_end_vertex]:
                if u == v:
                    # Direct block
                    other_blocked += 1
                else:
                    # Check if u would be blocked by v (u adjacent to v)
                    for w in neighbors[u]:
                        if w == v:
                            other_blocked += 1
                            break
        
        # Combined score: primarily Warnsdorff, with a penalty for blocking other end
        # We want to minimize, so add other_blocked as a small penalty
        score = unvisited_count + 0.5 * other_blocked
        return score

    def bidirectional_grow(initial_path=None, rng=None):
        if rng is None:
            rng = random
        
        if initial_path is None:
            v0 = rng.randint(0, MAX_VERTS - 1)
            v1 = rng.choice(neighbors[v0])
            path = [v0, v1]
        else:
            path = list(initial_path)
            
        if len(path) < 2:
            return path
            
        used = [False] * MAX_VERTS
        for v in path:
            used[v] = True
            
        while True:
            if time.time() - start_time > DEADLINE:
                return path
            
            left_end = path[0]
            right_end = path[-1]
            
            left_moves = get_valid_moves(used, left_end)
            right_moves = get_valid_moves(used, right_end)
            
            if not left_moves and not right_moves:
                return path
            
            # Score all moves
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                scored = [(v, score_move(used, v, left_end, right_end)) for v in left_moves]
                min_score = min(s for _, s in scored)
                candidates = [v for v, s in scored if s == min_score]
                best_left = rng.choice(candidates)
                best_left_score = min_score
                
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                scored = [(v, score_move(used, v, right_end, left_end)) for v in right_moves]
                min_score = min(s for _, s in scored)
                candidates = [v for v, s in scored if s == min_score]
                best_right = rng.choice(candidates)
                best_right_score = min_score
            
            # Choose end: prefer the more constrained side (lower score)
            if not left_moves:
                path.append(best_right)
                used[best_right] = True
            elif not right_moves:
                path.insert(0, best_left)
                used[best_left] = True
            else:
                if best_left_score < best_right_score:
                    path.insert(0, best_left)
                    used[best_left] = True
                elif best_right_score < best_left_score:
                    path.append(best_right)
                    used[best_right] = True
                else:
                    # Tie: pick randomly, but weight by number of options
                    if rng.random() < 0.5:
                        path.insert(0, best_left)
                        used[best_left] = True
                    else:
                        path.append(best_right)
                        used[best_right] = True

    def generate_initial_greedy(rng=None):
        """
        Generate a short initial path using greedy extension.
        """
        if rng is None:
            rng = random
            
        # Try different starting vertices
        v0 = rng.randint(0, MAX_VERTS - 1)
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        for _ in range(20):
            current = path[-1]
            moves = get_valid_moves(used, current)
            if not moves:
                break
            # Pick the one with lowest Warnsdorff score
            scored = [(v, sum(1 for u in neighbors[v] if not used[u])) for v in moves]
            min_score = min(s for _, s in scored)
            candidates = [v for v, s in scored if s == min_score]
            next_v = rng.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            
        return path

    attempts = 0
    
    # Phase 1: Structured attempts with varied seeds
    for i in range(8):
        if time.time() - start_time > DEADLINE:
            break
        rng = random.Random(i * 7919 + 13)  # Deterministic but varied
        initial = generate_initial_greedy(rng)
        if len(initial) < 2:
            continue
        p = bidirectional_grow(initial, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Phase 2: Random restarts with diverse initial edges
    restart_rng = random.Random()
    while time.time() - start_time < DEADLINE:
        # Create a random initial edge
        v0 = restart_rng.randint(0, MAX_VERTS - 1)
        v1 = restart_rng.choice(neighbors[v0])
        initial = [v0, v1]
        
        # Sometimes extend the initial path a bit more before bidirectional growth
        if restart_rng.random() < 0.3:
            used = [False] * MAX_VERTS
            for v in initial:
                used[v] = True
            for _ in range(restart_rng.randint(2, 6)):
                current = initial[-1]
                moves = get_valid_moves(used, current)
                if not moves:
                    break
                scored = [(v, sum(1 for u in neighbors[v] if not used[u])) for v in moves]
                min_score = min(s for _, s in scored)
                candidates = [v for v, s in scored if s == min_score]
                next_v = restart_rng.choice(candidates)
                initial.append(next_v)
                used[next_v] = True
        
        p = bidirectional_grow(initial, restart_rng)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        attempts += 1

    if not best_path:
        best_path = [0, 1]
        
    return best_path
