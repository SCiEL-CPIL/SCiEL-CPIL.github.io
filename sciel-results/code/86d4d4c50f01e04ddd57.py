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
    
    def is_valid_move(used, candidate, current_end):
        """
        Check if candidate can be added to the path end defined by current_end.
        candidate must be adjacent to current_end.
        candidate must not be used.
        candidate must not be adjacent to any other used vertex.
        """
        if used[candidate]:
            return False
        
        # Check neighbors of candidate
        # If any neighbor (other than current_end) is used, it's invalid
        for u in neighbors[candidate]:
            if u == current_end:
                continue
            if used[u]:
                return False
        return True

    def get_scored_moves(used, end_vertex, other_end_vertex=None):
        """
        Returns a list of (candidate, score) tuples.
        score is the number of unvisited neighbors of the candidate (Warnsdorff heuristic).
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            
            # Basic validity: v is not adjacent to any other used vertex
            # We know v is adjacent to end_vertex.
            # Check if v is adjacent to any OTHER used vertex.
            valid = True
            for u in neighbors[v]:
                if u == end_vertex:
                    continue
                if used[u]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate Warnsdorff score: number of unvisited neighbors of v
            # These are potential future moves from v
            score = 0
            for u in neighbors[v]:
                if not used[u]:
                    score += 1
            
            moves.append((v, score))
        
        return moves

    def bidirectional_grow(initial_path=None):
        if initial_path is None:
            v0 = random.randint(0, MAX_VERTS - 1)
            v1 = random.choice(neighbors[v0])
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
            
            left_moves = get_scored_moves(used, left_end)
            right_moves = get_scored_moves(used, right_end)
            
            if not left_moves and not right_moves:
                return path
            
            # Decide which end to grow and which vertex to pick
            # We want to pick the move with the lowest score (most constrained)
            # If one side has no moves, pick from the other.
            
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                # Find min score
                min_score = min(s for _, s in left_moves)
                # Pick random among those with min score to add diversity
                candidates = [v for v, s in left_moves if s == min_score]
                best_left = random.choice(candidates)
                best_left_score = min_score
                
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                min_score = min(s for _, s in right_moves)
                candidates = [v for v, s in right_moves if s == min_score]
                best_right = random.choice(candidates)
                best_right_score = min_score
            
            # Choose end
            if not left_moves:
                path.append(best_right)
                used[best_right] = True
            elif not right_moves:
                path.insert(0, best_left)
                used[best_left] = True
            else:
                # Both have moves. Pick the end with the more constrained move (lower score)
                # If scores are equal, pick randomly or based on count of options?
                # Usually, lower score is better.
                if best_left_score < best_right_score:
                    path.insert(0, best_left)
                    used[best_left] = True
                elif best_right_score < best_left_score:
                    path.append(best_right)
                    used[best_right] = True
                else:
                    # Tie: pick randomly between the two best options
                    if random.random() < 0.5:
                        path.insert(0, best_left)
                        used[best_left] = True
                    else:
                        path.append(best_right)
                        used[best_right] = True

    def generate_initial_greedy():
        """
        Generate a short initial path. 
        We can try to start from 0 and use a greedy approach to build a decent seed.
        """
        v0 = 0
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        for _ in range(20):
            current = path[-1]
            moves = get_scored_moves(used, current)
            if not moves:
                break
            # Pick the one with lowest score
            min_score = min(s for _, s in moves)
            candidates = [v for v, s in moves if s == min_score]
            next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            
        return path

    attempts = 0
    # Strategy:
    # 1. Try a few structured starts (e.g., starting at 0)
    # 2. Then random restarts
    
    # Initial structured attempts
    for _ in range(5):
        if time.time() - start_time > DEADLINE:
            break
        initial = generate_initial_greedy()
        if len(initial) < 2:
            continue
        p = bidirectional_grow(initial)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Random restarts
    while time.time() - start_time < DEADLINE:
        initial = None # Let bidirectional_grow create a random edge
        p = bidirectional_grow(initial)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        attempts += 1

    if not best_path:
        best_path = [0, 1]
        
    return best_path
