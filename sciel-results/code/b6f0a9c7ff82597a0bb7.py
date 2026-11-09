import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0

def construct():
    start_time = time.time()
    
    # Precompute neighbors as a list of lists for fast access
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = []
    best_len = 0

    def get_valid_moves(used, end_vertex):
        """
        Returns a list of valid candidates to add to the path end 'end_vertex'.
        A candidate is valid if:
        1. It is adjacent to end_vertex.
        2. It is not used.
        3. It is not adjacent to any other used vertex (except end_vertex).
        """
        moves = []
        for v in neighbors[end_vertex]:
            if used[v]:
                continue
            
            # Check if v is adjacent to any OTHER used vertex
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

    def get_scores(used, moves):
        """
        Calculate Warnsdorff scores for a list of moves.
        Score is the number of unvisited neighbors of the candidate.
        """
        scores = []
        for v in moves:
            score = 0
            for u in neighbors[v]:
                if not used[u]:
                    score += 1
            scores.append(score)
        return scores

    def bidirectional_grow(initial_path=None, use_heuristic=True):
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
            
            left_moves = get_valid_moves(used, left_end)
            right_moves = get_valid_moves(used, right_end)
            
            if not left_moves and not right_moves:
                return path
            
            # Decide which end to grow and which vertex to pick
            # We want to pick the move with the lowest score (most constrained)
            # If one side has no moves, pick from the other.
            
            best_left = None
            best_left_score = float('inf')
            if left_moves:
                if use_heuristic:
                    left_scores = get_scores(used, left_moves)
                    min_score = min(left_scores)
                    # Pick random among those with min score to add diversity
                    candidates = [v for v, s in zip(left_moves, left_scores) if s == min_score]
                    best_left = random.choice(candidates)
                    best_left_score = min_score
                else:
                    best_left = random.choice(left_moves)
                    best_left_score = 0 # Dummy score for comparison if not using heuristic
            
            best_right = None
            best_right_score = float('inf')
            if right_moves:
                if use_heuristic:
                    right_scores = get_scores(used, right_moves)
                    min_score = min(right_scores)
                    # Pick random among those with min score to add diversity
                    candidates = [v for v, s in zip(right_moves, right_scores) if s == min_score]
                    best_right = random.choice(candidates)
                    best_right_score = min_score
                else:
                    best_right = random.choice(right_moves)
                    best_right_score = 0

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
        Generate a short initial path using greedy heuristic.
        """
        v0 = random.randint(0, MAX_VERTS - 1)
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        for _ in range(10):
            current = path[-1]
            moves = get_valid_moves(used, current)
            if not moves:
                break
            scores = get_scores(used, moves)
            min_score = min(scores)
            candidates = [v for v, s in zip(moves, scores) if s == min_score]
            next_v = random.choice(candidates)
            path.append(next_v)
            used[next_v] = True
            
        return path

    attempts = 0
    
    # Strategy:
    # 1. Try a few structured starts (e.g., starting at 0 or random)
    # 2. Then random restarts
    
    # Initial structured attempts
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        initial = generate_initial_greedy()
        if len(initial) < 2:
            continue
        p = bidirectional_grow(initial, use_heuristic=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        attempts += 1

    # Random restarts with heuristic
    while time.time() - start_time < DEADLINE:
        initial = None
        # Occasionally try without heuristic to explore different parts of the space?
        # But heuristic is generally good. Let's stick to heuristic for speed and quality.
        p = bidirectional_grow(initial, use_heuristic=True)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
        
        attempts += 1

    if not best_path:
        best_path = [0, 1]
        
    return best_path
