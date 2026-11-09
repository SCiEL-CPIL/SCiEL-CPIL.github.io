import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Optimized Bidirectional growth with Warnsdorff's rule using O(N) validity checks.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def bidirectional_grow(start_vertex=0, time_limit=None, randomness=0.1):
        """
        Grow a snake bidirectionally with optimized checks.
        Uses deg_in_path array for O(N) validity check.
        """
        v0 = start_vertex
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        # deg_in_path[v] = number of neighbors of v that are in the current path
        deg_in_path = [0] * MAX_VERTS
        
        # Initialize deg_in_path for the start vertex
        for w in neighbors[v0]:
            deg_in_path[w] += 1

        if not neighbors[v0]:
            return path
            
        # Randomly pick first step to break symmetry
        v1_candidates = [v for v in neighbors[v0] if not used[v]]
        if not v1_candidates:
            return path
            
        v1 = random.choice(v1_candidates)
        path.append(v1)
        used[v1] = True
        
        # Update deg_in_path for v1's neighbors
        for w in neighbors[v1]:
            deg_in_path[w] += 1

        def count_onward_moves(candidate):
            count = 0
            for w in neighbors[candidate]:
                if not used[w] and deg_in_path[w] == 0:
                    count += 1
            return count

        while True:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            left_end = path[0]
            right_end = path[-1]
            
            # Get valid moves for left end
            valid_left = []
            for v in neighbors[left_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_left.append(v)
            
            # Get valid moves for right end
            valid_right = []
            for v in neighbors[right_end]:
                if not used[v] and deg_in_path[v] == 1:
                    valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores (Warnsdorff: min onward moves)
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                for v in valid_left:
                    score = count_onward_moves(v)
                    score += random.random() * randomness
                    if score < min_score_left:
                        min_score_left = score
                        best_moves_left = [v]
                    elif score == min_score_left:
                        best_moves_left.append(v)
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                for v in valid_right:
                    score = count_onward_moves(v)
                    score += random.random() * randomness
                    if score < min_score_right:
                        min_score_right = score
                        best_moves_right = [v]
                    elif score == min_score_right:
                        best_moves_right.append(v)

            if not valid_left:
                next_v = random.choice(best_moves_right)
                path.append(next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                path.insert(0, next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            else:
                if min_score_left < min_score_right:
                    next_v = random.choice(best_moves_left)
                    path.insert(0, next_v)
                    used[next_v] = True
                    for w in neighbors[next_v]:
                        deg_in_path[w] += 1
                elif min_score_right < min_score_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used[next_v] = True
                    for w in neighbors[next_v]:
                        deg_in_path[w] += 1
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.insert(0, next_v)
                        used[next_v] = True
                        for w in neighbors[next_v]:
                            deg_in_path[w] += 1
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used[next_v] = True
                        for w in neighbors[next_v]:
                            deg_in_path[w] += 1

        return path

    def try_improve_path(path, time_limit=None):
        """
        Attempt to improve a path by removing a vertex and regrowing.
        """
        if len(path) <= 2:
            return path
        
        # Try removing a vertex from the middle of the path
        # We pick a random index to remove, then regrow from the ends
        # To be safe, we only try a few removals per call
        
        # Pick a random vertex to remove (not the ends)
        # Actually, removing from the ends is just shortening.
        # Removing from the middle splits the path. 
        # A better strategy: Remove a vertex, then try to regrow from the new ends.
        
        # Let's try removing a random vertex from the middle 20% of the path
        # Or just a random vertex.
        
        # Simple heuristic: try removing a few random vertices and regrowing
        # Since regrowing is expensive, we limit attempts.
        
        # For now, a simple local search:
        # Pick a random index i. Remove path[i].
        # The path becomes path[:i] + path[i+1:].
        # This is no longer a valid path because path[i-1] and path[i+1] might not be connected.
        # So we need to regrow from path[i-1] and path[i+1] towards each other? 
        # Or just regrow from the ends of the new segment?
        
        # This is complex. Let's stick to a simpler improvement:
        # Try to extend the path if possible (should be handled by bidirectional_grow if it didn't get stuck).
        # Since bidirectional_grow stops when stuck, the path is maximal.
        
        # Alternative: Perturbation.
        # Take the path, remove a random middle vertex, and try to reconnect the gap with a new path.
        
        # Let's implement a "bridge" search.
        # Remove vertex at index i.
        # Left end: path[i-1]
        # Right end: path[i+1]
        # Find a path between path[i-1] and path[i+1] that avoids used vertices (except endpoints).
        # This is a Hamiltonian path problem on a subgraph, which is hard.
        
        # Given time constraints, we will skip complex local search and rely on
        # more diverse multi-starts with different randomness parameters.
        
        return path

    # Multi-start
    # Use random start vertices to break symmetry
    # We can also vary the randomness parameter
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        # Occasionally use less randomness to be more greedy
        rand_param = random.choice([0.0, 0.1, 0.5, 1.0])
        p = bidirectional_grow(start_vertex=v0, time_limit=DEADLINE - (time.time() - start_time), randomness=rand_param)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path
