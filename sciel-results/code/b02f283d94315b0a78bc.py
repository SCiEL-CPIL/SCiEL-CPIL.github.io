import time
import random
import sys
import numpy as np

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
    
    # NumPy array of neighbors for vectorized operations
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    # Precompute neighbor masks: for each vertex, a boolean array of its neighbors
    # neighbor_mask[v][w] = 1 if w is a neighbor of v
    neighbor_mask = np.zeros((MAX_VERTS, MAX_VERTS), dtype=np.int8)
    for v in range(MAX_VERTS):
        for w in neighbors[v]:
            neighbor_mask[v, w] = 1

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
        # Or use greedy for first step?
        # Let's pick random neighbor for v1 to allow diversity
        v1_candidates = [v for v in neighbors[v0] if not used[v]]
        if not v1_candidates:
            return path
            
        # For the first step, validity is always true if unused
        # because deg_in_path[v1] will be 1 (only v0 is in path)
        v1 = random.choice(v1_candidates)
        path.append(v1)
        used[v1] = True
        
        # Update deg_in_path for v1's neighbors
        for w in neighbors[v1]:
            deg_in_path[w] += 1

        # Helper to calculate onward moves count for a candidate v
        # v is adjacent to end.
        # If we add v, new end is v.
        # Next move w from v is valid if:
        # 1. not used[w]
        # 2. deg_in_path[w] == 0 (before adding v) -> because v adds 1 to deg_in_path[w]
        #    So after adding v, deg_in_path[w] becomes 1, which is the required condition for w to be a valid extension of v.
        def count_onward_moves(candidate):
            count = 0
            for w in neighbors[candidate]:
                if not used[w] and deg_in_path[w] == 0:
                    count += 1
            return count
        
        # Vectorized onward moves count using NumPy
        # For a batch of candidates, compute onward moves for each
        def count_onward_moves_vectorized(candidates):
            """Compute onward moves for multiple candidates at once."""
            if not candidates:
                return []
            cand_arr = np.array(candidates, dtype=np.int32)
            # For each candidate, count neighbors w where used[w]==False and deg_in_path[w]==0
            # neighbors_np[cand_arr] gives shape (len(candidates), N)
            # We need to check used and deg_in_path for each neighbor
            # Reshape to get all neighbors
            all_neighbors = neighbors_np[cand_arr]  # shape (len(candidates), N)
            # Check used and deg_in_path
            # used_arr and deg_arr are numpy arrays
            used_arr = np.array(used, dtype=np.int8)
            deg_arr = np.array(deg_in_path, dtype=np.int32)
            # For each candidate, count valid onward moves
            valid_mask = (~used_arr[all_neighbors]) & (deg_arr[all_neighbors] == 0)
            return valid_mask.sum(axis=1)

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
            
            # Evaluate scores (Warnsdorff: min onward moves) using vectorized counting
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                scores = count_onward_moves_vectorized(valid_left)
                # Add randomness
                scores = scores + np.array([random.random() * randomness for _ in valid_left])
                min_idx = np.argmin(scores)
                min_score_left = scores[min_idx]
                # Find all with min score (within floating point tolerance)
                for i in range(len(valid_left)):
                    if abs(scores[i] - min_score_left) < 1e-9:
                        best_moves_left.append(valid_left[i])
                if not best_moves_left:
                    best_moves_left = [valid_left[min_idx]]
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                scores = count_onward_moves_vectorized(valid_right)
                scores = scores + np.array([random.random() * randomness for _ in valid_right])
                min_idx = np.argmin(scores)
                min_score_right = scores[min_idx]
                for i in range(len(valid_right)):
                    if abs(scores[i] - min_score_right) < 1e-9:
                        best_moves_right.append(valid_right[i])
                if not best_moves_right:
                    best_moves_right = [valid_right[min_idx]]

            if not valid_left:
                next_v = random.choice(best_moves_right)
                # Extend right
                path.append(next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                # Extend left
                path.insert(0, next_v)
                used[next_v] = True
                for w in neighbors[next_v]:
                    deg_in_path[w] += 1
            else:
                # Choose side based on score (tightness)
                # If scores are equal, random
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

    # Multi-start
    # Use random start vertices to break symmetry
    # We can also vary the randomness parameter
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        # Occasionally use less randomness to be more greedy
        rand_param = random.choice([0.0, 0.1, 0.5])
        p = bidirectional_grow(start_vertex=v0, time_limit=DEADLINE - (time.time() - start_time), randomness=rand_param)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            # Reset timer? No, we have a global deadline.
            # But we can record the best.

    if best_path is None:
        # Fallback
        return [0]
    
    # Try to improve the best path with a local search pass
    # Attempt to extend the best path by trying different last moves
    if time.time() - start_time < DEADLINE - 5:
        improved = try_improve_path(best_path, neighbors, start_time, DEADLINE - 5)
        if len(improved) > len(best_path):
            best_path = improved
            best_len = len(best_path)
    
    return best_path


def try_improve_path(path, neighbors, start_time, time_limit):
    """
    Try to improve a path by backtracking and trying alternative moves.
    This is a simple local search: try to extend the path from both ends.
    """
    if not path:
        return path
    
    best_path = list(path)
    best_len = len(best_path)
    
    # Try to extend from both ends
    for _ in range(10):  # Try a few times
        if time.time() - start_time > time_limit:
            break
        
        # Try extending from the right end
        current = list(best_path)
        extended = try_extend_from_end(current, neighbors, 'right', time_limit, start_time)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
        
        # Try extending from the left end
        current = list(best_path)
        extended = try_extend_from_end(current, neighbors, 'left', time_limit, start_time)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
    
    return best_path


def try_extend_from_end(path, neighbors, side, time_limit, start_time):
    """
    Try to extend a path from one end by backtracking and trying alternatives.
    """
    if not path:
        return path
    
    # Build used set
    used = [False] * MAX_VERTS
    for v in path:
        used[v] = True
    
    # deg_in_path
    deg_in_path = [0] * MAX_VERTS
    for v in path:
        for w in neighbors[v]:
            deg_in_path[w] += 1
    
    # Try to extend from the specified end
    if side == 'right':
        end_vertex = path[-1]
    else:
        end_vertex = path[0]
    
    # Get valid moves
    valid_moves = []
    for v in neighbors[end_vertex]:
        if not used[v] and deg_in_path[v] == 1:
            valid_moves.append(v)
    
    if not valid_moves:
        return path
    
    # Try each valid move
    for move in valid_moves:
        if time.time() - start_time > time_limit:
            break
        
        new_path = list(path)
        if side == 'right':
            new_path.append(move)
        else:
            new_path.insert(0, move)
        
        # Try to extend further from this new end
        further = try_extend_recursive(new_path, neighbors, side, time_limit, start_time, max_depth=20)
        if len(further) > len(path):
            return further
    
    return path


def try_extend_recursive(path, neighbors, side, time_limit, start_time, max_depth=10, current_depth=0):
    """
    Recursively try to extend the path from one end.
    """
    if current_depth >= max_depth:
        return path
    
    if time.time() - start_time > time_limit:
        return path
    
    # Build used set and deg_in_path
    used = [False] * MAX_VERTS
    for v in path:
        used[v] = True
    
    deg_in_path = [0] * MAX_VERTS
    for v in path:
        for w in neighbors[v]:
            deg_in_path[w] += 1
    
    # Get the end vertex
    if side == 'right':
        end_vertex = path[-1]
    else:
        end_vertex = path[0]
    
    # Get valid moves
    valid_moves = []
    for v in neighbors[end_vertex]:
        if not used[v] and deg_in_path[v] == 1:
            valid_moves.append(v)
    
    if not valid_moves:
        return path
    
    # Try the best move (Warnsdorff's rule)
    best_move = None
    best_score = float('inf')
    for v in valid_moves:
        # Count onward moves
        score = 0
        for w in neighbors[v]:
            if not used[w] and deg_in_path[w] == 0:
                score += 1
        if score < best_score:
            best_score = score
            best_move = v
    
    if best_move is None:
        return path
    
    # Try extending with this move
    new_path = list(path)
    if side == 'right':
        new_path.append(best_move)
    else:
        new_path.insert(0, best_move)
    
    # Recursively try to extend further
    further = try_extend_recursive(new_path, neighbors, side, time_limit, start_time, max_depth, current_depth + 1)
    return further

# EVOLVE-BLOCK-END
