import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 120s limit

def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy: Optimized Bidirectional growth with Warnsdorff's rule.
    Uses bitmask-based used set for speed and vectorized adjacency checks.
    """
    start_time = time.time()
    best_path = None
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute bit masks for quick adjacency check
    # adj_mask[v] is a list of vertices adjacent to v
    # We can also precompute a set or use numpy for fast checks
    # But for N=9, simple loops are okay if optimized.
    # Let's try to use a global array for 'used' to speed up checks.
    
    def bidirectional_grow(start_vertex=0, time_limit=None):
        """
        Grow a snake bidirectionally with optimized checks.
        """
        v0 = start_vertex
        path = [v0]
        used = [False] * MAX_VERTS
        used[v0] = True
        
        # Pick a random neighbor to start the second end
        if not neighbors[v0]:
            return path
            
        v1 = random.choice(neighbors[v0])
        path.append(v1)
        used[v1] = True
        
        # Helper to check if a candidate is valid to extend
        # candidate is neighbor of end. Must not be used and not adjacent to any other path vertex.
        # end_idx is 0 for left, len(path)-1 for right
        def is_valid(candidate, end_idx):
            if used[candidate]:
                return False
            # Check against all path vertices except the one it connects to
            for i, p in enumerate(path):
                if i == end_idx:
                    continue
                if (candidate ^ p).bit_count() == 1:
                    return False
            return True

        def count_onward_moves(candidate, end_idx):
            """
            Count valid moves from candidate if it were added.
            candidate will be the new end.
            """
            count = 0
            for w in neighbors[candidate]:
                if used[w]:
                    continue
                # Check if w is adjacent to any path vertex
                # If w is adjacent to any p in path, it's invalid.
                # Note: w is adjacent to candidate. Candidate is NOT in path yet.
                # So we just check against path.
                valid = True
                for p in path:
                    if (w ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
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
                if is_valid(v, 0):
                    valid_left.append(v)
            
            # Get valid moves for right end
            valid_right = []
            for v in neighbors[right_end]:
                if is_valid(v, len(path)-1):
                    valid_right.append(v)
            
            if not valid_left and not valid_right:
                break
            
            # Evaluate scores
            best_moves_left = []
            min_score_left = float('inf')
            if valid_left:
                for v in valid_left:
                    score = count_onward_moves(v, 0)
                    if score < min_score_left:
                        min_score_left = score
                        best_moves_left = [v]
                    elif score == min_score_left:
                        best_moves_left.append(v)
            
            best_moves_right = []
            min_score_right = float('inf')
            if valid_right:
                for v in valid_right:
                    score = count_onward_moves(v, len(path)-1)
                    if score < min_score_right:
                        min_score_right = score
                        best_moves_right = [v]
                    elif score == min_score_right:
                        best_moves_right.append(v)

            if not valid_left:
                next_v = random.choice(best_moves_right)
                path.append(next_v)
                used[next_v] = True
            elif not valid_right:
                next_v = random.choice(best_moves_left)
                path.insert(0, next_v)
                used[next_v] = True
            else:
                # Choose side based on score (tightness)
                # If scores are equal, random
                if min_score_left < min_score_right:
                    next_v = random.choice(best_moves_left)
                    path.insert(0, next_v)
                    used[next_v] = True
                elif min_score_right < min_score_left:
                    next_v = random.choice(best_moves_right)
                    path.append(next_v)
                    used[next_v] = True
                else:
                    if random.random() < 0.5:
                        next_v = random.choice(best_moves_left)
                        path.insert(0, next_v)
                        used[next_v] = True
                    else:
                        next_v = random.choice(best_moves_right)
                        path.append(next_v)
                        used[next_v] = True

        return path

    # Multi-start
    # Use random start vertices to break symmetry
    while time.time() - start_time < DEADLINE:
        v0 = random.randint(0, MAX_VERTS - 1)
        p = bidirectional_grow(start_vertex=v0, time_limit=DEADLINE - (time.time() - start_time))
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    if best_path is None:
        # Fallback
        return [0]

    return best_path

# EVOLVE-BLOCK-END
