"""
Bidirectional snake construction for Q_9 using Warnsdorff's rule.
Strategy:
1. Initialize snake with a single vertex.
2. Grow from the 'tail' (most recently added) or 'head' (start) based on which side has more options.
3. Use Warnsdorff's rule: at each step, choose the neighbor that has the fewest available neighbors (excluding used vertices and the immediate predecessor/successor).
   This heuristic tends to keep the path open and avoids creating isolated pockets or trapping the end.
4. If stuck, try to "repair" by backtracking a few steps or restarting with a different seed.
5. Use randomization in tie-breaking to explore diverse paths.
"""

import time
import random
import sys

N = 9
MAX_VERTS = 2 ** N

# Precompute neighbors for all vertices
# neighbors[v] is a list of 9 neighbors
NEIGHBORS = [None] * MAX_VERTS
for v in range(MAX_VERTS):
    nbrs = []
    for b in range(N):
        nbrs.append(v ^ (1 << b))
    NEIGHBORS[v] = nbrs

def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Returns a list of distinct integers representing the vertices in order.
    """
    deadline = time.time() + 115  # Safety margin
    
    best_path = None
    best_len = 0
    
    # Helper to check if a vertex is valid to add to a path
    # path: list of vertices
    # v: candidate vertex
    # side: 0 for head (insert at front), 1 for tail (append at end)
    # We need to check against the "far" end and all intermediate vertices.
    # Actually, for an induced path, v must not be adjacent to ANY vertex in the path 
    # except the one it connects to.
    
    def is_valid_addition(path, v, connect_idx):
        """
        Check if v can be added to path at position connect_idx (either 0 or len(path)-1).
        v must be adjacent to path[connect_idx].
        v must NOT be adjacent to any other vertex in path.
        """
        if v in path:
            return False
        
        # Check adjacency to the connection point is already assumed by caller (we pick neighbors)
        # Check non-adjacency to all other vertices
        for i, p in enumerate(path):
            if i == connect_idx:
                continue
            if (v ^ p).bit_count() == 1:
                return False
        return True

    def get_score(path, v, connect_idx):
        """
        Warnsdorff's heuristic: count the number of valid neighbors of v 
        that are not in the path and not the connection point.
        Lower score is better (forces the path into tighter spaces first).
        """
        count = 0
        # The set of vertices in the path for quick lookup
        # Note: path is a list, so 'in' is O(N). For N=9 and path len ~200, this is fine.
        # But we can optimize if needed.
        for nbr in NEIGHBORS[v]:
            if nbr == path[connect_idx]:
                continue
            if nbr in path:
                continue
            # Check if nbr would be valid if added? No, just count potential options.
            # Standard Warnsdorff counts available neighbors.
            count += 1
        return count

    def grow_bidirectional(path, rng, max_steps=1000):
        """
        Grow the path from both ends using Warnsdorff's rule.
        path is a list. We maintain a set for O(1) lookup if possible, 
        but for simplicity and correctness first, we use list operations 
        and rely on the small N (9) and moderate path length.
        
        Actually, let's use a set for 'in path' checks to speed up.
        """
        path_set = set(path)
        current_path = list(path)
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            # Determine which end to grow
            # Calculate options for head (index 0) and tail (index -1)
            head = current_path[0]
            tail = current_path[-1]
            
            head_candidates = []
            for nbr in NEIGHBORS[head]:
                if nbr in path_set:
                    continue
                # Check non-adjacency to all other vertices
                valid = True
                for p in current_path[1:]: # Exclude head itself
                    if (nbr ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    # Score: number of neighbors of nbr not in path_set and not head
                    score = 0
                    for nnbr in NEIGHBORS[nbr]:
                        if nnbr != head and nnbr not in path_set:
                            score += 1
                    head_candidates.append((score, nbr))
            
            tail_candidates = []
            for nbr in NEIGHBORS[tail]:
                if nbr in path_set:
                    continue
                valid = True
                for p in current_path[:-1]: # Exclude tail itself
                    if (nbr ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    score = 0
                    for nnbr in NEIGHBORS[nbr]:
                        if nnbr != tail and nnbr not in path_set:
                            score += 1
                    tail_candidates.append((score, nbr))
            
            if not head_candidates and not tail_candidates:
                break
            
            # Choose the end with fewer candidates? Or the candidate with lowest score?
            # Warnsdorff says pick the move with the fewest options.
            # We can pick the best move from either end.
            
            best_head_score = min([s for s, _ in head_candidates], default=float('inf'))
            best_tail_score = min([s for s, _ in tail_candidates], default=float('inf'))
            
            # Add some randomness to break ties or explore
            if best_head_score < best_tail_score:
                # Pick from head
                # Filter candidates with score == best_head_score
                cands = [v for s, v in head_candidates if s == best_head_score]
                v = rng.choice(cands)
                current_path.insert(0, v)
                path_set.add(v)
            elif best_tail_score < best_head_score:
                # Pick from tail
                cands = [v for s, v in tail_candidates if s == best_tail_score]
                v = rng.choice(cands)
                current_path.append(v)
                path_set.add(v)
            else:
                # Tie. Randomly pick an end, then best candidate
                if rng.random() < 0.5:
                    if head_candidates:
                        cands = [v for s, v in head_candidates if s == best_head_score]
                        v = rng.choice(cands)
                        current_path.insert(0, v)
                        path_set.add(v)
                    else:
                        cands = [v for s, v in tail_candidates if s == best_tail_score]
                        v = rng.choice(cands)
                        current_path.append(v)
                        path_set.add(v)
                else:
                    if tail_candidates:
                        cands = [v for s, v in tail_candidates if s == best_tail_score]
                        v = rng.choice(cands)
                        current_path.append(v)
                        path_set.add(v)
                    else:
                        cands = [v for s, v in head_candidates if s == best_head_score]
                        v = rng.choice(cands)
                        current_path.insert(0, v)
                        path_set.add(v)
                        
            steps += 1
            
        return current_path

    # Main loop: multiple restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Start with a random vertex
        start_v = rng.randint(0, MAX_VERTS - 1)
        
        # Initial path
        path = [start_v]
        
        # Grow
        extended_path = grow_bidirectional(path, rng, max_steps=500)
        
        if len(extended_path) > best_len:
            best_len = len(extended_path)
            best_path = extended_path
            
            # If we found a very long path, we might want to spend more time on it?
            # For now, just continue searching for better ones.
            # We can also try to extend the best path further in subsequent iterations
            # by using it as a seed.
            
    # Final check
    if best_path is None:
        # Fallback
        best_path = [0]
        
    return best_path

# EVOLVE-BLOCK-END
