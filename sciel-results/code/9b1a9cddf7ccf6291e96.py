"""
Snake-in-the-box in Q_9 using constraint-based backtracking with strong pruning.
Goal: find an induced path (snake) longer than the greedy seed (104 edges).
Uses NumPy for vectorized safety checks and forward checking to maximize speed.
"""
import time
import numpy as np

N = 9
NUM_VERTICES = 1 << N  # 512

def construct():
    """
    Returns a list of distinct integers in [0, 2**N) representing an induced path
    in Q_N. Consecutive vertices differ in exactly one bit; non-consecutive
    vertices differ in at least two bits.
    """
    start_time = time.time()
    TIME_LIMIT = 115  # Leave 5s buffer for safety
    
    # Precompute adjacency matrix as a NumPy boolean array
    # adj[u, v] is True if u and v are adjacent
    adj = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=bool)
    for u in range(NUM_VERTICES):
        for b in range(N):
            v = u ^ (1 << b)
            adj[u, v] = True
    
    best_path = None
    best_len = 0
    
    def backtrack(path, cur, used, blocked):
        """
        path: list of vertices
        cur: current vertex (path[-1])
        used: np.array bool, True if vertex is in path
        blocked: np.array bool, True if vertex is adjacent to any vertex in path[:-1]
        """
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        # Find candidates: neighbors of cur that are not used and not blocked
        # blocked marks neighbors of path[:-1]
        # We need neighbors of cur that are not in used and not in blocked
        candidate_mask = adj[cur] & ~used & ~blocked
        candidates = np.where(candidate_mask)[0]
        
        if len(candidates) == 0:
            return
        
        # Heuristic: Score candidates by the number of valid next moves they allow
        # For a candidate c, the next state will have:
        # used' = used | e_c
        # blocked' = blocked | adj[c]  (because c becomes part of the "previous" history for the next step)
        # Valid next moves from c are neighbors of c that are not in used' and not in blocked'
        
        scores = np.zeros(len(candidates), dtype=int)
        for i, c in enumerate(candidates):
            next_used = used.copy()
            next_used[c] = True
            next_blocked = blocked | adj[c]
            # Count valid next steps
            # Neighbors of c: adj[c]
            # Must not be in next_used
            # Must not be in next_blocked
            valid_next = adj[c] & ~next_used & ~next_blocked
            scores[i] = np.sum(valid_next)
        
        # Sort candidates by score (ascending) to try most constrained first
        # This is a "fail fast" heuristic
        order = np.argsort(scores)
        
        for idx in order:
            c = candidates[idx]
            
            # Optional: Prune if score is 0 (no valid next moves)
            if scores[idx] == 0:
                continue
            
            # Update state
            new_used = used.copy()
            new_used[c] = True
            new_blocked = blocked | adj[c]
            
            path.append(c)
            backtrack(path, c, new_used, new_blocked)
            path.pop()
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Initialize search
    # Start with path [0, 1]
    # used: 0 and 1 are used
    # blocked: neighbors of path[:-1] = neighbors of [0]
    
    used = np.zeros(NUM_VERTICES, dtype=bool)
    used[0] = True
    used[1] = True
    
    blocked = adj[0].copy()
    
    initial_path = [0, 1]
    backtrack(initial_path, 1, used, blocked)
    
    # If we have time, try a few more restarts with different first steps
    # Note: By symmetry, starting with 0->1 is equivalent to 0->2^k.
    # However, the structure of the search tree might differ slightly due to bit ordering?
    # Actually, the hypercube is symmetric under bit permutations.
    # So 0->1, 0->2, etc. are isomorphic.
    # We should try different *second* steps if we reset?
    # Or just run the single search which should be exhaustive enough with good pruning.
    # Given the time limit, a single well-optimized search is better than many shallow ones.
    
    if best_path:
        return best_path
    else:
        # Fallback to greedy if search fails or finds nothing better
        path, used_set = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used_set or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used_set.add(nxt)
