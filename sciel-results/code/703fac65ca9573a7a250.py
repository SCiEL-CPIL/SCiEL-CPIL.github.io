"""
Snake-in-the-box in Q_9 using constraint-based backtracking with strong pruning.
Goal: find an induced path (snake) longer than the greedy seed (104 edges).
"""
import time
import random

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
    
    # Precompute neighbors for each vertex
    neighbors = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    best_path = None
    best_len = 0
    
    def is_safe_fast(v, path, used_mask):
        """Check if v can be added to the end of path using bitmask."""
        if (used_mask >> v) & 1:
            return False
        # Check if v is adjacent to any vertex in path[:-1]
        # v is adjacent to path[-1] by construction, which is allowed.
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                return False
        return True
    
    def forward_check(c, path, used_mask):
        """Check if choosing c leaves at least one valid next move."""
        new_used_mask = used_mask | (1 << c)
        # Count valid next moves from c
        valid_next = 0
        for nb in neighbors[c]:
            if (new_used_mask >> nb) & 1:
                continue
            # Check if nb is safe relative to path + [c]
            # nb must not be adjacent to any in path[:-1] and path[-1] (which is cur)
            # Actually, nb is adjacent to c, and c is adjacent to path[-1].
            # The constraint: nb not adjacent to any p in path[:-1]
            safe = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    safe = False
                    break
            if safe:
                valid_next += 1
        return valid_next > 0
    
    def backtrack(path, used_mask):
        nonlocal best_path, best_len, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        
        cur = path[-1]
        
        # Get candidates: neighbors of cur that are not used and safe
        candidates = []
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            if is_safe_fast(nb, path, used_mask):
                candidates.append(nb)
        
        if not candidates:
            return
        
        # Heuristic: Order candidates by forward-check score
        # Prefer candidates that leave more options (or use constraint propagation)
        # Try most constrained first (fewer valid next moves) to fail fast
        def score_candidate(c):
            # Count valid next moves after choosing c
            new_used_mask = used_mask | (1 << c)
            count = 0
            for nb2 in neighbors[c]:
                if (new_used_mask >> nb2) & 1:
                    continue
                # Check if nb2 is safe relative to path + [c]
                # nb2 must not be adjacent to any in path[:-1]
                safe = True
                for p in path[:-1]:
                    if (nb2 ^ p).bit_count() == 1:
                        safe = False
                        break
                if safe:
                    count += 1
            return count
        
        # Sort by score: try most constrained (lowest score) first
        candidates.sort(key=score_candidate)
        
        for c in candidates:
            # Forward check: does this choice leave any valid next move?
            if not forward_check(c, path, used_mask):
                continue
            
            path.append(c)
            new_used_mask = used_mask | (1 << c)
            backtrack(path, new_used_mask)
            path.pop()
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    # Try multiple starting configurations to escape local optima
    # By symmetry, we can fix start=0. But try different first steps.
    # Also, we can try random restarts with different initial paths.
    
    # First, try deterministic search with start=0, first_step=1
    initial_path = [0, 1]
    initial_used_mask = (1 << 0) | (1 << 1)
    backtrack(initial_path, initial_used_mask)
    
    # If we have time, try a few more restarts with different first steps
    if time.time() - start_time < TIME_LIMIT * 0.7:
        for first_step in [2, 4, 8, 16, 32, 64, 128, 256]:
            if time.time() - start_time > TIME_LIMIT * 0.9:
                break
            init_path = [0, first_step]
            init_mask = (1 << 0) | (1 << first_step)
            backtrack(init_path, init_mask)
    
    if best_path:
        return best_path
    else:
        # Fallback to greedy if search fails or finds nothing better
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
