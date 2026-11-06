"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
import numpy as np

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _beam_search_construct(deadline, beam_width=50, seed=42):
    """Beam search to construct a long snake in Q_9."""
    rng = random.Random(seed)
    
    # Precompute neighbors for all vertices
    neighbors = np.empty((TOTAL_VERTICES, N), dtype=np.int32)
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            neighbors[v, b] = v ^ (1 << b)
    
    # Beam: list of (path_list, used_set, last_vertex)
    # Start with multiple random initial vertices
    beam = []
    for _ in range(beam_width):
        sv = rng.randint(0, TOTAL_VERTICES - 1)
        beam.append(([sv], {sv}, sv))
    
    best_path = []
    best_len = 0
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 0.5:
            break
        
        new_beam = []
        max_candidates_per_state = 3  # Limit branching to keep it fast
        
        for path, used, last in beam:
            if time.time() > deadline - 0.5:
                break
            
            cur = last
            # Get all valid extensions
            valid_nexts = []
            for b in range(N):
                v = int(neighbors[cur, b])
                if v in used:
                    continue
                # Check against all previous vertices except the last one (which is cur)
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    valid_nexts.append(v)
            
            if not valid_nexts:
                # Dead end - keep this path as a candidate
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                continue
            
            # Score candidates by number of further extensions (2-ply lookahead)
            scored = []
            for v in valid_nexts:
                score = 0
                for b2 in range(N):
                    v2 = int(neighbors[v, b2])
                    if v2 in used or v2 == cur:
                        continue
                    ok2 = True
                    for p in path[:-1]:
                        if (v2 ^ p).bit_count() == 1:
                            ok2 = False
                            break
                    if ok2:
                        score += 1
                scored.append((score, v))
            
            # Sort by score descending, take top max_candidates_per_state
            scored.sort(key=lambda x: -x[0])
            top = scored[:max_candidates_per_state]
            
            for score, v in top:
                new_path = path + [v]
                new_used = used | {v}
                new_beam.append((new_path, new_used, v))
        
        if not new_beam:
            break
        
        # Keep only the best beam_width states by length, then by last-vertex flexibility
        new_beam.sort(key=lambda x: (-len(x[0]), x[2]))
        beam = new_beam[:beam_width]
        
        # Track best
        for path, used, last in beam:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
        
        # If all beam states are dead ends, stop
        all_dead = True
        for path, used, last in beam:
            cur = last
            for b in range(N):
                v = int(neighbors[cur, b])
                if v not in used:
                    ok = True
                    for p in path[:-1]:
                        if (v ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        all_dead = False
                        break
            if not all_dead:
                break
        if all_dead:
            break
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_path = []
    best_len = 0
    
    # Multiple beam search restarts with different seeds
    random.seed(42)
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        path = _beam_search_construct(deadline, beam_width=50, seed=seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: simple greedy from vertex 0
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
# EVOLVE-BLOCK-END
