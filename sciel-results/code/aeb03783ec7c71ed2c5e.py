"""Snake-in-the-box Q_9: Bidirectional growth with backtracking and degree-of-freedom heuristic."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N  # 512
TIME_LIMIT = 140  # safety margin below 150s hard kill

# Precompute neighbor masks: for each vertex, the 512-bit mask of its 9 neighbors
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        nb = v ^ (1 << b)
        m |= (1 << nb)
    NEIGHBOR_MASKS[v] = m

# Precompute single-bit masks for each vertex
VERTEX_MASKS = [1 << v for v in range(SIZE)]

# Precompute popcount for all 512 vertices (for tie-breaking)
POPCOUNT = [v.bit_count() for v in range(SIZE)]

def construct():
    start_time = time.time()
    random.seed(42)  # Deterministic for reproducibility, but we vary start vertices

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    def verify_snake(path):
        """Verify validity."""
        if not path:
            return False
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                diff = path[i] ^ path[j]
                if abs(i - j) == 1:
                    if diff == 0 or (diff & (diff - 1)) != 0:
                        return False
                else:
                    if diff == 0 or (diff & (diff - 1)) == 0:
                        return False
        return True

    def build_initial_mask(path):
        """Build the forbidden mask for a given path."""
        mask = 0
        for v in path:
            mask |= VERTEX_MASKS[v]
        # Internal vertices: all except first and last
        for i in range(1, len(path) - 1):
            mask |= NEIGHBOR_MASKS[path[i]]
        return mask

    def greedy_extend_bidirectional(path, max_steps=500, use_backtrack=True, backtrack_depth=3):
        """
        Extend the path from both ends.
        Uses a degree-of-freedom heuristic: prefer vertices whose neighbors
        are least likely to be forbidden (i.e., have many available neighbors
        that are not adjacent to any internal vertex).
        """
        path = list(path)
        
        # State: path, and we'll maintain mask incrementally
        # But for simplicity and correctness with backtracking, we'll rebuild mask
        # when needed, or maintain it carefully.
        
        # Let's use a state object for backtracking
        # State: (path_list, mask_int)
        # We'll do a DFS with limited depth
        
        def dfs_extend(state, depth_limit, max_extensions):
            """
            state: (path_list, mask_int)
            Returns the best path found from this state.
            """
            best_path = state[0][:]
            best_len = len(best_path)
            
            # Try to extend forward and backward
            # We'll alternate or pick the side with more options
            for _ in range(max_extensions):
                if time_left() < 1.0:
                    break
                
                path, mask = state
                if len(path) < 2:
                    break
                    
                # Check forward extension
                cur_f = path[-1]
                avail_f = NEIGHBOR_MASKS[cur_f] & ~mask
                
                # Check backward extension
                cur_b = path[0]
                avail_b = NEIGHBOR_MASKS[cur_b] & ~mask
                
                if avail_f == 0 and avail_b == 0:
                    break
                
                # Score available vertices
                def score_vertex(v, cur, mask):
                    """
                    Score a candidate vertex v.
                    Prefer vertices that have many neighbors NOT in mask and NOT adjacent to internal vertices.
                    """
                    # Count available neighbors of v
                    v_avail = NEIGHBOR_MASKS[v] & ~mask
                    # Exclude the current vertex (it's already in mask as used, but let's be explicit)
                    # Actually cur is in mask, so it's excluded.
                    # The key is: how many of v's neighbors are "safe" (not forbidden)?
                    # A neighbor is forbidden if it's in mask.
                    # So score = popcount(v_avail)
                    score = v_avail.bit_count()
                    # Tie-break: prefer lower popcount vertex (more room in the cube)
                    return score, POPCOUNT[v]
                
                candidates_f = []
                if avail_f:
                    a = avail_f
                    while a:
                        lb = a & (-a)
                        v = lb.bit_length() - 1
                        a ^= lb
                        s, pc = score_vertex(v, cur_f, mask)
                        candidates_f.append((s, pc, v))
                
                candidates_b = []
                if avail_b:
                    a = avail_b
                    while a:
                        lb = a & (-a)
                        v = lb.bit_length() - 1
                        a ^= lb
                        s, pc = score_vertex(v, cur_b, mask)
                        candidates_b.append((s, pc, v))
                
                # Choose the side with the best candidate
                best_f = max(candidates_f, key=lambda x: x[0]) if candidates_f else None
                best_b = max(candidates_b, key=lambda x: x[0]) if candidates_b else None
                
                if best_f is None and best_b is None:
                    break
                
                # Decide which side to extend
                # Prefer the side with higher score
                if best_f and best_b:
                    if best_f[0] > best_b[0]:
                        extend_forward = True
                    elif best_b[0] > best_f[0]:
                        extend_forward = False
                    else:
                        # Tie: prefer forward
                        extend_forward = True
                else:
                    extend_forward = (best_f is not None)
                
                # Pick the vertex
                if extend_forward:
                    # Use the best candidate, but with some randomness
                    candidates_f.sort(key=lambda x: (-x[0], x[1]))
                    # Top 3 candidates
                    k = min(3, len(candidates_f))
                    if random.random() < 0.3:
                        v = candidates_f[random.randint(0, k-1)][2]
                    else:
                        v = candidates_f[0][2]
                    
                    # Commit
                    new_path = path + [v]
                    # Update mask: old last vertex (cur_f) becomes internal
                    new_mask = mask | NEIGHBOR_MASKS[cur_f] | VERTEX_MASKS[v]
                    state = (new_path, new_mask)
                    
                    if len(new_path) > best_len:
                        best_len = len(new_path)
                        best_path = new_path[:]
                else:
                    candidates_b.sort(key=lambda x: (-x[0], x[1]))
                    k = min(3, len(candidates_b))
                    if random.random() < 0.3:
                        v = candidates_b[random.randint(0, k-1)][2]
                    else:
                        v = candidates_b[0][2]
                    
                    # Commit
                    new_path = [v] + path
                    # Update mask: old first vertex (cur_b) becomes internal
                    new_mask = mask | NEIGHBOR_MASKS[cur_b] | VERTEX_MASKS[v]
                    state = (new_path, new_mask)
                    
                    if len(new_path) > best_len:
                        best_len = len(new_path)
                        best_path = new_path[:]
            
            return best_path, state
        
        # Main loop with backtracking
        # We'll do a series of greedy extensions, and if stuck, backtrack
        current_state = (path, build_initial_mask(path))
        best_result = path[:]
        
        for attempt in range(100):
            if time_left() < 2.0:
                break
            
            # Try to extend
            extended, new_state = dfs_extend(current_state, 0, 50)
            
            if len(extended) > len(best_result):
                best_result = extended
                current_state = new_state
            else:
                # Stuck or no improvement, try backtracking
                if use_backtrack and len(current_state[0]) > 2:
                    # Undo the last move
                    p, m = current_state
                    if p[-1] not in best_result or len(p) > len(best_result):
                        # We can backtrack from the last step
                        # Actually, let's just restart from a slightly shorter prefix
                        if len(p) > len(best_result) + 1:
                            # Truncate to best_result length
                            current_state = (best_result, build_initial_mask(best_result))
                            continue
                    # If we're stuck, break
                    break
                else:
                    break
        
        return best_result

    def beam_search_bidirectional(start_vertex, beam_width=20, max_steps=300):
        """
        Beam search that grows from both ends.
        State: (path_list, mask_int)
        """
        path = [start_vertex]
        mask = VERTEX_MASKS[start_vertex]
        beams = [(path, mask)]
        
        for step in range(max_steps):
            if time_left() < 3.0:
                break
            
            new_beams = []
            for (bp, bm) in beams:
                if len(bp) < 2:
                    # Only one vertex, extend forward
                    cur = bp[-1]
                    avail = NEIGHBOR_MASKS[cur] & ~bm
                    if avail:
                        a = avail
                        while a:
                            lb = a & (-a)
                            v = lb.bit_length() - 1
                            a ^= lb
                            new_bp = bp + [v]
                            new_bm = bm | VERTEX_MASKS[v]
                            new_beams.append((new_bp, new_bm))
                    continue
                
                # Extend forward
                cur_f = bp[-1]
                avail_f = NEIGHBOR_MASKS[cur_f] & ~bm
                if avail_f:
                    a = avail_f
                    scored = []
                    while a:
                        lb = a & (-a)
                        v = lb.bit_length() - 1
                        a ^= lb
                        # Score: available neighbors of v
                        v_avail = NEIGHBOR_MASKS[v] & ~bm
                        score = v_avail.bit_count()
                        scored.append((score, POPCOUNT[v], v))
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    for s, pc, v in scored[:beam_width//max(1,len(beams))]:
                        new_bp = bp + [v]
                        new_bm = bm | NEIGHBOR_MASKS[cur_f] | VERTEX_MASKS[v]
                        new_beams.append((new_bp, new_bm))
                
                # Extend backward
                cur_b = bp[0]
                avail_b = NEIGHBOR_MASKS[cur_b] & ~bm
                if avail_b:
                    a = avail_b
                    scored = []
                    while a:
                        lb = a & (-a)
                        v = lb.bit_length() - 1
                        a ^= lb
                        v_avail = NEIGHBOR_MASKS[v] & ~bm
                        score = v_avail.bit_count()
                        scored.append((score, POPCOUNT[v], v))
                    scored.sort(key=lambda x: (-x[0], x[1]))
                    for s, pc, v in scored[:beam_width//max(1,len(beams))]:
                        new_bp = [v] + bp
                        new_bm = bm | NEIGHBOR_MASKS[cur_b] | VERTEX_MASKS[v]
                        new_beams.append((new_bp, new_bm))
            
            if not new_beams:
                break
            
            # Deduplicate by path (or just keep top by length and mask popcount)
            # To save memory, we'll just keep top beam_width
            new_beams.sort(key=lambda x: (len(x[0]), -x[1].bit_count()))
            beams = new_beams[:beam_width]
        
        if not beams:
            return [start_vertex]
        best = max(beams, key=lambda x: len(x[0]))
        return best[0]

    best_path = None
    best_len = 0

    # Phase 1: Bidirectional greedy with backtracking
    deadline1 = start_time + time_left() * 0.5
    
    # Start from 0
    p = greedy_extend_bidirectional([0], max_steps=500, use_backtrack=True)
    best_path = p
    best_len = len(p)
    
    # Multiple restarts
    attempts = 0
    while time.time() < deadline1:
        sv = random.randint(0, SIZE - 1)
        # Start with a small random path of length 2-3 to give it a direction
        start_path = [sv]
        # Add one more vertex
        avail = NEIGHBOR_MASKS[sv]
        if avail:
            a = avail
            lb = a & (-a)
            v = lb.bit_length() - 1
            start_path.append(v)
        
        p = greedy_extend_bidirectional(start_path, max_steps=500, use_backtrack=True)
        attempts += 1
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Phase 2: Beam search bidirectional
    deadline2 = start_time + time_left() * 0.8
    beam_widths = [15, 20, 30]
    bw_idx = 0
    while time.time() < deadline2:
        sv = random.randint(0, SIZE - 1)
        bw = beam_widths[bw_idx % len(beam_widths)]
        bw_idx += 1
        p = beam_search_bidirectional(sv, beam_width=bw, max_steps=300)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p

    # Phase 3: Local improvement on best path
    if time_left() > 5.0:
        # Try to extend the best path further
        improved = greedy_extend_bidirectional(best_path, max_steps=50, use_backtrack=False)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)

    # Safety: verify before returning
    if best_path and not verify_snake(best_path):
        # Fallback: simple valid path
        best_path = [0]
    
    return best_path
