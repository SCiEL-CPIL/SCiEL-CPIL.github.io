import random
import time
import math

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute for each vertex: bitmask of all neighbors
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def construct():
    deadline = time.time() + 112.0
    rnd = random.Random(20260715)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count."""
        if not path:
            return 0
        n = len(path)
        # Check consecutive differ by exactly 1 bit
        for i in range(n - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return -1
        # Check non-consecutive differ by at least 2 bits
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1  # duplicate
            # v must not be adjacent to any vertex at position < i-1
            if i >= 2:
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def greedy_optimized(priority):
        """Greedy search using bitmask for O(L) complexity per step."""
        path = [0]
        used_mask = 1  # bit 0 set
        blocked_mask = 0 # neighbors of vertices at index < len(path)-1
        # Initially, no vertices at index < -1, so blocked is 0.
        
        for _ in range(1000):
            if time.time() > deadline:
                break
            cur = path[-1]
            candidates = []
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                if blocked_mask & (1 << v):
                    continue
                candidates.append(v)
            
            if not candidates:
                break
            
            # Heuristic: pick candidate that maximizes future options? 
            # For now, just pick first or random. Let's pick first for speed.
            nxt = candidates[0]
            
            # Update masks
            # The current vertex 'cur' becomes part of the 'blocked' set for the NEXT step
            # because the next-next vertex cannot be adjacent to 'cur'.
            # So we add neighbors of 'cur' to blocked_mask.
            blocked_mask |= NEIGHBOR_MASK[cur]
            used_mask |= (1 << nxt)
            path.append(nxt)
            
        return path

    def beam_search_optimized(width, max_steps, priority):
        """
        Beam search using bitmask states.
        State: (path_tuple, used_mask, blocked_mask, last_vertex)
        """
        # Initial state
        beams = [((0,), 1, 0, 0)]
        
        best_found = (0,)
        best_found_len = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            new_beams = []
            seen_states = set()
            
            for path_tuple, used_mask, blocked_mask, last_v in beams:
                cur = last_v
                # Generate candidates
                for b in priority:
                    v = cur ^ (1 << b)
                    bit_v = 1 << v
                    if used_mask & bit_v:
                        continue
                    if blocked_mask & bit_v:
                        continue
                    
                    # Valid candidate
                    new_used = used_mask | bit_v
                    new_blocked = blocked_mask | NEIGHBOR_MASK[cur]
                    new_path = path_tuple + (v,)
                    state_key = (v, new_used, new_blocked)
                    
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    
                    new_beams.append((new_path, new_used, new_blocked, v))
                    
                    L = len(new_path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = new_path
            
            if not new_beams:
                break
            
            # Prune: Keep top `width` beams.
            # Heuristic: "Freedom" (number of available moves for next step)
            def score_beam(state):
                p, u, b, l = state
                avail = NEIGHBOR_MASK[l] & ~u & ~b
                return avail.bit_count()
            
            new_beams.sort(key=score_beam, reverse=True)
            beams = new_beams[:width]
            
        return list(best_found), best_found_len

    # Phase 0: Quick greedy to get a baseline
    seed_path = greedy_optimized(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 1: Beam search with various widths and priorities
    beam_configs = [
        (16, 200),
        (32, 200),
        (64, 150),
        (16, 250),
    ]
    
    for width, max_steps in beam_configs:
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            priority = list(range(N))
            rnd.shuffle(priority)
            p, L = beam_search_optimized(width, max_steps, priority)
            if p and L > best_len:
                vL = validate_and_score(p)
                if vL > best_len:
                    best_len = vL
                    best_path = p

    # Phase 2: More greedy restarts with random priorities
    while time.time() < deadline - 5:
        priority = list(range(N))
        rnd.shuffle(priority)
        p = greedy_optimized(priority)
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p

    if best_path is None:
        return [0]
    return best_path
