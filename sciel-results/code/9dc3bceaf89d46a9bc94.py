"""Snake-in-the-box Q_9: bitmask-accelerated beam search with time-budgeted restarts."""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _local_improve(path, max_time, rng):
    """Try to extend/improve a path via limited backtracking at various positions."""
    best = list(path)
    best_len = len(best)
    
    # Try backtracking from various positions (start from near the end)
    for pos in range(len(path) - 1, max(0, len(path) - 15), -1):
        if time.time() > max_time:
            break
        prefix = path[:pos+1]
        used_mask = 0
        for v in prefix:
            used_mask |= (1 << v)
        forbidden_mask = 0
        for i in range(len(prefix) - 1):
            forbidden_mask |= NEIGHBOR_MASKS[prefix[i]]
        
        beam = [(prefix, used_mask, forbidden_mask)]
        local_best = list(prefix)
        local_best_len = len(prefix)
        
        steps = 0
        while steps < 25:
            if time.time() > max_time:
                break
            new_beam = []
            for p, um, fm in beam:
                cur = p[-1]
                candidates = NEIGHBOR_MASKS[cur] & ~(um | fm)
                if candidates == 0:
                    continue
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    new_p = p + [v]
                    new_um = um | (1 << v)
                    new_fm = fm | NEIGHBOR_MASKS[cur]
                    new_beam.append((new_p, new_um, new_fm))
            if not new_beam:
                break
            if len(new_beam) > 15:
                rng.shuffle(new_beam)
                beam = new_beam[:15]
            else:
                beam = new_beam
            for p, _, _ in beam:
                if len(p) > local_best_len:
                    local_best_len = len(p)
                    local_best = p
            steps += 1
        
        if local_best_len > best_len:
            best_len = local_best_len
            best = local_best
    
    return best

# Precompute bit masks for each dimension
BIT_MASKS = [1 << b for b in range(N)]
# Precompute neighbor mask for each vertex: bitmask of all 9 neighbors
NEIGHBOR_MASKS = [0] * (1 << N)
for v in range(1 << N):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m

def _greedy_extend(start, rng):
    """Greedy extension from start vertex. Returns list of vertices."""
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0  # vertices at dist 1 from any non-last path vertex
    
    while True:
        cur = path[-1]
        # Candidate neighbors of cur
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break
        # Pick first available (or random)
        if rng is not None:
            # Pick a random available bit
            bits = []
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                bits.append(lsb.bit_length() - 1)
                tmp ^= lsb
            v = bits[rng.randrange(len(bits))]
        else:
            v = (candidates & (-candidates)).bit_length() - 1
        
        path.append(v)
        used_mask |= (1 << v)
        # cur becomes non-consecutive, so its neighbors are now forbidden
        forbidden_mask |= NEIGHBOR_MASKS[cur]
    
    return path

def _beam_search(start, beam_width, max_time, rng=None):
    """
    Beam search using bitmask state for O(1) safety checks.
    State: (path_list, used_mask, forbidden_mask)
    Returns list of vertices.
    """
    # Each beam entry: (path_list, used_mask, forbidden_mask)
    beam = [([start], 1 << start, 0)]
    best_path = [start]
    best_len = 1
    
    while True:
        if time.time() > max_time:
            break
        
        new_beam = []
        for path, used_mask, forbidden_mask in beam:
            cur = path[-1]
            # Available candidates: neighbors of cur not in used or forbidden
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates == 0:
                continue
            
            # Iterate over set bits in candidates
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                # cur becomes non-consecutive; add its neighbors to forbidden
                new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                new_beam.append((new_path, new_used, new_forbidden))
        
        if not new_beam:
            break
        
        # Sort by length descending, keep top beam_width
        # All paths in a single step have same length, so sorting is stable/diverse
        # Use a diversity-aware selection: keep first beam_width unique paths
        if len(new_beam) > beam_width:
            # Sort by path content for diversity (lexicographic)
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
        
        # Track best
        for p, _, _ in beam:
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
        
        # Early exit if no beam can extend (checked at top of next iteration)
    
    return best_path

def _beam_search_random(start, beam_width, max_time, rng):
    """Beam search with random candidate ordering for diversity."""
    beam = [([start], 1 << start, 0)]
    best_path = [start]
    best_len = 1
    
    while True:
        if time.time() > max_time:
            break
        
        new_beam = []
        for path, used_mask, forbidden_mask in beam:
            cur = path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
            if candidates == 0:
                continue
            
            # Get list of candidate vertices and shuffle
            bits = []
            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                bits.append(lsb.bit_length() - 1)
                tmp ^= lsb
            
            rng.shuffle(bits)
            
            for v in bits:
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
                new_beam.append((new_path, new_used, new_forbidden))
        
        if not new_beam:
            break
        
        if len(new_beam) > beam_width:
            # Random selection for diversity
            rng.shuffle(new_beam)
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
        
        for p, _, _ in beam:
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_path = None
    best_len = 0
    rng = random.Random(42)
    
    # Greedy seed
    path = _greedy_extend(0, None)
    best_path = list(path)
    best_len = len(path)
    
    # Beam search with increasing widths from start=0
    beam_widths = [50, 100, 200, 400, 800]
    
    for bw in beam_widths:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        t0 = time.time()
        alloc = min(20.0, remaining * 0.22)
        result = _beam_search(0, bw, t0 + alloc, None)
        if len(result) > best_len:
            best_len = len(result)
            best_path = list(result)
    
    # Random-start beam searches
    if time.time() < deadline:
        starts = list(range(1 << N))
        rng.shuffle(starts)
        for start in starts[:25]:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 10:
                break
            t0 = time.time()
            alloc = min(6.0, remaining - 3.0)
            if alloc < 2:
                break
            if rng.random() < 0.5:
                result = _beam_search(start, 100, t0 + alloc, None)
            else:
                result = _beam_search_random(start, 100, t0 + alloc, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = list(result)
    
    # Final: try largest beam with best remaining time
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 15:
            t0 = time.time()
            result = _beam_search(0, 1000, t0 + remaining - 2.0, None)
            if len(result) > best_len:
                best_len = len(result)
                best_path = list(result)
    
    # Local improvement on the best path found
    if time.time() < deadline and best_path is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            improved = _local_improve(best_path, time.time() + remaining - 2.0, rng)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = improved
    
    return best_path
