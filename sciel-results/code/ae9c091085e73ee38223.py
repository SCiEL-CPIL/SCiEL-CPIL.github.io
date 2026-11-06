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

# Precompute for each vertex: bitmask of vertices at distance >= 2 from v
# (i.e., not v itself and not neighbors)
NON_NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    NON_NEIGHBOR_MASK[v] = MASK ^ NEIGHBOR_MASK[v] ^ (1 << v)

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
        # Use bitmask: for each vertex, its forbidden set (neighbors + itself)
        # A valid snake: no two non-consecutive vertices are adjacent
        used_mask = 0
        for i, v in enumerate(path):
            if used_mask & (1 << v):
                return -1  # duplicate
            # v must not be adjacent to any vertex at position < i-1
            # (position i-1 is the previous, which IS adjacent)
            if i >= 2:
                # Check v against all earlier except immediate predecessor
                # We can check: v's neighbors intersect with used_mask minus path[i-1]
                forbidden = NEIGHBOR_MASK[v] & used_mask
                # The immediate predecessor path[i-1] IS adjacent, so remove it
                forbidden &= ~(1 << path[i-1])
                if forbidden:
                    return -1
            used_mask |= (1 << v)
        return n - 1

    def beam_search(width, max_steps, seed_priority=None):
        """Beam search: maintain up to `width` partial paths."""
        if seed_priority is None:
            seed_priority = list(range(N))
        
        # Each beam entry: (path_list, used_mask, last_vertex)
        beams = [([], 0, -1)]
        # Start with vertex 0
        beams = [([0], 1, 0)]
        
        best_found = beams[0][0]
        best_found_len = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            new_beams = []
            seen_paths = set()
            
            for path, used_mask, last_v in beams:
                cur = path[-1]
                # Try all bit flips in priority order
                candidates = []
                for b in seed_priority:
                    v = cur ^ (1 << b)
                    if used_mask & (1 << v):
                        continue
                    # Check v is not adjacent to any earlier vertex (except cur)
                    # v's neighbors that are in used_mask, excluding cur
                    forbidden = NEIGHBOR_MASK[v] & used_mask
                    forbidden &= ~(1 << cur)
                    if forbidden:
                        continue
                    candidates.append(v)
                
                if not candidates:
                    # Dead end: record this path
                    L = len(path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(path)
                    continue
                
                # Keep up to `width` total new beams; add all candidates from this beam
                # but cap total
                for v in candidates:
                    new_path = path + [v]
                    key = tuple(new_path)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    new_beams.append((new_path, used_mask | (1 << v), v))
                    
                    # Track best so far
                    L = len(new_path)
                    if L > best_found_len:
                        best_found_len = L
                        best_found = list(new_path)
                
                if len(new_beams) > width * 2:
                    break
            
            if not new_beams:
                break
            
            # Prune: keep only top `width` beams by length (and some diversity)
            new_beams.sort(key=lambda x: -len(x[0]))
            beams = new_beams[:width]
            
            # Early stop if all beams are same length and we've gone deep
            if len(beams) > 0 and all(len(b[0]) == len(beams[0][0]) for b in beams):
                # All same length; continue to see if any can extend further
                pass
        
        return best_found, best_found_len

    # Phase 0: Quick greedy to get a baseline
    def greedy(priority):
        path = [0]
        used = {0}
        for _ in range(500):
            cur = path[-1]
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if v in used:
                    continue
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path

    seed_path = greedy(list(range(N)))
    L = validate_and_score(seed_path)
    if L > 0:
        best_len = L
        best_path = seed_path

    # Phase 1: Beam search with various widths and priorities
    beam_configs = [
        (8, 200),
        (16, 200),
        (32, 180),
        (16, 220),
        (8, 250),
    ]
    
    for width, max_steps in beam_configs:
        if time.time() > deadline:
            break
        for trial in range(20):
            if time.time() > deadline:
                break
            priority = list(range(N))
            rnd.shuffle(priority)
            p, L = beam_search(width, max_steps, priority)
            if p and L > best_len:
                # Validate
                vL = validate_and_score(p)
                if vL > best_len:
                    best_len = vL
                    best_path = p

    # Phase 2: More greedy restarts with random priorities
    while time.time() < deadline:
        priority = list(range(N))
        rnd.shuffle(priority)
        p = greedy(priority)
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p

    # Phase 3: Beam search with remaining time
    while time.time() < deadline - 5:
        width = rnd.choice([8, 16, 32])
        max_steps = rnd.randint(150, 250)
        priority = list(range(N))
        rnd.shuffle(priority)
        p, L = beam_search(width, max_steps, priority)
        if p and L > best_len:
            vL = validate_and_score(p)
            if vL > best_len:
                best_len = vL
                best_path = p

    if best_path is None:
        return [0]
    return best_path
