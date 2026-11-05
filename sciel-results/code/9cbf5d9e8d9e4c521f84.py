"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9

# EVOLVE-BLOCK-START
def _greedy_extend_fast(path, used, bit_order=None):
    """Fast greedy extension with precomputed bit order. Returns final path length."""
    if bit_order is None:
        bit_order = list(range(N))
    while True:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check distance >= 2 from all non-last vertices
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return len(path)
        path.append(nxt)
        used.add(nxt)

def _try_extend_from(path, used, deadline, bit_order=None):
    """Try to extend path greedily, with time check. Returns new length."""
    if bit_order is None:
        bit_order = list(range(N))
    while time.time() < deadline:
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 1):
                if (v ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return len(path)
        path.append(nxt)
        used.add(nxt)
    return len(path)

def _local_search_extend(path, used, deadline, max_attempts=50):
    """
    Local search: try to find a new endpoint by swapping the tail.
    For each vertex w at distance >= 2 from all path vertices except possibly
    the last k, check if we can form a valid path ending at w, then extend.
    """
    best_path = list(path)
    best_len = len(path)
    n = len(path)
    
    if n < 2:
        return best_path
    
    # Try to find alternative continuations by looking at neighbors of the last
    # few vertices and seeing if we can jump to a new position
    last_k = min(5, n)
    for k in range(1, last_k + 1):
        if time.time() > deadline:
            break
        # The vertex at position n - k is a potential "branch point"
        branch_idx = n - k
        branch_vertex = path[branch_idx]
        remaining_tail = path[branch_idx + 1:]  # vertices after branch
        
        # Try each neighbor of branch_vertex that isn't in the tail
        for b in range(N):
            if time.time() > deadline:
                break
            w = branch_vertex ^ (1 << b)
            if w in used:
                continue
            # w must be at distance >= 2 from all vertices in path except
            # those in the tail we're about to discard
            # Actually, w replaces the tail, so w must be valid against
            # path[0:branch_idx+1] (the prefix including branch)
            valid = True
            for i in range(branch_idx + 1):
                if i == branch_idx:
                    continue  # adjacent to branch, that's fine
                if (w ^ path[i]).bit_count() == 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Try building a new tail from w
            new_path = path[:branch_idx + 1] + [w]
            new_used = set(path[:branch_idx + 1])
            new_used.add(w)
            
            # Greedily extend from w
            bit_order = list(range(N))
            random.shuffle(bit_order)
            _try_extend_from(new_path, new_used, deadline, bit_order)
            
            if len(new_path) > best_len:
                best_len = len(new_path)
                best_path = list(new_path)
    
    return best_path

def _beam_search(deadline, beam_width=10, max_time_per_beam=0.5):
    """
    Beam search: maintain multiple partial snakes, extend the most promising.
    """
    # Start with a few random initial snakes
    beams = []
    for _ in range(beam_width):
        start = random.randint(0, 511)
        path = [start]
        used = {start}
        bit_order = list(range(N))
        random.shuffle(bit_order)
        _greedy_extend_fast(path, used, bit_order)
        beams.append((path, used))
    
    best_path = max(beams, key=lambda x: len(x[0]))[0]
    
    # Iteratively extend beams
    while time.time() < deadline:
        # Sort beams by length
        beams.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Try to extend each beam with a local search step
        for i, (path, used) in enumerate(beams):
            if time.time() > deadline:
                break
            new_path = _local_search_extend(path, used, deadline, max_attempts=10)
            if len(new_path) > len(path):
                beams[i] = (new_path, set(new_path))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        
        # Add a fresh random beam to maintain diversity
        if time.time() < deadline:
            start = random.randint(0, 511)
            path = [start]
            used = {start}
            bit_order = list(range(N))
            random.shuffle(bit_order)
            _greedy_extend_fast(path, used, bit_order)
            beams.append((path, used))
            # Keep only top beam_width
            beams.sort(key=lambda x: len(x[0]), reverse=True)
            beams = beams[:beam_width]
    
    return best_path

def construct():
    """Construct the longest snake in Q_9 using multi-strategy search."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_snake = []
    
    # Phase 1: Simple greedy from vertex 0 (baseline)
    path = [0]
    used = {0}
    bit_order = list(range(N))
    _greedy_extend_fast(path, used, bit_order)
    best_snake = list(path)
    
    # Phase 2: Greedy from many starting vertices with shuffled bit orders
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        start = random.randint(0, 511)
        path = [start]
        used = {start}
        bit_order = list(range(N))
        random.shuffle(bit_order)
        _greedy_extend_fast(path, used, bit_order)
        if len(path) > len(best_snake):
            best_snake = list(path)
    
    # Phase 3: Local search refinement on the best snake
    if time.time() < deadline - 5.0:
        path = list(best_snake)
        used = set(best_snake)
        refined = _local_search_extend(path, used, deadline, max_attempts=100)
        if len(refined) > len(best_snake):
            best_snake = refined
    
    # Phase 4: Beam search for more exploration
    if time.time() < deadline - 2.0:
        beam_result = _beam_search(deadline, beam_width=5, max_time_per_beam=0.3)
        if len(beam_result) > len(best_snake):
            best_snake = beam_result
    
    # Phase 5: Final local search polish
    if time.time() < deadline - 1.0:
        path = list(best_snake)
        used = set(best_snake)
        refined = _local_search_extend(path, used, deadline, max_attempts=50)
        if len(refined) > len(best_snake):
            best_snake = refined
    
    return best_snake
# EVOLVE-BLOCK-END
