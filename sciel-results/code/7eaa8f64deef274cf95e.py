"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

N = 9
TIME_LIMIT = 110  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_path = _greedy_seed()
    best_len = len(best_path) - 1  # number of edges
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return TIME_LIMIT - elapsed()
    
    # Precompute neighbors for all vertices
    # For each vertex, list of adjacent vertices
    # We'll use bit_count for Hamming distance checks
    
    def is_valid_extension(path, used, candidate):
        """Check if candidate can be added to path (induced path constraint)."""
        if candidate in used:
            return False
        # Check against all previous vertices except the last one (which is the current end)
        # Consecutive must differ by 1 bit (guaranteed by construction)
        # Non-consecutive must differ by at least 2 bits
        for i in range(len(path) - 2):
            if (path[i] ^ candidate).bit_count() == 1:
                return False
        return True
    
    def dfs(path, used, depth):
        """DFS with backtracking to extend the snake."""
        nonlocal best_path, best_len
        
        if elapsed() > TIME_LIMIT - 5:  # safety margin
            return False
        
        cur = path[-1]
        
        # Try all bit flips in different orders
        bits = list(range(N))
        
        for b in bits:
            v = cur ^ (1 << b)
            if not is_valid_extension(path, used, v):
                continue
            
            # Try extending with v
            path.append(v)
            used.add(v)
            
            new_depth = depth + 1
            if new_depth > best_len:
                best_len = new_depth
                best_path = path[:]
            
            # Recurse
            dfs(path, used, new_depth)
            
            # Backtrack
            path.pop()
            used.remove(v)
            
            if elapsed() > TIME_LIMIT - 5:
                return False
        
        return False
    
    # Try multiple greedy seeds with different bit orderings
    for seed_path in _generate_seeds():
        if elapsed() > TIME_LIMIT - 10:
            break
        
        if len(seed_path) - 1 >= best_len:
            continue
        
        # Reset best to this seed if it's better
        if len(seed_path) - 1 > best_len:
            best_len = len(seed_path) - 1
            best_path = seed_path[:]
        
        # Run DFS from this seed
        used = set(seed_path)
        dfs(seed_path, used, len(seed_path) - 1)
        
        if elapsed() > TIME_LIMIT - 10:
            break
    
    # Return the best path found
    # Convert to list of distinct integers
    return list(best_path)


def _greedy_seed():
    """Generate a greedy snake with lowest-bit-first heuristic."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced path constraint
            valid = True
            for i in range(len(path) - 2):
                if (path[i] ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path


def _generate_seeds():
    """Generate multiple greedy seeds with different heuristics."""
    seeds = []
    
    # Seed 1: lowest bit first
    seeds.append(_greedy_seed())
    
    # Seed 2: highest bit first
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N - 1, -1, -1):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 2):
                if (path[i] ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    seeds.append(path)
    
    # Seed 3: random bit order
    random.seed(42)
    path, used = [0], {0}
    while True:
        cur = path[-1]
        bits = list(range(N))
        random.shuffle(bits)
        nxt = None
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 2):
                if (path[i] ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    seeds.append(path)
    
    # Seed 4: start from a different vertex
    path, used = [255], {255}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 2):
                if (path[i] ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    seeds.append(path)
    
    # Seed 5: another random start
    random.seed(123)
    start_v = random.randint(0, 2**N - 1)
    path, used = [start_v], {start_v}
    while True:
        cur = path[-1]
        bits = list(range(N))
        random.shuffle(bits)
        nxt = None
        for b in bits:
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for i in range(len(path) - 2):
                if (path[i] ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    seeds.append(path)
    
    return seeds
# EVOLVE-BLOCK-END
