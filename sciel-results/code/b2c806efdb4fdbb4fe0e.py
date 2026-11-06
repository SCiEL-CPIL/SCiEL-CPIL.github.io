"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    def is_valid_path(path):
        """Check if path is a valid induced path (snake)."""
        if not path:
            return True
        used = set(path)
        if len(used) != len(path):
            return False
        for i in range(len(path) - 1):
            if (path[i] ^ path[i+1]).bit_count() != 1:
                return False
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if (path[i] ^ path[j]).bit_count() == 1:
                    return False
        return True
    
    def get_neighbors(v):
        """Get all valid extensions from vertex v given the current blocked set."""
        return [v ^ (1 << b) for b in range(N)]
    
    def beam_search(initial_path, beam_width, max_steps, deadline):
        """Beam search to extend the path."""
        best_path = list(initial_path)
        best_len = len(initial_path)
        
        # Track blocked vertices (adjacent to any path vertex except last)
        # A vertex v is "blocked" if it's adjacent to any path[i] for i < len(path)-1
        # and v is not in the path itself
        
        def compute_blocked(path):
            """Compute set of vertices that cannot be added as next vertex."""
            blocked = set()
            # All vertices in the path are used
            blocked.update(path)
            # Vertices adjacent to any non-last path vertex
            for i in range(len(path) - 1):
                v = path[i]
                for b in range(N):
                    nb = v ^ (1 << b)
                    if nb not in path:
                        blocked.add(nb)
            return blocked
        
        def can_extend(path, candidate):
            """Check if candidate can be added to path."""
            if candidate in path:
                return False
            # Check adjacency to last vertex
            if (candidate ^ path[-1]).bit_count() != 1:
                return False
            # Check not adjacent to any non-last vertex
            for i in range(len(path) - 1):
                if (candidate ^ path[i]).bit_count() == 1:
                    return False
            return True
        
        # Beam search
        beam = [list(initial_path)]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            new_beam = []
            seen_paths = set()
            
            for path in beam:
                if time.time() > deadline:
                    break
                
                cur = path[-1]
                # Try all bit flips
                for b in range(N):
                    candidate = cur ^ (1 << b)
                    if can_extend(path, candidate):
                        new_path = path + [candidate]
                        # Use tuple for hashing
                        key = tuple(new_path[-5:])  # partial key for speed
                        if key not in seen_paths:
                            seen_paths.add(key)
                            new_beam.append(new_path)
            
            if not new_beam:
                break
            
            # Score paths by length, then by some heuristic
            # Prefer longer paths, break ties randomly for diversity
            new_beam.sort(key=lambda p: (-len(p), random.random()))
            
            # Keep top beam_width paths
            beam = new_beam[:beam_width]
            
            # Update best
            for p in beam:
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
            
            if time.time() > deadline:
                break
        
        return best_path
    
    # Start with greedy seed
    greedy_path, used = [0], {0}
    while True:
        cur, nxt = greedy_path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in greedy_path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        greedy_path.append(nxt)
        used.add(nxt)
    
    best_path = greedy_path
    best_len = len(greedy_path)
    
    # Beam search with different widths
    beam_configs = [
        (20, 50),   # width 20, up to 50 steps
        (50, 30),   # width 50, up to 30 steps  
        (100, 20),  # width 100, up to 20 steps
        (30, 40),   # width 30, up to 40 steps
    ]
    
    for beam_width, max_steps in beam_configs:
        if time.time() > deadline:
            break
        
        result = beam_search(greedy_path, beam_width, max_steps, deadline)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    # Try random restarts with different starting vertices
    if time.time() < deadline - 10:
        for start in [0, 1, 255, 256, 511]:
            if time.time() > deadline:
                break
            
            # Build a short greedy path from this start
            path, used = [start], {start}
            for _ in range(20):
                if time.time() > deadline:
                    break
                cur, nxt = path[-1], None
                for b in range(N):
                    v = cur ^ (1 << b)
                    if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                        continue
                    nxt = v
                    break
                if nxt is None:
                    break
                path.append(nxt)
                used.add(nxt)
            
            if len(path) >= 10:
                result = beam_search(path, 30, 40, deadline)
                if len(result) > best_len:
                    best_len = len(result)
                    best_path = result
    
    return best_path
# EVOLVE-BLOCK-END
