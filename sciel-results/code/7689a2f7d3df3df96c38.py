import time
import random
import sys
import numpy as np

N = 9
MAX_V = 1 << N  # 512

def construct():
    deadline = time.time() + 115.0  # Safety margin: killed at 150s
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute popcount
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(MAX_V):
        popcount[i] = bin(i).count('1')
    
    # Precompute compatibility: for each vertex v, a set of vertices u with dist(v,u) >= 2
    # We'll use a list of lists for faster access than bitmasks
    incompatible = [set() for _ in range(MAX_V)]
    for v in range(MAX_V):
        for u in range(MAX_V):
            if v != u and popcount[v ^ u] == 1:
                incompatible[v].add(u)
                incompatible[u].add(v)
    
    # Also precompute for each v, the list of vertices at distance >= 2
    # Actually, let's keep the bitmask approach but make it faster
    # For each v, compatible_mask[v] has bits set for all u where dist(v,u) >= 2
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        mask = 0
        for u in range(MAX_V):
            if u != v and popcount[v ^ u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    def is_valid_extension(path, used_mask, candidate):
        """Check if candidate can be appended to path."""
        if (used_mask >> candidate) & 1:
            return False
        # Check distance >= 2 from all vertices in path except the last
        # The last vertex is path[-1], candidate must be adjacent to it (dist 1)
        # All other vertices in path must be at distance >= 2 from candidate
        for i in range(len(path) - 1):
            v = path[i]
            if popcount[v ^ candidate] < 2:
                return False
        return True
    
    def count_extensions(path, used_mask, candidate):
        """Count how many valid extensions candidate would allow."""
        new_used = used_mask | (1 << candidate)
        count = 0
        for nn in neighbors[candidate]:
            if (new_used >> nn) & 1:
                continue
            # Check compatibility with all vertices in path
            valid = True
            for i in range(len(path)):
                if popcount[path[i] ^ nn] < 2:
                    valid = False
                    break
            if valid:
                count += 1
        return count
    
    def greedy_extend(path, used_mask, randomize=True, seed_rng=None):
        """Greedily extend a path using Warnsdorff-like heuristic."""
        if seed_rng is None:
            seed_rng = random
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            candidates = []
            for nb in neighbors[cur]:
                if (used_mask >> nb) & 1:
                    continue
                if is_valid_extension(path, used_mask, nb):
                    new_used = used_mask | (1 << nb)
                    deg = count_extensions(path + [nb], new_used, nb)
                    candidates.append((deg, nb))
            
            if not candidates:
                return path, used_mask
            
            candidates.sort(key=lambda x: x[0])
            
            if randomize and len(candidates) > 1:
                k = min(3, len(candidates))
                chosen = seed_rng.choice(candidates[:k])[1]
            else:
                chosen = candidates[0][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
    
    def backtrack_extend(path, used_mask, depth=0):
        """
        Constraint-propagation backtracking solver.
        Uses MRV heuristic.
        """
        nonlocal best_path, best_len, deadline
        
        if time.time() > deadline:
            return path, used_mask
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        cur = path[-1]
        
        # Find valid candidates: unused vertices adjacent to cur
        candidates = []
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            if is_valid_extension(path, used_mask, nb):
                candidates.append(nb)
        
        if not candidates:
            return path, used_mask
        
        # MRV heuristic: for each candidate, count how many valid extensions it would have
        scored_candidates = []
        for nb in candidates:
            new_used = used_mask | (1 << nb)
            deg = 0
            for nn in neighbors[nb]:
                if (new_used >> nn) & 1:
                    continue
                valid = True
                for i in range(len(path)):
                    if popcount[path[i] ^ nn] < 2:
                        valid = False
                        break
                if valid:
                    deg += 1
            scored_candidates.append((deg, nb))
        
        # Sort by degree (ascending) - most constrained first
        scored_candidates.sort(key=lambda x: x[0])
        
        # Try candidates in order of increasing degree
        for deg, nb in scored_candidates:
            if time.time() > deadline:
                return path, used_mask
            
            # Extend
            path.append(nb)
            used_mask |= (1 << nb)
            
            # Recurse
            path, used_mask = backtrack_extend(path, used_mask, depth + 1)
            
            # Backtrack
            path.pop()
            used_mask ^= (1 << nb)
        
        return path, used_mask
    
    def try_extend_both_ends(path, used_mask, max_iters=20):
        """Try to extend the path at either end."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try extending the front
            first = path[0]
            for nb in neighbors[first]:
                if (used_mask >> nb) & 1:
                    continue
                valid = True
                for v in path[1:]:  # All except first
                    if popcount[v ^ nb] < 2:
                        valid = False
                        break
                if valid:
                    path.insert(0, nb)
                    used_mask |= (1 << nb)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try extending the back
            last = path[-1]
            for nb in neighbors[last]:
                if (used_mask >> nb) & 1:
                    continue
                valid = True
                for v in path[:-1]:  # All except last
                    if popcount[v ^ nb] < 2:
                        valid = False
                        break
                if valid:
                    path.append(nb)
                    used_mask |= (1 << nb)
                    improved = True
                    break
        
        return path, used_mask
    
    # Main search
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts to find good initial solutions
    start_vertices = list(range(MAX_V))
    random.shuffle(start_vertices)
    
    num_greedy = 100
    for start_idx in range(num_greedy):
        if time.time() > deadline - 80:
            break
        
        start = start_vertices[start_idx]
        path = [start]
        used_mask = 1 << start
        
        path, used_mask = greedy_extend(path, used_mask, randomize=True)
        path, used_mask = try_extend_both_ends(path, used_mask, max_iters=30)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Backtracking from the best solution found
    if best_path:
        # Start backtracking from the beginning of the best path
        start = best_path[0]
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack_extend(path, used_mask)
        
        # Also try backtracking from the end
        start = best_path[-1]
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack_extend(path, used_mask)
    
    # Phase 3: More backtracking from different starts if time remains
    remaining_starts = start_vertices[num_greedy:]
    random.shuffle(remaining_starts)
    
    for start in remaining_starts:
        if time.time() > deadline - 5:
            break
        
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack_extend(path, used_mask)
        
        # After backtracking, try to extend both ends
        if best_path:
            path2 = best_path[:]
            used_mask2 = 0
            for v in path2:
                used_mask2 |= (1 << v)
            path2, used_mask2 = try_extend_both_ends(path2, used_mask2, max_iters=50)
            if len(path2) > best_len:
                best_len = len(path2)
                best_path = path2[:]
    
    # Final polish: try to extend the best path at both ends
    if best_path:
        used_mask = 0
        for v in best_path:
            used_mask |= (1 << v)
        best_path, used_mask = try_extend_both_ends(best_path, used_mask, max_iters=100)
    
    if not best_path:
        return [0]
    
    return best_path
