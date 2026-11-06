import time
import random
import numpy as np

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute parity (even/odd) for bipartite structure
    parity = np.zeros(MAX_V, dtype=np.uint8)
    popcount = np.zeros(MAX_V, dtype=np.uint8)
    for i in range(1, MAX_V):
        popcount[i] = popcount[i >> 1] + (i & 1)
        parity[i] = popcount[i] & 1
    
    # Precompute compatible_mask[v]: bitmask of vertices u such that dist(v,u) >= 2
    # Using numpy for speed
    all_vertices = np.arange(MAX_V, dtype=np.int32)
    compatible_mask = [0] * MAX_V
    for v in range(MAX_V):
        xors = v ^ all_vertices
        dists = popcount[xors]
        # Build bitmask: for each u where dist >= 2, set bit u
        # Use chunks for efficiency
        mask = 0
        for u in range(MAX_V):
            if dists[u] >= 2:
                mask |= (1 << u)
        compatible_mask[v] = mask
    
    best_path = []
    best_len = 0
    
    def get_candidates(cur, used_mask):
        """Get valid candidates: unused neighbors of cur compatible with all used vertices."""
        candidates = []
        cur_parity = parity[cur]
        mask_others = used_mask ^ (1 << cur)
        for nb in neighbors[cur]:
            if (used_mask >> nb) & 1:
                continue
            # Bipartite: nb should have opposite parity to cur (guaranteed by edge)
            # Check compatibility with all other used vertices
            if (mask_others & ~compatible_mask[nb]) == 0:
                candidates.append(nb)
        return candidates
    
    def forward_degree(nb, used_mask):
        """Count how many valid extensions nb would have."""
        new_used = used_mask | (1 << nb)
        deg = 0
        mask_others = new_used ^ (1 << nb)
        for nn in neighbors[nb]:
            if (new_used >> nn) & 1:
                continue
            if (mask_others & ~compatible_mask[nn]) == 0:
                deg += 1
        return deg
    
    def greedy_extend(path, used_mask, randomize=True):
        """Greedily extend using Warnsdorff's rule."""
        if time.time() > deadline:
            return path, used_mask
        
        while True:
            if time.time() > deadline:
                return path, used_mask
            
            cur = path[-1]
            candidates = get_candidates(cur, used_mask)
            
            if not candidates:
                return path, used_mask
            
            # Score by forward degree (Warnsdorff)
            scored = []
            for nb in candidates:
                deg = forward_degree(nb, used_mask)
                scored.append((deg, nb))
            
            scored.sort(key=lambda x: x[0])
            
            if randomize and len(scored) > 1:
                k = min(3, len(scored))
                chosen = random.choice(scored[:k])[1]
            else:
                chosen = scored[0][1]
            
            path.append(chosen)
            used_mask |= (1 << chosen)
    
    def backtrack_extend(path, used_mask, max_depth=30):
        """
        Backtracking with Warnsdorff heuristic and bipartite-aware candidate generation.
        max_depth limits recursion depth to control time.
        """
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return path, used_mask
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) >= max_depth:
            return path, used_mask
        
        cur = path[-1]
        candidates = get_candidates(cur, used_mask)
        
        if not candidates:
            return path, used_mask
        
        # Score by forward degree
        scored = []
        for nb in candidates:
            deg = forward_degree(nb, used_mask)
            scored.append((deg, nb))
        
        scored.sort(key=lambda x: x[0])
        
        for deg, nb in scored:
            if time.time() > deadline:
                return path, used_mask
            
            path.append(nb)
            used_mask |= (1 << nb)
            
            path, used_mask = backtrack_extend(path, used_mask, max_depth)
            
            path.pop()
            used_mask ^= (1 << nb)
        
        return path, used_mask
    
    def extend_both_ends(path, used_mask):
        """Try to extend the path at both ends repeatedly."""
        if time.time() > deadline:
            return path, used_mask
        
        new_path = path[:]
        new_mask = used_mask
        
        # Prepend
        while True:
            if time.time() > deadline:
                break
            found = False
            first = new_path[0]
            mask_others = new_mask ^ (1 << first)
            for nb in neighbors[first]:
                if (new_mask >> nb) & 1:
                    continue
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = [nb] + new_path
                    new_mask |= (1 << nb)
                    found = True
                    break
            if not found:
                break
        
        # Append
        while True:
            if time.time() > deadline:
                break
            found = False
            last = new_path[-1]
            mask_others = new_mask ^ (1 << last)
            for nb in neighbors[last]:
                if (new_mask >> nb) & 1:
                    continue
                if (mask_others & ~compatible_mask[nb]) == 0:
                    new_path = new_path + [nb]
                    new_mask |= (1 << nb)
                    found = True
                    break
            if not found:
                break
        
        return new_path, new_mask
    
    # Main search
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts
    starts = list(range(MAX_V))
    random.shuffle(starts)
    
    for start in starts[:200]:
        if time.time() > deadline - 40:
            break
        
        path = [start]
        used_mask = 1 << start
        path, used_mask = greedy_extend(path, used_mask, randomize=True)
        path, used_mask = extend_both_ends(path, used_mask)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Backtracking from the best start
    if best_path:
        start = best_path[0]
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack_extend(path, used_mask, max_depth=40)
        path, used_mask = extend_both_ends(path, used_mask)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 3: More backtracking from different starts
    for start in starts:
        if time.time() > deadline - 5:
            break
        if start == (best_path[0] if best_path else -1):
            continue
        
        path = [start]
        used_mask = 1 << start
        path, used_mask = backtrack_extend(path, used_mask, max_depth=35)
        path, used_mask = extend_both_ends(path, used_mask)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    return best_path if best_path else [0]
