import time
import random

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Seed solution
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
        
    target_n = 39
    all_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        neighbors = [0] * target_n
        
        for u, v in all_edges:
            if neighbors[u] & neighbors[v]:
                continue
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            
        # Check if alpha <= 10
        # Use a fast heuristic first, then exact search if needed
        # Heuristic: Greedy independent set
        # If greedy finds > 10, it's definitely invalid.
        # If greedy finds <= 10, it might be valid, need exact check.
        
        # Simple greedy: pick vertex with min degree, remove it and its neighbors
        # This gives a lower bound on alpha. If lower bound > 10, invalid.
        # Actually, greedy gives a valid independent set, so size is <= alpha.
        # If we find a set of size 11, alpha >= 11 -> Invalid.
        
        if find_independent_set_11_fast(neighbors, target_n, time.time()):
            continue # Invalid, try next
            
        # If fast check didn't find 11, do a more thorough check
        # But to save time, if we are close to deadline, we might just accept?
        # No, must be valid.
        
        if time.time() > deadline - 5:
            break
            
        # Exact check with timeout
        if has_independent_set_size_11_exact(neighbors, target_n, time.time()):
            continue # Invalid
            
        # Valid!
        edges = []
        for i in range(target_n):
            mask = neighbors[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask -= lsb
                
        return {'n': target_n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}

def find_independent_set_11_fast(neighbors, n, current_time):
    """Tries to find an independent set of size 11 using greedy/randomized heuristics."""
    # Try a few random greedy passes
    for _ in range(10):
        available = list(range(n))
        random.shuffle(available)
        independent_set = []
        blocked = set()
        
        for v in available:
            if v in blocked:
                continue
            independent_set.append(v)
            if len(independent_set) == 11:
                return True
            # Block neighbors
            nb_mask = neighbors[v]
            while nb_mask:
                lsb = nb_mask & -nb_mask
                u = lsb.bit_length() - 1
                blocked.add(u)
                nb_mask -= lsb
                
    return False

def has_independent_set_size_11_exact(neighbors, n, start_time):
    """Exact backtracking search for independent set of size 11. Returns True if found."""
    # Sort by degree
    degrees = [bin(neighbors[i]).count('1') for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Use a global timeout flag
    timeout = [False]
    
    def backtrack(idx, current_size, available_mask):
        if timeout[0]:
            return False
        if current_size == 11:
            return True
        if idx >= n:
            return False
        if current_size + bin(available_mask).count('1') < 11:
            return False
            
        if time.time() > start_time + 0.5: # 0.5s limit per check
            timeout[0] = True
            return False
            
        v = vertices[idx]
        
        # Try include v
        if available_mask & (1 << v):
            new_avail = available_mask & ~(1 << v) & ~neighbors[v]
            if backtrack(idx + 1, current_size + 1, new_avail):
                return True
                
        # Try exclude v
        if backtrack(idx + 1, current_size, available_mask & ~(1 << v)):
            return True
            
        return False

    initial_mask = (1 << n) - 1
    return backtrack(0, 0, initial_mask)
