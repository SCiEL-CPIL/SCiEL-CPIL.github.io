import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    N = 256
    # Perfect squares up to 512
    squares = set()
    i = 1
    while i * i <= 512:
        squares.add(i * i)
        i += 1
    
    # Build conflict graph: conflict[a][b] = True if a+b is a perfect square
    # We use 0-indexed: vertex i represents number i+1
    # Self-loop: 2*(i+1) in squares
    # Edge: (i+1)+(j+1) in squares for i < j
    
    # Precompute conflicts as bitmasks for speed
    # n = 256 vertices
    n = N
    # conflicts[i] is a bitmask of vertices that conflict with vertex i
    conflicts = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i, n):
            b = j + 1
            if (a + b) in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    def is_valid(mask):
        """Check if the set represented by mask is valid."""
        # For each pair (i,j) with i<=j in mask, check no conflict
        for i in range(n):
            if (mask >> i) & 1:
                # Check self
                if (conflicts[i] >> i) & 1:
                    return False
                # Check others
                others = mask & conflicts[i] & ~((1 << (i+1)) - 1)
                if others:
                    return False
        return True
    
    # Start with the known good solution: all numbers ≡ 1 mod 3
    mask = 0
    for i in range(n):
        if (i + 1) % 3 == 1:
            mask |= (1 << i)
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Try to add elements greedily
    # For each element not in mask, try adding it and see if it's still valid
    def try_add(mask, i):
        """Try adding vertex i to mask. Returns new mask if valid, None otherwise."""
        if (mask >> i) & 1:
            return mask
        # Check if i conflicts with any existing vertex
        if mask & conflicts[i]:
            return None
        return mask | (1 << i)
    
    # Greedy addition pass
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not ((mask >> i) & 1):
                new_mask = try_add(mask, i)
                if new_mask is not None:
                    mask = new_mask
                    changed = True
        if time.time() > deadline:
            break
    
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Local search: try removing one and adding two
    # This is expensive, so we do limited iterations
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Try single removals that allow additions
        for remove_i in range(n):
            if not ((mask >> remove_i) & 1):
                continue
            # Try removing remove_i
            reduced = mask & ~(1 << remove_i)
            # Try adding pairs
            for add_i in range(n):
                if (reduced >> add_i) & 1:
                    continue
                if reduced & conflicts[add_i]:
                    continue
                new_mask = reduced | (1 << add_i)
                if bin(new_mask).count('1') > best_size:
                    # Try to add more
                    for add_j in range(add_i + 1, n):
                        if (new_mask >> add_j) & 1:
                            continue
                        if new_mask & conflicts[add_j]:
                            continue
                        candidate = new_mask | (1 << add_j)
                        if bin(candidate).count('1') > best_size:
                            best_mask = candidate
                            best_size = bin(candidate).count('1')
                            mask = candidate
                            improved = True
                            break
            if improved:
                break
        if not improved:
            # Try simpler: remove one, add one
            for remove_i in range(n):
                if not ((mask >> remove_i) & 1):
                    continue
                reduced = mask & ~(1 << remove_i)
                for add_i in range(n):
                    if (reduced >> add_i) & 1:
                        continue
                    if reduced & conflicts[add_i]:
                        continue
                    candidate = reduced | (1 << add_i)
                    # Same size, but might enable further improvements
                    # Skip for now, focus on size-increasing moves
            break
    
    # Final greedy pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(n):
            if not ((mask >> i) & 1):
                new_mask = try_add(mask, i)
                if new_mask is not None:
                    mask = new_mask
                    changed = True
    
    if bin(mask).count('1') > best_size:
        best_mask = mask
        best_size = bin(mask).count('1')
    
    # Convert back to list
    result = [i + 1 for i in range(n) if (best_mask >> i) & 1]
    
    # Verify
    sq_set = squares
    for idx, a in enumerate(result):
        for b in result[idx:]:
            if (a + b) in sq_set:
                # Fallback to known good
                return list(range(1, 257, 3))
    
    return result
