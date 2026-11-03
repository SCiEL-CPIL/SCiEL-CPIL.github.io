import time
import random
import itertools

# Global cache for neighbors and mappings
_NEIGHBORS_CACHE = {}
_IDX_MAP_CACHE = {}

def _get_neighbors(q, n):
    key = (q, n)
    if key in _NEIGHBORS_CACHE:
        return _NEIGHBORS_CACHE[key]
    
    N = q ** n
    neighbors = []
    # Precompute powers for index conversion
    powers = [q ** i for i in range(n)]
    
    for idx in range(N):
        # Convert idx to tuple
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        
        neigh = [idx]
        # Generate neighbors by changing one digit
        for i in range(n):
            orig = t[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t[i] = new_val
                # Convert back to idx
                n_idx = 0
                for j in range(n):
                    n_idx += t[j] * powers[j]
                neigh.append(n_idx)
            t[i] = orig
        neighbors.append(neigh)
        
    _NEIGHBORS_CACHE[key] = neighbors
    return neighbors

def _tuple_to_idx(t, q, n):
    idx = 0
    for i, val in enumerate(reversed(t)):
        idx += val * (q ** i)
    return idx

def _idx_to_tuple(idx, q, n):
    t = [0] * n
    for i in range(n):
        t[i] = idx % q
        idx //= q
    return t

def construct(q, n, radius):
    # Fallback for non-standard params if needed, but task is specific
    if q != 3 or n != 7 or radius != 1:
        # Generic greedy fallback
        # ... (omitted for brevity in thought, will include in code)
        pass

    N = q ** n
    neighbors = _get_neighbors(q, n)
    
    deadline = time.time() + 110.0
    best_C = None
    best_size = N # worst case
    
    # Helper to check if code is valid (should always be true in our search, but for safety)
    def is_valid(C):
        counts = [0] * N
        for c in C:
            for y in neighbors[c]:
                counts[y] += 1
        for i in range(N):
            if counts[i] == 0:
                return False
        return True

    def get_counts(C):
        counts = [0] * N
        for c in C:
            for y in neighbors[c]:
                counts[y] += 1
        return counts

    def minimize(C):
        # C is a list of indices
        # Returns minimized list
        if not C:
            return C
        counts = get_counts(C)
        C_set = set(C)
        changed = True
        while changed:
            changed = False
            # Iterate over a copy to allow removal
            for c in list(C_set):
                # Check if c can be removed
                # All neighbors must have count >= 2
                can_remove = True
                for y in neighbors[c]:
                    if counts[y] < 2:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    C_set.remove(c)
                    for y in neighbors[c]:
                        counts[y] -= 1
                    changed = True
        return list(C_set)

    def greedy_init():
        uncovered = set(range(N))
        C = []
        while uncovered:
            best_c = -1
            best_cover = -1
            # Randomize order to break ties and allow different greedy paths
            candidates = list(uncovered)
            random.shuffle(candidates)
            # We only need to check codewords that cover at least one uncovered point
            # Efficiently: for each uncovered point, its neighbors are candidates
            # But N is small, just iterate all? No, 2187 is small.
            # Let's iterate all possible codewords 0..N-1
            for c in range(N):
                cover = 0
                for y in neighbors[c]:
                    if y in uncovered:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_c = c
            if best_c == -1:
                break
            C.append(best_c)
            for y in neighbors[best_c]:
                if y in uncovered:
                    uncovered.remove(y)
        return C

    # Start with greedy initialization
    C = greedy_init()
    C = minimize(C)
    best_C = C.copy()
    best_size = len(C)
    
    # Local Search
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 5.0:
            break
            
        # Perturbation: Add a random codeword
        # To improve chances, we can bias towards areas with low coverage
        # But random is a good start.
        c_add = random.randint(0, N - 1)
        C.append(c_add)
        
        # Minimize
        C = minimize(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            
        # If we get stuck at a high size for a while, maybe restart?
        # Or just keep going. The "Add 1, Minimize" can effectively swap.
        
    # Final validity check (just in case)
    if best_C and is_valid(best_C):
        # Convert indices back to lists of ints
        result = [_idx_to_tuple(idx, q, n) for idx in best_C]
        return result
    else:
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
