import time
import random

T_START = time.time()
DEADLINE = 75.0  # Safety margin

def _square_nonzero_count(coeffs):
    """Return number of nonzero coefficients in the exact square of coeff list."""
    n = len(coeffs)
    if n == 0:
        return 0
    sq = [0] * (2 * n - 1)
    # Optimization: only iterate over non-zero terms
    nz_indices = [i for i in range(n) if coeffs[i] != 0]
    for i_idx in nz_indices:
        a = coeffs[i_idx]
        ai2 = a * a
        for j_idx in nz_indices:
            b = coeffs[j_idx]
            # i_idx + j_idx is the degree
            sq[i_idx + j_idx] += a * b
    count = 0
    for v in sq:
        if v != 0:
            count += 1
    return count

def _valid_candidate(coeffs):
    """Check all constraints on the candidate list."""
    if not isinstance(coeffs, list):
        return False
    if len(coeffs) == 0 or len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    for v in coeffs:
        if v < -64 or v > 64:
            return False
    nnz = sum(1 for v in coeffs if v != 0)
    if nnz != 12:
        return False
    return True

def _try_polynomial(coeffs):
    """If valid, return (count, coeffs); else (None, None)."""
    if not _valid_candidate(coeffs):
        return None, None
    return _square_nonzero_count(coeffs), coeffs

def construct():
    best_count = 100
    best_coeffs = None
    end_time = T_START + DEADLINE

    def expired():
        return time.time() > end_time

    # Helper to build coeff list from positions and values
    def build_poly(positions, values):
        coeffs = [0] * 41
        for p, v in zip(positions, values):
            if 0 <= p < 41:
                coeffs[p] = v
            else:
                return None
        # Trim
        k = 40
        while k >= 0 and coeffs[k] == 0:
            k -= 1
        if k < 0:
            return None
        return coeffs[:k+1]

    def update_best(coeffs):
        nonlocal best_count, best_coeffs
        if coeffs is None:
            return
        c, _ = _try_polynomial(coeffs)
        if c is not None and c < best_count:
            best_count = c
            best_coeffs = list(coeffs)

    # 1. Structured Families
    # A. Arithmetic Progressions
    for gap in range(1, 4):
        if expired(): break
        for start in range(0, 41 - 11*gap):
            if expired(): break
            positions = [start + i * gap for i in range(12)]
            # Try all +1
            update_best(build_poly(positions, [1]*12))
            # Try alternating signs
            signs = [1 if i % 2 == 0 else -1 for i in range(12)]
            update_best(build_poly(positions, signs))
            # Try symmetric signs (first half +, second half -)
            signs2 = [1]*6 + [-1]*6
            update_best(build_poly(positions, signs2))

    # B. Symmetric positions around a center
    for center in range(20, 30):
        if expired(): break
        for radius in range(1, 15):
            if expired(): break
            # Try to form 12 distinct positions
            # Simple symmetric: c-r, c-r+1... and c+r-1...
            # Just generate a set of 12 symmetric points
            # c - 5.5 + i ? No, integers.
            # Use pairs (c-k, c+k)
            positions = []
            for k in range(1, 10):
                if len(positions) >= 12: break
                if center - k >= 0:
                    positions.append(center - k)
                if center + k < 41:
                    positions.append(center + k)
            # Add center if needed? No, we need 12 distinct.
            # If we have fewer, fill with nearby?
            if len(positions) == 12:
                positions.sort()
                update_best(build_poly(positions, [1]*12))
                signs = [1 if i % 2 == 0 else -1 for i in range(12)]
                update_best(build_poly(positions, signs))

    # C. Random Search with Local Improvement
    random.seed(42)
    current_best = best_coeffs if best_coeffs else [1]*12
    current_best_count = best_count
    
    # If we haven't found anything better than 23, we start from a good random guess
    # But let's just run local search from the current best and some random starts
    
    def local_search(start_coeffs, max_steps=500):
        nonlocal best_count, best_coeffs
        if start_coeffs is None:
            return
        curr = list(start_coeffs)
        curr_count, _ = _try_polynomial(curr)
        if curr_count is None:
            return
        
        for _ in range(max_steps):
            if expired():
                return
            
            # Random move
            move_type = random.randint(0, 2)
            new_curr = list(curr)
            
            if move_type == 0:
                # Flip sign of a random nonzero term
                nz = [i for i, v in enumerate(new_curr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                new_curr[idx] = -new_curr[idx]
            elif move_type == 1:
                # Change coefficient of a random nonzero term to +/- 2
                nz = [i for i, v in enumerate(new_curr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                if new_curr[idx] == 1:
                    new_curr[idx] = 2
                elif new_curr[idx] == -1:
                    new_curr[idx] = -2
                elif new_curr[idx] == 2:
                    new_curr[idx] = 1
                else:
                    new_curr[idx] = -1
            else:
                # Shift a random nonzero term by +/- 1 or +/- 2
                nz = [i for i, v in enumerate(new_curr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                shift = random.choice([-2, -1, 1, 2])
                new_idx = idx + shift
                if 0 <= new_idx < 41 and new_idx != idx:
                    # Check if target is occupied
                    if new_curr[new_idx] != 0:
                        # Swap or skip? Skip to maintain 12 terms easily
                        continue
                    new_curr[new_idx] = new_curr[idx]
                    new_curr[idx] = 0
                    # Ensure length constraint: if we shift out of bounds, handled by check
                    # If we shift to a higher index, we might need to extend?
                    # No, we work with fixed length 41 array internally?
                    # _try_polynomial trims. But we need to keep track of positions.
                    # It's safer to work with a dict or list of 41.
                    # My curr is a trimmed list.
                    # If I shift an element, the list length might change.
                    # Let's convert to 41-length list for manipulation
                    pass
        
        # The local search above is complex with trimmed lists.
        # Let's implement a simpler version using fixed 41-length arrays.
        
    def local_search_fixed(start_coeffs):
        nonlocal best_count, best_coeffs
        if start_coeffs is None:
            return
        
        # Convert to 41-length
        arr = [0] * 41
        for i, v in enumerate(start_coeffs):
            arr[i] = v
            
        curr_arr = list(arr)
        curr_count, _ = _try_polynomial(curr_arr)
        if curr_count is None:
            return
            
        for _ in range(1000):
            if expired():
                return
            
            new_arr = list(curr_arr)
            move_type = random.randint(0, 2)
            
            if move_type == 0:
                # Flip sign
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                new_arr[idx] = -new_arr[idx]
            elif move_type == 1:
                # Change coeff
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                old = new_arr[idx]
                if old == 1: new_arr[idx] = 2
                elif old == -1: new_arr[idx] = -2
                elif old == 2: new_arr[idx] = 1
                else: new_arr[idx] = -1
            else:
                # Shift
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                shift = random.choice([-2, -1, 1, 2])
                new_idx = idx + shift
                if 0 <= new_idx < 41 and new_idx != idx:
                    if new_arr[new_idx] == 0:
                        new_arr[new_idx] = new_arr[idx]
                        new_arr[idx] = 0
                    else:
                        continue # Collision, skip move
            
            # Evaluate
            # Trim for validity check
            k = 40
            while k >= 0 and new_arr[k] == 0:
                k -= 1
            if k < 0:
                continue
            cand = new_arr[:k+1]
            c, _ = _try_polynomial(cand)
            if c is not None:
                if c < curr_count:
                    curr_count = c
                    curr_arr = list(new_arr)
                    if c < best_count:
                        best_count = c
                        best_coeffs = list(cand)
                # Accept move even if not better? No, hill climbing.
                # To escape local minima, we could accept worse moves, but let's stick to strict improvement for now
                # Or use simulated annealing?
                # Given time limits, strict hill climbing from multiple starts is safer.

    # Run local search from best found so far
    if best_coeffs:
        local_search_fixed(best_coeffs)

    # Run local search from a few random starts
    for _ in range(5):
        if expired(): break
        positions = random.sample(range(41), 12)
        positions.sort()
        values = [random.choice([1, -1]) for _ in range(12)]
        start_c = build_poly(positions, values)
        if start_c:
            local_search_fixed(start_c)

    if best_coeffs is None:
        best_coeffs = [1] * 12

    return best_coeffs
