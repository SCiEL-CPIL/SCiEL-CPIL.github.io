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
    nz_indices = [i for i in range(n) if coeffs[i] != 0]
    for i_idx in nz_indices:
        a = coeffs[i_idx]
        for j_idx in nz_indices:
            b = coeffs[j_idx]
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

    def build_poly(positions, values):
        coeffs = [0] * 41
        for p, v in zip(positions, values):
            if 0 <= p < 41:
                coeffs[p] = v
            else:
                return None
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

    # --- Phase 1: Structured Position Families ---
    position_sets = []

    # Consecutive blocks
    for start in range(0, 30):
        position_sets.append(list(range(start, start + 12)))

    # Arithmetic progressions (gaps 1-5)
    for gap in range(1, 6):
        for start in range(0, 41 - 11*gap + 1):
            positions = [start + i * gap for i in range(12)]
            position_sets.append(positions)

    # Two-gap patterns
    for g1 in range(1, 6):
        for g2 in range(1, 6):
            for start in range(0, 30):
                pos = [start + i * g1 for i in range(6)]
                mid = pos[-1] + 1
                pos += [mid + i * g2 for i in range(6)]
                if max(pos) <= 40:
                    position_sets.append(pos)

    # Symmetric about center
    for center in range(20, 30):
        for radius in range(1, 15):
            positions = []
            for k in range(1, 10):
                if len(positions) >= 12: break
                if center - k >= 0:
                    positions.append(center - k)
                if center + k < 41:
                    positions.append(center + k)
            if len(positions) == 12:
                positions.sort()
                position_sets.append(positions)

    # Powers of 2
    p2 = [1, 2, 4, 8, 16, 32]
    for offset in range(0, 20):
        base = [offset + x for x in p2]
        if max(base) <= 40:
            position_sets.append(base)

    # Additive-energy-guided: positions with many sum collisions
    # Try specific known sparse structures
    # e.g., positions that form a difference set
    # For 12 terms, a (12, k, lambda) difference set would be ideal
    # We'll generate some candidates via random but biased towards clusters

    # Deduplicate
    seen = set()
    unique_pos = []
    for ps in position_sets:
        key = tuple(sorted(ps))
        if key not in seen:
            seen.add(key)
            unique_pos.append(sorted(ps))
    position_sets = unique_pos

    # Sign patterns to try
    sign_patterns = [
        [1]*12,
        [1 if i%2==0 else -1 for i in range(12)],
        [1]*6 + [-1]*6,
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
    ]

    for ps in position_sets:
        if expired():
            break
        for signs in sign_patterns:
            if expired():
                break
            update_best(build_poly(ps, signs))

    # --- Phase 2: Local Search with Sign/Position Moves ---
    def local_search_fixed(start_coeffs, max_steps=2000):
        nonlocal best_count, best_coeffs
        if start_coeffs is None:
            return
        
        # Convert to 41-length
        arr = [0] * 41
        for i, v in enumerate(start_coeffs):
            if i < 41:
                arr[i] = v
                
        curr_arr = list(arr)
        # Trim for evaluation
        def get_trimmed(a):
            k = 40
            while k >= 0 and a[k] == 0:
                k -= 1
            if k < 0: return None
            return a[:k+1]
            
        curr_trimmed = get_trimmed(curr_arr)
        curr_count, _ = _try_polynomial(curr_trimmed)
        if curr_count is None:
            return
            
        for _ in range(max_steps):
            if expired():
                return
            
            new_arr = list(curr_arr)
            move_type = random.randint(0, 3)
            
            if move_type == 0:
                # Flip sign
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                new_arr[idx] = -new_arr[idx]
            elif move_type == 1:
                # Change coefficient magnitude
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                old = new_arr[idx]
                # Try small changes
                choices = [1, -1, 2, -2, 3, -3]
                choices = [c for c in choices if c != old and abs(c) <= 64]
                if choices:
                    new_arr[idx] = random.choice(choices)
            elif move_type == 2:
                # Shift position
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if not nz: continue
                idx = random.choice(nz)
                shift = random.choice([-2, -1, 1, 2, 3])
                new_idx = idx + shift
                if 0 <= new_idx < 41 and new_idx != idx:
                    if new_arr[new_idx] == 0:
                        new_arr[new_idx] = new_arr[idx]
                        new_arr[idx] = 0
                    else:
                        continue
            else:
                # Swap two positions
                nz = [i for i, v in enumerate(new_arr) if v != 0]
                if len(nz) < 2: continue
                idx1, idx2 = random.sample(nz, 2)
                new_arr[idx1], new_arr[idx2] = new_arr[idx2], new_arr[idx1]

            new_trimmed = get_trimmed(new_arr)
            if new_trimmed is None:
                continue
            c, _ = _try_polynomial(new_trimmed)
            if c is not None:
                if c < curr_count:
                    curr_count = c
                    curr_arr = list(new_arr)
                    if c < best_count:
                        best_count = c
                        best_coeffs = list(new_trimmed)

    # Run local search from best found so far
    if best_coeffs:
        local_search_fixed(best_coeffs)

    # Run local search from a few random starts
    random.seed(42)
    for _ in range(10):
        if expired():
            break
        positions = random.sample(range(41), 12)
        positions.sort()
        values = [random.choice([1, -1]) for _ in range(12)]
        start_c = build_poly(positions, values)
        if start_c:
            local_search_fixed(start_c, max_steps=500)

    # Final polish: try all sign combinations for the best position set if time permits
    # This is expensive, so only if best_coeffs is found and time is left
    if best_coeffs and not expired():
        # Extract positions
        pos = [i for i, v in enumerate(best_coeffs) if v != 0]
        if len(pos) == 12:
            # Try a subset of sign flips (not all 2^12)
            # Focus on flipping 1 or 2 signs
            for i in range(12):
                if expired(): break
                signs = [1]*12
                signs[i] = -1
                update_best(build_poly(pos, signs))
            for i in range(12):
                for j in range(i+1, 12):
                    if expired(): break
                    signs = [1]*12
                    signs[i] = -1
                    signs[j] = -1
                    update_best(build_poly(pos, signs))

    if best_coeffs is None:
        best_coeffs = [1] * 12

    return best_coeffs
