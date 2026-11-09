"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    vectors = []

    # Part 1: Scaled E8 lattice vectors in dimensions 0-7 (240 vectors)
    # E8 minimal vectors have norm sqrt(2). Scaling by 2 gives norm 2*sqrt(2) = sqrt(8).
    
    # Type 1: Vectors with two non-zero coordinates (+/-1) in E8.
    # Scaled: (+/-2, +/-2, 0...) in 8 dims
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                if not elapsed_ok():
                    return np.array(vectors, dtype=np.int64)

    # Type 2: Vectors with eight non-zero coordinates (+/-1/2) in E8.
    # Condition: even number of minus signs.
    # Scaled: (+/-1, +/-1, ..., +/-1) (8 dims), even parity of -1s
    for mask in range(256):
        if bin(mask).count("1") % 2 == 0:
            v = [0] * D
            for k in range(8):
                if mask & (1 << k):
                    v[k] = -1
                else:
                    v[k] = 1
            vectors.append(v)
            if not elapsed_ok():
                return np.array(vectors, dtype=np.int64)

    # Part 2: Vectors in the remaining 3 dimensions (indices 8, 9, 10)
    # Permutations of (+/-2, +/-2, 0)
    idxs = [8, 9, 10]
    for i, j in itertools.combinations(idxs, 2):
        k = idxs[0] + idxs[1] + idxs[2] - i - j
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                v[k] = 0
                vectors.append(v)
                if not elapsed_ok():
                    return np.array(vectors, dtype=np.int64)

    # Now we have 252 vectors. Let's try to add more.
    # The key insight: we can mix dimensions. Let's try vectors with norm sqrt(8)
    # that span all 11 dimensions.
    
    # Strategy: Try to add vectors of the form with coordinates in {0, ±1, ±2} 
    # such that sum of squares = 8, and check distance constraint.
    
    # Current vectors have norm squared = 8.
    # For a new vector v with norm^2 = 8, we need |v - u|^2 >= 8 for all existing u.
    # |v-u|^2 = |v|^2 + |u|^2 - 2 v·u = 16 - 2 v·u >= 8
    # => v·u <= 4 for all u.
    # Also for negative: |v - (-u)|^2 = 16 - 2(-v·u) = 16 + 2 v·u >= 8
    # => v·u >= -4 for all u.
    # So we need |v·u| <= 4 for all existing u.
    
    # Also need v·v = 8.
    
    # Let's generate candidate vectors with integer coordinates and norm^2 = 8.
    # Possible patterns for norm^2 = 8:
    # - 2,2,0,0,0,0,0,0,0,0,0 (and permutations/signs)
    # - 1,1,1,1,1,1,1,1,0,0,0 (and sign variations)
    # - 2,1,1,0,0,0,0,0,0,0,0 (4+1+1=6, no)
    # - 2,2,? -> 4+4=8, so (±2,±2,0,...)
    # - (±2, ±1, ±1, ...) -> 4+1+1=6, need 2 more. 4+1+1+1+1=8 -> (±2,±1,±1,±1,±1)
    # - 1,1,1,1,1,1,1,1 -> 8 ones
    # - 2,2 -> covered
    # - 1*8 -> covered
    
    # Actually let me enumerate all patterns of integer coords with sum of squares = 8:
    # Pattern A: [2,2,0,0,0,0,0,0,0,0,0] - 11 choose 2 * 4 = 220 possibilities
    # Pattern B: [1,1,1,1,1,1,1,1,0,0,0] - 11 choose 8 * 2^8 = 165 * 256 = 42240
    # Pattern C: [2,1,1,1,1,0,0,0,0,0,0] - 11 choose 5 * 2^5 = 462 * 32 = 14784
    # Pattern D: [1,1,1,1,1,1,1,1,1,1,0] wait that's 10, sum=10 not 8
    # Pattern E: [2,2,? no more]
    # Pattern F: [1,1,1,1,1,1,1,1] = 8, done
    # Pattern G: [2,1,1,1,1] = 4+1+1+1+1 = 8, done
    # Pattern H: [2,2] = 4+4 = 8, done
    # Pattern I: [sqrt(8)] no
    # Pattern J: [2,1,1,1,1,0,...] done
    # Pattern K: [1,1,1,1,1,1,1,1,0,0,0] done
    # Pattern L: [2,2,0,...] done
    
    # Let me just try pattern B and C candidates, checking the constraint.
    # This could be slow, so let's be strategic.
    
    # For now, let's try to add a few more by checking small sets.
    # Actually, the E8 part is already very tight. Let me verify the current set is valid
    # and then try a greedy addition of a few more vectors.
    
    if not elapsed_ok():
        return np.array(vectors, dtype=np.int64)
    
    arr = np.array(vectors, dtype=np.int64)
    
    # Try adding vectors from pattern B: eight ±1's in some 8 positions, rest 0
    # We already have the ones in positions 0-7. Let's try other position sets.
    added = 0
    for pos_set in itertools.combinations(range(D), 8):
        if not elapsed_ok():
            break
        # Skip if we already have this set (positions 0-7)
        if pos_set == tuple(range(8)):
            continue
        # Try all even-parity sign assignments
        for mask in range(256):
            if not elapsed_ok():
                break
            if bin(mask).count("1") % 2 != 0:
                continue
            v = [0] * D
            for idx, pos in enumerate(pos_set):
                if mask & (1 << idx):
                    v[pos] = -1
                else:
                    v[pos] = 1
            v_arr = np.array(v, dtype=np.int64)
            # Check: v·v should be 8
            if np.dot(v_arr, v_arr) != 8:
                continue
            # Check distance constraint: |v - u|^2 >= 8 for all u
            # i.e., |v·u| <= 4 for all u (since v·v = u·u = 8)
            dots = arr @ v_arr
            if np.max(np.abs(dots)) <= 4:
                vectors.append(v)
                arr = np.vstack([arr, v_arr])
                added += 1
                if added >= 50:  # limit to keep it fast
                    break
        if added >= 50:
            break
    
    if not elapsed_ok():
        return np.array(vectors, dtype=np.int64)
    
    # Also try pattern C: one ±2 and four ±1's
    added_c = 0
    for pos_2 in range(D):
        if not elapsed_ok():
            break
        for pos_set_1 in itertools.combinations([p for p in range(D) if p != pos_2], 4):
            if not elapsed_ok():
                break
            for s2 in (1, -1):
                if not elapsed_ok():
                    break
                for mask in range(16):
                    if not elapsed_ok():
                        break
                    v = [0] * D
                    v[pos_2] = 2 * s2
                    for idx, pos in enumerate(pos_set_1):
                        if mask & (1 << idx):
                            v[pos] = -1
                        else:
                            v[pos] = 1
                    v_arr = np.array(v, dtype=np.int64)
                    if np.dot(v_arr, v_arr) != 8:
                        continue
                    dots = arr @ v_arr
                    if np.max(np.abs(dots)) <= 4:
                        vectors.append(v)
                        arr = np.vstack([arr, v_arr])
                        added_c += 1
                        if added_c >= 30:
                            break
                if added_c >= 30:
                    break
            if added_c >= 30:
                break
        if added_c >= 30:
            break
    
    return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
