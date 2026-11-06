"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        # All four equal: c0==c1 and c0==c2 and c0==c3 (binary)
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.sum(mask)
    return int(count)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        c[i] = c[i >> 1] ^ (i & 1)
    return c

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence using bit operations."""
    c = np.zeros(n, dtype=np.int8)
    # t(i) = popcount(i) % 2
    for i in range(n):
        x = i
        cnt = 0
        while x:
            cnt += x & 1
            x >>= 1
        c[i] = cnt % 2
    return c

def _count_mono4_fast(c):
    """Faster count using numpy vectorization per d."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.sum(mask)
    return int(count)

def _try_flip(c, pos):
    """Try flipping position pos, return new count if improved."""
    old = c[pos]
    c[pos] = 1 - old
    new_count = _count_mono4_fast(c)
    c[pos] = old
    return new_count

def _local_search(c, time_limit):
    """Local search: try flipping individual positions to reduce mono-4AP count."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_fast(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for pos in range(N):
            if (time.time() - start) >= time_limit:
                break
            old_val = best_c[pos]
            best_c[pos] = 1 - old_val
            new_count = _count_mono4_fast(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[pos] = old_val
    
    return best_c, best_count

def _swap_search(c, time_limit):
    """Try swapping pairs of positions."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_fast(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start) >= time_limit:
                break
            for j in range(i + 1, N):
                if (time.time() - start) >= time_limit:
                    break
                if best_c[i] != best_c[j]:
                    # Swapping would change both positions
                    best_c[i], best_c[j] = best_c[j], best_c[i]
                    new_count = _count_mono4_fast(best_c)
                    if new_count < best_count:
                        best_count = new_count
                        improved = True
                    else:
                        best_c[i], best_c[j] = best_c[j], best_c[i]
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with Thue-Morse sequence
    c = _thue_morse_fast(N)
    best_c = c.copy()
    best_count = _count_mono4_fast(best_c)
    
    # Phase 1: Local search (flip individual positions)
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search(best_c, min(remaining - 5, 60))
    
    # Phase 2: More local search passes
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search(best_c, min(remaining - 5, 40))
    
    # Phase 3: Try swap search with different initializations
    remaining = deadline - time.time()
    if remaining > 10:
        # Try Thue-Morse with offset
        for offset in range(0, 8):
            if time.time() >= deadline - 10:
                break
            c2 = _thue_morse_fast(N)
            c2 = np.roll(c2, offset)
            c2, c2_count = _local_search(c2, min(10, deadline - time.time() - 5))
            if c2_count < best_count:
                best_c = c2
                best_count = c2_count
    
    # Phase 4: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search(best_c, min(remaining - 5, 30))
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
