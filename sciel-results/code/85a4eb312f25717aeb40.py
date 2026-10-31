import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    # Count unique elements
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _generate_geometric_set(n, base, offset=0, jitter=0.0, seed=0):
    """
    Generate a set based on geometric progression with optional jitter.
    U = { floor(base^i * (1 + jitter * sin(i*freq)) + offset) }
    """
    rng = np.random.RandomState(seed)
    vals = [0]
    curr = 1.0
    for i in range(1, n):
        # Add sinusoidal jitter to break perfect geometric structure
        # This can sometimes reduce sumset collisions
        jitter_term = 0.0
        if jitter > 0:
            jitter_term = jitter * math.sin(i * 1.1)
        val = curr * (1 + jitter_term)
        val_int = int(val) + offset
        if val_int <= vals[-1]:
            val_int = vals[-1] + 1
        vals.append(val_int)
        curr *= base
    return sorted(set(vals))

def _generate_binary_like_set(n, base=2.0, shift=0.0):
    """
    Generate a set that mimics binary representation properties.
    These often have good sumset properties (Minkowski sum is small relative to range).
    """
    vals = [0]
    curr = 1
    for i in range(1, n):
        # Slight variation from pure powers of 2
        val = int(curr + shift)
        if val <= vals[-1]:
            val = vals[-1] + 1
        vals.append(val)
        curr *= base
    return sorted(set(vals))

def _optimize_params(n, base_range, jitter_range, steps=50):
    """
    Grid search over base and jitter for a fixed size n.
    """
    best_u = None
    best_s = -1.0
    
    bases = np.linspace(base_range[0], base_range[1], steps)
    jitters = np.linspace(jitter_range[0], jitter_range[1], steps//2)
    
    for b in bases:
        for j in jitters:
            # Try a few seeds for jitter phase
            for seed in [0, 1, 2]:
                u = _generate_geometric_set(n, b, jitter=j, seed=seed)
                if len(u) < 2:
                    continue
                s = _score_from_set(u)
                if s > best_s:
                    best_s = s
                    best_u = u[:]
                    
    return best_u, best_s

def _local_refine(u, max_steps=100, timeout=5.0):
    """
    Simple local refinement: try changing the base slightly or adding/removing elements.
    """
    if not u or len(u) < 2:
        return u, _score_from_set(u)
    
    current_u = u[:]
    current_s = _score_from_set(current_u)
    best_u = current_u[:]
    best_s = current_s
    
    start_time = time.time()
    rng = random.Random(42)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        # Strategy 1: Perturb a random element (not 0)
        if len(current_u) > 2:
            idx = rng.randint(1, len(current_u) - 1)
            old_val = current_u[idx]
            # Small random change
            delta = rng.randint(-max(1, old_val//100), max(1, old_val//100))
            new_val = old_val + delta
            if new_val > 0 and new_val not in current_u:
                new_u = current_u[:]
                new_u[idx] = new_val
                new_u = sorted(set(new_u))
                if len(new_u) == len(current_u):
                    s = _score_from_set(new_u)
                    if s > current_s:
                        current_u = new_u
                        current_s = s
                        if s > best_s:
                            best_s = s
                            best_u = current_u[:]
        else:
            # If small, just try extending
            new_val = current_u[-1] + rng.randint(1, 10)
            new_u = current_u + [new_val]
            s = _score_from_set(new_u)
            if s > current_s:
                current_u = new_u
                current_s = s
                if s > best_s:
                    best_s = s
                    best_u = current_u[:]

    return best_u, best_s

def construct():
    global _DEADLINE
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Phase 1: Explore different sizes and structural families
    # The paper mentions sizes 2003 and 54265. 
    # However, for a 120s budget, we can't compute sum/diff sets for 50k elements easily with O(N^2).
    # N=5000 -> 25M ops. NumPy can handle 25M in ~1-2s. 
    # N=1000 -> 1M ops. Very fast.
    # Let's focus on N in [50, 500] for robust search, and maybe one large one if fast.
    
    sizes_to_try = [20, 50, 100, 200, 500]
    
    # Structural families:
    # 1. Pure Geometric (base 2 to 4)
    # 2. Geometric with Jitter
    # 3. Binary-like (base 2)
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Try pure geometric with a few bases
        for base in [1.8, 2.0, 2.2, 2.5, 3.0]:
            u = _generate_geometric_set(n, base, jitter=0.0)
            s = _score_from_set(u)
            if s > best_s:
                best_s = s
                best_u = u[:]
                
        # Try binary-like
        u = _generate_binary_like_set(n, base=2.0)
        s = _score_from_set(u)
        if s > best_s:
            best_s = s
            best_u = u[:]
            
        # Grid search for this size if time permits
        if time.time() < _DEADLINE - 30.0:
            # Search base in [1.5, 4.0] and jitter in [0, 0.1]
            u_new, s_new = _optimize_params(n, (1.5, 4.0), (0.0, 0.1), steps=30)
            if s_new > best_s:
                best_s = s_new
                best_u = u_new[:]

    # Phase 2: Local refinement on the best candidate
    if time.time() < _DEADLINE - 10.0:
        u_new, s_new = _local_refine(best_u, max_steps=200, timeout=10.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]

    # Phase 3: Try a larger set if we have time, using a very sparse geometric progression
    # Large N requires careful handling. Let's try N=1000 with base=2.
    if time.time() < _DEADLINE - 5.0:
        u_large = _generate_geometric_set(1000, 2.0, jitter=0.0)
        # Check if this is faster than expected
        t0 = time.time()
        s_large = _score_from_set(u_large)
        t1 = time.time()
        if t1 - t0 < 2.0: # If it was fast
            if s_large > best_s:
                best_s = s_large
                best_u = u_large[:]
            # Refine the large set
            if time.time() < _DEADLINE - 2.0:
                u_new, s_new = _local_refine(best_u, max_steps=50, timeout=2.0)
                if s_new > best_s:
                    best_s = s_new
                    best_u = u_new[:]

    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
