import itertools
import time
import random

def _is_valid_tournament(n, out_masks):
    """
    Check if every 4-set has a common dominator.
    out_masks[v] is a bitmask of vertices u such that v -> u.
    In[v] is derived: u is in In[v] if v is in out_masks[u].
    """
    # Precompute In masks
    In = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if v != u and (out_masks[v] >> u) & 1:
                mask |= (1 << v)
        In[u] = mask
        
    for S in itertools.combinations(range(n), 4):
        inter = In[S[0]]
        for u in S[1:]:
            inter &= In[u]
            if inter == 0:
                return False
    return True

def _local_search(n, time_limit=2.0):
    """
    Attempt to find a valid tournament on n vertices using local search.
    Returns orientation list if found, else None.
    """
    start_time = time.time()
    random.seed(42) # Deterministic for reproducibility in evaluation?
    
    for _ in range(10): # 10 restarts
        # Initialize random tournament
        out_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    out_masks[i] |= (1 << j)
                else:
                    out_masks[j] |= (1 << i)
        
        # Precompute In
        In = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if v != u and (out_masks[v] >> u) & 1:
                    mask |= (1 << v)
            In[u] = mask
            
        # Count bad 4-sets
        def count_bad():
            bad = 0
            for S in itertools.combinations(range(n), 4):
                inter = In[S[0]]
                for u in S[1:]:
                    inter &= In[u]
                    if inter == 0:
                        bad += 1
                        break
            return bad

        current_bad = count_bad()
        if current_bad == 0:
            # Construct orientation
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (out_masks[i] >> j) & 1:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return orientation

        # Local search
        while time.time() - start_time < time_limit:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip i <-> j
            # Currently: if i -> j, out_masks[i] has j.
            # We swap the direction.
            i_to_j = (out_masks[i] >> j) & 1
            if i_to_j:
                # i -> j becomes j -> i
                out_masks[i] &= ~(1 << j)
                out_masks[j] |= (1 << i)
                # Update In
                # In[i] loses j? No, In[i] is who points to i.
                # If i->j was true, j did NOT point to i.
                # Now j->i is true, so j points to i. In[i] gains j.
                # If i->j was true, i points to j. In[j] did NOT contain i.
                # Now j->i, so i does NOT point to j. In[j] loses i?
                # Wait. In[j] contains u if u->j.
                # Before: i->j. So i in In[j].
                # After: j->i. So i NOT in In[j].
                # So In[j] loses i.
                # In[i]: Before j NOT in In[i]. After j in In[i].
                In[i] |= (1 << j)
                In[j] &= ~(1 << i)
            else:
                # j -> i becomes i -> j
                out_masks[j] &= ~(1 << i)
                out_masks[i] |= (1 << j)
                # In[i] loses j. In[j] gains i.
                In[i] &= ~(1 << j)
                In[j] |= (1 << i)

            # Recalculate bad count (expensive, but n is small)
            # Optimization: Only check sets containing i or j?
            # For simplicity and safety, recalculate all if n is small enough.
            # For n=31, 31465 sets. Fast.
            # For n=40, 91390 sets. Still fast.
            # For n=50, 230300 sets. ~0.1s.
            # So we can do ~100 moves per second?
            # 2s limit -> ~200 moves.
            
            new_bad = count_bad()
            
            if new_bad < current_bad:
                current_bad = new_bad
                if current_bad == 0:
                    orientation = []
                    for k in range(n):
                        for l in range(k + 1, n):
                            if (out_masks[k] >> l) & 1:
                                orientation.append(0)
                            else:
                                orientation.append(1)
                    return orientation
            else:
                # Revert
                if i_to_j:
                    out_masks[i] |= (1 << j)
                    out_masks[j] &= ~(1 << i)
                    In[i] &= ~(1 << j)
                    In[j] |= (1 << i)
                else:
                    out_masks[j] |= (1 << i)
                    out_masks[i] &= ~(1 << j)
                    In[i] |= (1 << j)
                    In[j] &= ~(1 << i)

    return None

def construct():
    start_time = time.time()
    
    # 1. Check small Paley primes
    candidates = [31, 43, 47, 59]
    for p in candidates:
        if time.time() - start_time > 5:
            break
        if p % 4 != 3:
            continue
        squares = {x * x % p for x in range(1, p)}
        In = [0] * p
        for u in range(p):
            mask = 0
            for v in range(p):
                if v == u:
                    continue
                if (u - v) % p in squares:
                    mask |= (1 << v)
            In[u] = mask
            
        valid = True
        for S in itertools.combinations(range(p), 4):
            inter = In[S[0]]
            for u in S[1:]:
                inter &= In[u]
                if inter == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}

    # 2. Try Local Search for n < 67
    # Limit total time for this phase to 20 seconds
    ls_start = time.time()
    for n in range(31, 50): # Try up to 49
        if time.time() - ls_start > 20:
            break
        res = _local_search(n, time_limit=1.0)
        if res:
            return {'n': n, 'orientation': res}

    # 3. Fallback to Paley 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
