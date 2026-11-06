import time
import math
import random
from itertools import combinations

_N = 128
_DEADLINE = 85.0  # Safety margin for 90s kill
_CAND = list(range(1, _N + 1))
_SEED = 42

# Global precomputed data
_GCD = None
_BAD_MASK = None  # BAD_MASK[u][v] is a bitmask of w such that {u,v,w} is a bad triple

def _ensure_pre():
    global _GCD, _BAD_MASK
    if _GCD is not None:
        return
    
    # 1. GCD Matrix
    _GCD = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            g = math.gcd(i, j)
            _GCD[i][j] = g
            _GCD[j][i] = g

    # 2. Bad Triple Masks
    # BAD_MASK[u][v] has bit w set if gcd(u,v) == gcd(u,w) == gcd(v,w) and w != u, w != v
    _BAD_MASK = [[0] * (_N + 1) for _ in range(_N + 1)]
    
    for u in range(1, _N + 1):
        for v in range(u + 1, _N + 1):
            g_uv = _GCD[u][v]
            mask = 0
            for w in range(1, _N + 1):
                if w == u or w == v:
                    continue
                if _GCD[u][w] == g_uv and _GCD[v][w] == g_uv:
                    mask |= (1 << (w - 1))
            _BAD_MASK[u][v] = mask
            _BAD_MASK[v][u] = mask

def _popcount(x):
    return x.bit_count()

def _violations_count(mask):
    """
    Count the number of bad triples in the set represented by mask.
    A bad triple {a,b,c} is counted once for each of its 3 pairs.
    So total count = (1/3) * sum_{u<v in S} popcount(S & BAD_MASK[u][v])
    """
    s = [i for i in range(1, _N + 1) if (mask >> (i - 1)) & 1]
    n = len(s)
    if n < 3:
        return 0
    
    total = 0
    for i in range(n):
        u = s[i]
        for j in range(i + 1, n):
            v = s[j]
            # Count w in S such that {u,v,w} is bad
            # BAD_MASK[u][v] contains all w (including potentially u,v but we excluded them in precomp)
            # We just need to intersect with S
            total += _popcount(mask & _BAD_MASK[u][v])
    
    return total // 3

def _delta_violations_add(mask, x):
    """
    Calculate the increase in violation count if x is added to the set represented by mask.
    x must not be in mask.
    New violations are triples {x, u, v} where u, v are in mask.
    For each u in mask, we count v in mask such that {x, u, v} is bad.
    This is exactly popcount(mask & BAD_MASK[x][u]) for each u in mask.
    """
    if (mask >> (x - 1)) & 1:
        return 0 # Already there, or invalid move
    
    delta = 0
    # Iterate over all u in mask
    # To iterate efficiently, we can just loop 1..N and check bit
    # Or extract bits. N=128 is small.
    for u in range(1, _N + 1):
        if (mask >> (u - 1)) & 1:
            # Count v in mask such that {x, u, v} is bad
            # BAD_MASK[x][u] contains candidates for v
            # We need v to be in mask
            # Note: BAD_MASK[x][u] might contain x or u? 
            # In precomp, we excluded w==u and w==v. So BAD_MASK[x][u] does not contain x or u.
            # So we just intersect with mask.
            delta += _popcount(mask & _BAD_MASK[x][u])
            
    return delta

def _delta_violations_remove(mask, x):
    """
    Calculate the decrease in violation count if x is removed from the set represented by mask.
    x must be in mask.
    Violations removed are triples {x, u, v} where u, v are in mask\{x}.
    """
    if not ((mask >> (x - 1)) & 1):
        return 0
    
    delta = 0
    mask_without_x = mask & ~(1 << (x - 1))
    
    for u in range(1, _N + 1):
        if (mask_without_x >> (u - 1)) & 1:
            # Count v in mask_without_x such that {x, u, v} is bad
            delta += _popcount(mask_without_x & _BAD_MASK[x][u])
            
    return delta

def _greedy_construct(order):
    """
    Greedily add elements in the given order if they don't increase violations.
    """
    mask = 0
    violations = 0
    for x in order:
        dv = _delta_violations_add(mask, x)
        if dv == 0:
            mask |= (1 << (x - 1))
            violations += dv
    return mask, violations

def _local_search_polish(mask, violations, deadline):
    """
    Try to add any element that doesn't increase violations.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, _N + 1):
            if not ((mask >> (x - 1)) & 1):
                dv = _delta_violations_add(mask, x)
                if dv == 0:
                    mask |= (1 << (x - 1))
                    violations += dv
                    changed = True
    return mask, violations

def _remove_conflicts_greedy(mask, deadline):
    """
    Remove elements to eliminate all conflicts using a greedy strategy.
    Prioritize removing elements that are in the most bad triples.
    """
    cur_mask = mask
    cur_viol = _violations_count(cur_mask)
    
    while cur_viol > 0 and time.time() < deadline:
        # Find the element involved in the most bad triples
        conflict_count = [0] * (_N + 1)
        
        # Extract elements in cur_mask
        elems = [i for i in range(1, _N + 1) if (cur_mask >> (i - 1)) & 1]
        n = len(elems)
        
        if n < 3:
            break
            
        for i in range(n):
            u = elems[i]
            for j in range(i + 1, n):
                v = elems[j]
                # Count w in cur_mask such that {u,v,w} is bad
                cnt = _popcount(cur_mask & _BAD_MASK[u][v])
                if cnt > 0:
                    # This pair is in cnt bad triples
                    conflict_count[u] += cnt
                    conflict_count[v] += cnt
        
        # Find element with maximum conflict count
        max_conflict = 0
        remove_elem = -1
        for i in range(1, _N + 1):
            if (cur_mask >> (i - 1)) & 1 and conflict_count[i] > max_conflict:
                max_conflict = conflict_count[i]
                remove_elem = i
        
        if remove_elem == -1:
            break
            
        # Remove the element
        cur_mask &= ~(1 << (remove_elem - 1))
        cur_viol = _violations_count(cur_mask)
        
    return cur_mask, cur_viol

def _randomized_restarts(start_mask, num_restarts, deadline):
    """
    Try multiple randomized approaches:
    1. Start from different subsets
    2. Use different removal strategies
    """
    best_mask = start_mask
    best_size = _popcount(start_mask)
    best_viol = _violations_count(start_mask)
    
    rng = random.Random(_SEED)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random starting mask by sampling
        sample_size = random.randint(40, 80)
        sample = rng.sample(range(1, _N + 1), sample_size)
        mask = 0
        for x in sample:
            mask |= (1 << (x - 1))
        
        # Remove conflicts
        cleaned_mask, cleaned_viol = _remove_conflicts_greedy(mask, deadline)
        
        size = _popcount(cleaned_mask)
        if cleaned_viol == 0 and size > best_size:
            best_size = size
            best_mask = cleaned_mask
            best_viol = 0
        elif cleaned_viol < best_viol:
            best_viol = cleaned_viol
            best_size = size
            best_mask = cleaned_mask
        elif cleaned_viol == best_viol and size > best_size:
            best_size = size
            best_mask = cleaned_mask
    
    return best_mask, best_viol, best_size

def construct():
    _ensure_pre()
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_mask = 0
    best_size = 0
    best_viol = float('inf')
    
    # 1. Greedy constructions with different orderings
    orders = [
        list(range(1, _N + 1)),
        list(range(_N, 0, -1)),
        sorted(range(1, _N + 1), key=lambda x: x.bit_count()), # Fewer bits first?
        sorted(range(1, _N + 1), key=lambda x: -x.bit_count()),
    ]
    
    # Prime powers first
    def is_prime_power(n):
        if n < 2: return False
        v = n
        p = 2
        found = False
        while p * p <= v:
            if v % p == 0:
                if found: return False
                found = True
                while v % p == 0:
                    v //= p
            p += 1
        return True
        
    prime_powers = [x for x in range(1, _N + 1) if is_prime_power(x)]
    others = [x for x in range(1, _N + 1) if x not in prime_powers]
    orders.append(prime_powers + others)
    orders.append(others + prime_powers)
    
    # Random orders
    rng_init = random.Random(42)
    for _ in range(5):
        o = list(range(1, _N + 1))
        rng_init.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 20:
            break
        mask, viol = _greedy_construct(order)
        # Polish
        mask, viol = _local_search_polish(mask, viol, deadline - 20)
        
        size = _popcount(mask)
        if viol == 0 and size > best_size:
            best_size = size
            best_mask = mask
            best_viol = viol
        elif viol < best_viol:
            best_viol = viol
            best_size = size
            best_mask = mask
        elif viol == best_viol and size > best_size:
            best_size = size
            best_mask = mask
            best_viol = viol
            
    # 2. Randomized restarts with conflict-based removal
    if time.time() < deadline - 15:
        num_restarts = 50
        bm, bv, bs = _randomized_restarts(best_mask, num_restarts, deadline - 15)
        
        if bv == 0 and bs > best_size:
            best_size = bs
            best_mask = bm
            best_viol = bv
        elif bv < best_viol:
            best_viol = bv
            best_size = bs
            best_mask = bm
        elif bv == best_viol and bs > best_size:
            best_size = bs
            best_mask = bm
            best_viol = bv

    # 3. Final aggressive polish on the global best
    if time.time() < deadline - 2:
        best_mask, best_viol = _local_search_polish(best_mask, best_viol, deadline - 2)
        best_size = _popcount(best_mask)
        
    # 4. If still have violations, try more aggressive removal
    if best_viol > 0 and time.time() < deadline - 1:
        best_mask, best_viol = _remove_conflicts_greedy(best_mask, deadline - 1)
        best_size = _popcount(best_mask)
    
    # Convert mask to list
    result = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [2]
        
    return result

# Ensure the function is named exactly as requested
# The prompt says: "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.

if __name__ == "__main__":
    # Self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    P = _GCD
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            for k in range(j+1, len(res)):
                a, b, c = res[i], res[j], res[k]
                if P[a][b] == P[a][c] == P[b][c]:
                    valid = False
    print(f"Valid: {valid}")
