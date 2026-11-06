import time
import math
import random
from itertools import combinations

_N = 128
_DEADLINE = 85.0  # Safety margin for 90s kill
_CAND = list(range(1, _N + 1))

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

def _anneal(start_mask, start_violations, max_iters, seed, deadline):
    """
    Simulated Annealing with incremental violation updates.
    Moves: Flip a bit (add/remove).
    Objective: Maximize size, minimize violations.
    Score = size - 100 * violations.
    """
    rng = random.Random(seed)
    cur_mask = start_mask
    cur_viol = start_violations
    cur_size = _popcount(cur_mask)
    cur_score = cur_size - 100 * cur_viol
    
    best_mask = cur_mask
    best_viol = cur_viol
    best_size = cur_size
    best_score = cur_score
    
    T0 = 5.0
    T1 = 0.05
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        T = T0 * (T1 / T0) ** (i / max(1, max_iters))
        
        # Propose move: flip a random bit
        pos = rng.randrange(1, _N + 1)
        bit = 1 << (pos - 1)
        
        if (cur_mask & bit):
            # Remove pos
            new_mask = cur_mask ^ bit
            dv = _delta_violations_remove(cur_mask, pos)
            new_viol = cur_viol - dv
            new_size = cur_size - 1
        else:
            # Add pos
            new_mask = cur_mask ^ bit
            dv = _delta_violations_add(cur_mask, pos)
            new_viol = cur_viol + dv
            new_size = cur_size + 1
            
        new_score = new_size - 100 * new_viol
        delta = new_score - cur_score
        
        if delta >= 0 or rng.random() < math.exp(delta / max(1e-9, T)):
            cur_mask = new_mask
            cur_viol = new_viol
            cur_size = new_size
            cur_score = new_score
            
            if cur_viol == 0 and cur_size > best_size:
                best_mask = cur_mask
                best_viol = cur_viol
                best_size = cur_size
                best_score = cur_score
            elif cur_viol < best_viol:
                # Track best low-violation state too, in case we get stuck
                best_mask = cur_mask
                best_viol = cur_viol
                best_size = cur_size
                best_score = cur_score
            elif cur_viol == best_viol and cur_size > best_size:
                best_mask = cur_mask
                best_viol = cur_viol
                best_size = cur_size
                best_score = cur_score
                
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
            
    # 2. Simulated Annealing from the best greedy solution and a few others
    starts = [best_mask]
    # Add a couple of other greedy results as starts
    for order in orders[:3]:
        mask, viol = _greedy_construct(order)
        mask, viol = _local_search_polish(mask, viol, deadline - 20)
        starts.append(mask)
        
    seed = 100
    for sm in starts:
        if time.time() > deadline - 10:
            break
        sv = _violations_count(sm)
        ss = _popcount(sm)
        
        # Run annealing
        bm, bv, bs = _anneal(sm, sv, max_iters=20000, seed=seed, deadline=deadline - 10)
        seed += 1
        
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
            
        # Final polish on the annealed result
        if time.time() < deadline - 5:
            bm, bv = _local_search_polish(bm, bv, deadline - 5)
            bs = _popcount(bm)
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
