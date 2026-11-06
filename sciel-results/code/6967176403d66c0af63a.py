import time
import math
import random

_N = 128
_DEADLINE = 85.0  # Safety margin for 90s kill

# Global precomputed data
_GCD = None
_BAD_MASK = None  # BAD_MASK[u][v] is a bitmask of w such that {u,v,w} is a bad triple
_NEIGHBOR_MASK = None  # NEIGHBOR_MASK[u] is bitmask of v such that u,v are in some bad triple

def _ensure_pre():
    global _GCD, _BAD_MASK, _NEIGHBOR_MASK
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
    _NEIGHBOR_MASK = [0] * (_N + 1)
    
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
            if mask:
                _NEIGHBOR_MASK[u] |= (1 << (v - 1))
                _NEIGHBOR_MASK[v] |= (1 << (u - 1))

def _popcount(x):
    return x.bit_count()

def _violations_count(mask):
    """
    Count the number of bad triples in the set represented by mask.
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
            total += _popcount(mask & _BAD_MASK[u][v])
    
    return total // 3

def _delta_violations_add(mask, x):
    """
    Calculate the increase in violation count if x is added to the set represented by mask.
    """
    if (mask >> (x - 1)) & 1:
        return 0
    
    delta = 0
    for u in range(1, _N + 1):
        if (mask >> (u - 1)) & 1:
            delta += _popcount(mask & _BAD_MASK[x][u])
            
    return delta

def _delta_violations_remove(mask, x):
    """
    Calculate the decrease in violation count if x is removed from the set represented by mask.
    """
    if not ((mask >> (x - 1)) & 1):
        return 0
    
    delta = 0
    mask_without_x = mask & ~(1 << (x - 1))
    
    for u in range(1, _N + 1):
        if (mask_without_x >> (u - 1)) & 1:
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
        conflict_count = [0] * (_N + 1)
        elems = [i for i in range(1, _N + 1) if (cur_mask >> (i - 1)) & 1]
        n = len(elems)
        
        if n < 3:
            break
            
        for i in range(n):
            u = elems[i]
            for j in range(i + 1, n):
                v = elems[j]
                cnt = _popcount(cur_mask & _BAD_MASK[u][v])
                if cnt > 0:
                    conflict_count[u] += cnt
                    conflict_count[v] += cnt
        
        max_conflict = 0
        remove_elem = -1
        for i in range(1, _N + 1):
            if (cur_mask >> (i - 1)) & 1 and conflict_count[i] > max_conflict:
                max_conflict = conflict_count[i]
                remove_elem = i
        
        if remove_elem == -1:
            break
            
        cur_mask &= ~(1 << (remove_elem - 1))
        cur_viol = _violations_count(cur_mask)
        
    return cur_mask, cur_viol

def _branch_and_bound(deadline):
    """
    Branch and bound to find maximum independent set in the 3-uniform hypergraph.
    This is a heuristic version using a greedy upper bound based on the conflict graph.
    """
    best_mask = 0
    best_size = 0
    
    # We will try to build a set from scratch using B&B.
    # However, B&B on 128 nodes is too large for exhaustive search.
    # We will use a heuristic B&B that prunes based on a greedy packing bound.
    
    # Let's use a different approach: 
    # 1. Start with a valid set from greedy.
    # 2. Try to improve it by swapping elements.
    
    # Actually, the prompt suggests "maximum independent set in a 3-uniform hypergraph".
    # A common heuristic is to use the conflict graph (2-uniform projection).
    # If we treat the conflict graph (where u~v if they are in a bad triple) as a graph,
    # an independent set in this graph is NOT necessarily a valid set for the 3-uniform problem.
    # But a valid set for the 3-uniform problem IS an independent set in the conflict graph.
    # So we can use the maximum independent set of the conflict graph as an upper bound?
    # No, the MIS of the conflict graph is an upper bound on the size of the valid set.
    
    # Let's implement a simple MIS solver for the conflict graph to get a bound.
    # But we don't need the exact MIS, just a good bound.
    
    # Alternative: Use the current best valid set and try to add elements using a
    # more sophisticated search than simple greedy.
    
    # Let's try a randomized local search that allows temporary violations.
    
    return best_mask, best_size

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
        sorted(range(1, _N + 1), key=lambda x: x.bit_count()),
        sorted(range(1, _N + 1), key=lambda x: -x.bit_count()),
    ]
    
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
    
    rng_init = random.Random(42)
    for _ in range(5):
        o = list(range(1, _N + 1))
        rng_init.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 20:
            break
        mask, viol = _greedy_construct(order)
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
        rng = random.Random(_SEED)
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            sample_size = random.randint(40, 80)
            sample = rng.sample(range(1, _N + 1), sample_size)
            mask = 0
            for x in sample:
                mask |= (1 << (x - 1))
            
            cleaned_mask, cleaned_viol = _remove_conflicts_greedy(mask, deadline - 15)
            
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
        
    # 3. Simulated Annealing
    if time.time() < deadline - 5:
        max_iters = 5000
        bm, bv, bs = _anneal(best_mask, best_viol, max_iters, 42, deadline - 5)
        
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

    # 4. Final aggressive polish on the global best
    if time.time() < deadline - 2:
        best_mask, best_viol = _local_search_polish(best_mask, best_viol, deadline - 2)
        best_size = _popcount(best_mask)
        
    # 5. If still have violations, try more aggressive removal
    if best_viol > 0 and time.time() < deadline - 1:
        best_mask, best_viol = _remove_conflicts_greedy(best_mask, deadline - 1)
        best_size = _popcount(best_mask)
    
    # Convert mask to list
    result = [i for i in range(1, _N + 1) if (best_mask >> (i - 1)) & 1]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [2]
        
    return result

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

_SEED = 42
