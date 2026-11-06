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

def _anneal(start_mask, deadline):
    """
    Simulated annealing on the mask.
    Objective: maximize size - penalty * violations.
    Moves: add a random element, remove a random element, swap.
    """
    rng = random.Random(_SEED + 1000)
    
    cur_mask = start_mask
    cur_viol = _violations_count(cur_mask)
    cur_size = _popcount(cur_mask)
    
    best_mask = cur_mask
    best_size = cur_size
    best_viol = cur_viol
    
    # Penalty for violations - make it large enough to discourage them
    # but allow temporary violations to escape local optima
    penalty = 100
    
    def objective(mask, viol, size):
        return size - penalty * viol
    
    cur_obj = objective(cur_mask, cur_viol, cur_size)
    best_obj = cur_obj
    
    # Cooling schedule
    T_start = 50.0
    T_end = 0.1
    steps = 200000
    
    # Precompute list of all elements for random selection
    all_elems = list(range(1, _N + 1))
    
    step = 0
    while step < steps and time.time() < deadline:
        step += 1
        
        # Linear cooling
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Choose a move
        move_type = rng.random()
        new_mask = cur_mask
        new_viol = cur_viol
        new_size = cur_size
        
        if move_type < 0.35:
            # Add a random element not in set
            not_in = [x for x in all_elems if not (cur_mask >> (x - 1)) & 1]
            if not_in:
                x = rng.choice(not_in)
                new_mask = cur_mask | (1 << (x - 1))
                new_size = cur_size + 1
                new_viol = cur_viol + _delta_violations_add(cur_mask, x)
        elif move_type < 0.70:
            # Remove a random element from set
            in_set = [x for x in all_elems if (cur_mask >> (x - 1)) & 1]
            if in_set:
                x = rng.choice(in_set)
                new_mask = cur_mask & ~(1 << (x - 1))
                new_size = cur_size - 1
                new_viol = cur_viol - _delta_violations_remove(cur_mask, x)
        else:
            # Swap: remove one, add one
            in_set = [x for x in all_elems if (cur_mask >> (x - 1)) & 1]
            not_in = [x for x in all_elems if not (cur_mask >> (x - 1)) & 1]
            if in_set and not_in:
                rem = rng.choice(in_set)
                add = rng.choice(not_in)
                # Remove rem first
                mask_without_rem = cur_mask & ~(1 << (rem - 1))
                viol_after_rem = cur_viol - _delta_violations_remove(cur_mask, rem)
                # Add add
                new_mask = mask_without_rem | (1 << (add - 1))
                new_size = cur_size
                new_viol = viol_after_rem + _delta_violations_add(mask_without_rem, add)
        
        new_obj = objective(new_mask, new_viol, new_size)
        delta_obj = new_obj - cur_obj
        
        # Accept or reject
        if delta_obj >= 0:
            accept = True
        else:
            # Boltzmann probability
            if T > 0.001:
                prob = math.exp(delta_obj / T)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            cur_mask = new_mask
            cur_viol = new_viol
            cur_size = new_size
            cur_obj = new_obj
            
            if cur_obj > best_obj:
                best_obj = cur_obj
                best_mask = cur_mask
                best_size = cur_size
                best_viol = cur_viol
    
    # Final polish: try to add any element that doesn't increase violations
    best_mask, best_viol = _local_search_polish(best_mask, best_viol, deadline)
    best_size = _popcount(best_mask)
    
    return best_mask, best_viol, best_size

def _structural_initial():
    """
    Build a good initial solution using structural insights.
    Group numbers by their prime factorization patterns.
    Key idea: if we pick at most 2 numbers from each "gcd-equivalence class",
    we avoid many bad triples.
    """
    # Strategy: for each d in 1..128, consider multiples of d.
    # Numbers a, b, c where gcd(a,b)=gcd(a,c)=gcd(b,c)=d form a bad triple.
    # So for each d, we can pick at most 2 numbers from the set of multiples of d
    # that have gcd exactly d with each other.
    
    # Simpler approach: greedy with smart ordering
    # Order by: numbers with fewer divisors first (primes, prime powers)
    # because they're less likely to form bad triples
    
    def divisor_count(n):
        cnt = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                cnt += 1
                if i != n // i:
                    cnt += 1
        return cnt
    
    # Sort by number of divisors (ascending), then by value
    ordered = sorted(range(1, _N + 1), key=lambda x: (divisor_count(x), x))
    
    mask, viol = _greedy_construct(ordered)
    mask, viol = _local_search_polish(mask, viol, time.time() + 5)
    
    return mask, viol

def construct():
    _ensure_pre()
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Phase 1: Get a good initial solution using structural insights
    init_mask, init_viol = _structural_initial()
    init_size = _popcount(init_mask)
    
    best_mask = init_mask
    best_size = init_size
    best_viol = init_viol
    
    # Phase 2: Simulated annealing from the initial solution
    if time.time() < deadline - 5:
        anneal_deadline = deadline - 3
        am, av, asz = _anneal(init_mask, anneal_deadline)
        
        if av == 0 and asz > best_size:
            best_size = asz
            best_mask = am
            best_viol = av
        elif av < best_viol:
            best_viol = av
            best_size = asz
            best_mask = am
        elif av == best_viol and asz > best_size:
            best_size = asz
            best_mask = am
            best_viol = av
    
    # Phase 3: If we have a valid solution, try to improve with more annealing
    # from slightly perturbed versions
    if best_viol == 0 and time.time() < deadline - 10:
        rng = random.Random(_SEED + 2000)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            # Perturb: remove a few random elements, then re-add
            perturbed = best_mask
            elems_in = [x for x in range(1, _N + 1) if (perturbed >> (x - 1)) & 1]
            num_remove = max(1, len(elems_in) // 10)
            to_remove = rng.sample(elems_in, min(num_remove, len(elems_in)))
            for x in to_remove:
                perturbed &= ~(1 << (x - 1))
            
            # Polish
            perturbed, _ = _local_search_polish(perturbed, 0, deadline - 5)
            ps = _popcount(perturbed)
            
            if ps > best_size:
                best_size = ps
                best_mask = perturbed
    
    # Phase 4: Final polish
    if time.time() < deadline - 1:
        best_mask, best_viol = _local_search_polish(best_mask, best_viol, deadline - 1)
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
