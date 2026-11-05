import time
import random

_DEADLINE = 50.0
_N = 256

def _build_conflict_graph():
    """Build adjacency bitmasks: conflict_mask[a] has bit b set if (a*b)%(a+b)==0."""
    conflict_mask = [0] * (_N + 1)
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    return conflict_mask

def _exact_mis_bnb(conflict_mask, deadline, initial_set=None):
    """
    Branch and bound for Maximum Independent Set using bitset representation.
    Uses greedy coloring for upper bounds.
    Returns the maximum independent set found within deadline.
    """
    start = time.time()
    all_vertices_mask = 0
    for v in range(1, _N + 1):
        all_vertices_mask |= (1 << v)
    
    best_set_mask = [0]
    best_size = [0]
    
    if initial_set:
        for v in initial_set:
            best_set_mask[0] |= (1 << v)
        best_size[0] = len(initial_set)
    
    def _popcount(x):
        return bin(x).count('1')
    
    def _greedy_color_bound(cand_mask):
        """
        Upper bound on independence number of subgraph induced by cand_mask.
        Uses greedy coloring: repeatedly pick a vertex, remove it and its neighbors.
        Each iteration assigns one "color" to a maximal independent set in the remaining graph.
        The number of iterations is an upper bound on the independence number.
        """
        remaining = cand_mask
        count = 0
        while remaining:
            count += 1
            # Pick vertex with highest degree in remaining for tighter bound
            best_v_bit = 0
            best_deg = -1
            temp = remaining
            while temp:
                v_bit = temp & (-temp)
                v = v_bit.bit_length() - 1
                deg = _popcount(remaining & conflict_mask[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v_bit = v_bit
                temp &= ~(v_bit)
            if best_v_bit == 0:
                break
            v = best_v_bit.bit_length() - 1
            remaining &= ~(conflict_mask[v] | best_v_bit)
        return count
    
    def _bnb(current_mask, cand_mask):
        if time.time() - start > deadline:
            return
        
        cur_size = _popcount(current_mask)
        
        if cand_mask == 0:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
                best_set_mask[0] = current_mask
            return
        
        # Upper bound via greedy coloring
        ub = cur_size + _greedy_color_bound(cand_mask)
        if ub <= best_size[0]:
            return
        
        # Select branch vertex: highest degree in cand_mask
        best_v_bit = 0
        best_deg = -1
        temp = cand_mask
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            deg = _popcount(cand_mask & conflict_mask[v])
            if deg > best_deg:
                best_deg = deg
                best_v_bit = v_bit
            temp &= ~(v_bit)
        
        if best_v_bit == 0:
            return
        
        v = best_v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_cand = cand_mask & ~conflict_mask[v] & ~best_v_bit
        _bnb(current_mask | best_v_bit, new_cand)
        
        if time.time() - start > deadline:
            return
        
        # Branch 2: Exclude v
        new_cand2 = cand_mask & ~best_v_bit
        _bnb(current_mask, new_cand2)
    
    _bnb(0, all_vertices_mask)
    
    result = set()
    mask = best_set_mask[0]
    while mask:
        v_bit = mask & (-mask)
        v = v_bit.bit_length() - 1
        result.add(v)
        mask &= ~v_bit
    return result

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict_mask, order):
    """Greedy: add x if it doesn't conflict with any already-chosen element."""
    S_mask = 0
    S = set()
    for x in order:
        if not (S_mask & conflict_mask[x]):
            S_mask |= (1 << x)
            S.add(x)
    return S

def _try_add_to_set(conflict_mask, S):
    """Try to add more elements to S by finding vertices with no conflict."""
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
    added = True
    while added:
        added = False
        for v in range(1, _N + 1):
            bit = 1 << v
            if not (S_mask & bit) and not (S_mask & conflict_mask[v]):
                S_mask |= bit
                S.add(v)
                added = True
    return S

def _local_search(conflict_mask, S, time_limit, rng):
    """
    Local search on an independent set using bitmask operations.
    Moves: add a vertex (if no conflict), best single swap, or multi-swap.
    Objective: maximize |S|.
    """
    S = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
    best_S = set(S)
    best_size = len(S)
    start = time.time()
    all_vertices = list(range(1, _N + 1))
    all_mask = 0
    for v in all_vertices:
        all_mask |= (1 << v)

    def _can_add(S_mask, v):
        return not (S_mask & conflict_mask[v])

    def _best_single_swap(S_mask, S):
        """Find the best swap: remove u, add v such that v is compatible with S\\{u}."""
        best_gain = 0
        best_u = None
        best_v = None
        for u in S:
            new_mask = S_mask & ~(1 << u)
            # Find all v not in S that can be added to new_mask
            for v in all_vertices:
                if v in S:
                    continue
                if _can_add(new_mask, v):
                    # This is a valid swap (size stays same), but we want to check
                    # if after this swap we can add more
                    # For now, just track valid swaps
                    if best_gain == 0:
                        best_gain = 1
                        best_u = u
                        best_v = v
                    break
        return best_u, best_v

    while time.time() - start < time_limit:
        improved = False

        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in all_vertices if not (S_mask & (1 << v)) and _can_add(S_mask, v)]
        if candidates_to_add:
            v = rng.choice(candidates_to_add)
            S_mask |= (1 << v)
            S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # Try best single swap: find u in S and v not in S such that v fits in S\{u}
        # and then we can potentially add more after
        if not improved and len(S) > 0:
            found_swap = False
            # Systematically try all u, and for each find if any v works
            for u in list(S):
                new_mask = S_mask & ~(1 << u)
                # Try to add as many as possible after removing u
                new_S = S - {u}
                new_S_mask = new_mask
                added_count = 0
                for v in all_vertices:
                    if v in new_S:
                        continue
                    if _can_add(new_S_mask, v):
                        new_S_mask |= (1 << v)
                        new_S.add(v)
                        added_count += 1
                if len(new_S) > len(S):
                    S = new_S
                    S_mask = new_S_mask
                    improved = True
                    found_swap = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
                elif len(new_S) == len(S) and not found_swap:
                    # Try a random valid swap to explore
                    pass

            if not found_swap:
                # Try random swaps for exploration
                for _ in range(50):
                    u = rng.choice(list(S))
                    new_mask = S_mask & ~(1 << u)
                    v = rng.choice(all_vertices)
                    if v in S:
                        continue
                    if _can_add(new_mask, v):
                        S_mask = new_mask | (1 << v)
                        S.discard(u)
                        S.add(v)
                        improved = True
                        break

        # If no improvement, do a perturbation: remove 1-2 random elements, refill greedily
        if not improved:
            removed = rng.randint(1, 2)
            for _ in range(removed):
                if S:
                    u = rng.choice(list(S))
                    S_mask &= ~(1 << u)
                    S.discard(u)
            # Greedily refill
            for v in all_vertices:
                if v not in S and _can_add(S_mask, v):
                    S_mask |= (1 << v)
                    S.add(v)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    return best_S

def _simulated_annealing(conflict_mask, S, time_limit, rng):
    """
    Simulated annealing local search on an independent set using bitmasks.
    Allows temporary removal of elements to escape local optima.
    """
    S = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << v)
    best_S = set(S)
    best_size = len(S)
    start = time.time()
    all_vertices = list(range(1, _N + 1))
    
    # Temperature schedule - slower cooling
    T = 5.0
    cooling = 0.9995
    
    def _can_add(S_mask, v):
        return not (S_mask & conflict_mask[v])

    while time.time() - start < time_limit:
        T *= cooling
        if T < 0.005:
            break
        
        improved = False
        
        # Try to add any vertex that has no conflict with S
        candidates_to_add = [v for v in all_vertices if not (S_mask & (1 << v)) and _can_add(S_mask, v)]
        if candidates_to_add:
            v = rng.choice(candidates_to_add)
            S_mask |= (1 << v)
            S.add(v)
            improved = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
        
        # Try swaps: remove u, add v
        if not improved and len(S) > 0:
            for _ in range(30):
                u = rng.choice(list(S))
                new_mask = S_mask & ~(1 << u)
                v = rng.choice(all_vertices)
                if v in S:
                    continue
                if _can_add(new_mask, v):
                    S_mask = new_mask | (1 << v)
                    S.discard(u)
                    S.add(v)
                    improved = True
                    break
        
        # SA move: accept a worsening move with probability based on temperature
        if not improved and len(S) > 3:
            # Try removing a random element without immediate refill
            if rng.random() < T / 5.0:
                u = rng.choice(list(S))
                S_mask &= ~(1 << u)
                S.discard(u)
                # Try to greedily refill
                for v in all_vertices:
                    if v not in S and _can_add(S_mask, v):
                        S_mask |= (1 << v)
                        S.add(v)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        # Aggressive perturbation at low temperature
        if not improved and T < 1.0:
            for _ in range(rng.randint(1, 3)):
                if S:
                    u = rng.choice(list(S))
                    S_mask &= ~(1 << u)
                    S.discard(u)
            # Greedily refill
            for v in all_vertices:
                if v not in S and _can_add(S_mask, v):
                    S_mask |= (1 << v)
                    S.add(v)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    return best_S

def _search(conflict_mask, total_time):
    """Run multiple rounds of greedy init + local search / simulated annealing."""
    start = time.time()
    best_S = set()
    best_size = 0
    rng = random.Random(42)
    
    # Precompute degrees
    degrees = [bin(conflict_mask[v]).count('1') for v in range(_N + 1)]

    while time.time() - start < total_time:
        # Phase 1: Multiple greedy initializations
        S_curr = set()
        
        # Randomized greedy
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        S_init = _greedy_build(conflict_mask, order)
        
        # Degree-ordered greedy (low degree first)
        deg_order = sorted(range(1, _N + 1), key=lambda v: degrees[v])
        S_deg = _greedy_build(conflict_mask, deg_order)
        
        # Reverse degree greedy
        rdeg_order = sorted(range(1, _N + 1), key=lambda v: -degrees[v])
        S_rdeg = _greedy_build(conflict_mask, rdeg_order)
        
        # Pick the best initialization
        candidates = [S_init, S_deg, S_rdeg]
        S_curr = max(candidates, key=len)
        
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
        
        # Phase 2: Local search or simulated annealing
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break
        
        ls_time = min(remaining * 0.25, 3.0)
        
        # Alternate between local search and simulated annealing
        if rng.random() < 0.5:
            S_ls = _local_search(conflict_mask, S_curr, ls_time, rng)
        else:
            S_ls = _simulated_annealing(conflict_mask, S_curr, ls_time, rng)
        
        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = set(S_ls)

    return best_S

def construct():
    start = time.time()
    conflict_mask = _build_conflict_graph()
    
    # Quick baseline: greedy with low-degree ordering
    degrees = [bin(conflict_mask[v]).count('1') for v in range(_N + 1)]
    deg_order = sorted(range(1, _N + 1), key=lambda v: degrees[v])
    S_baseline = _greedy_build(conflict_mask, deg_order)
    S_baseline = _try_add_to_set(conflict_mask, S_baseline)
    
    # Allocate time with safety margin
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 5.0
    if remaining < 5:
        remaining = 5
    
    # Run branch-and-bound with the baseline as initial solution
    best_S = _exact_mis_bnb(conflict_mask, remaining, initial_set=S_baseline)
    
    # Final greedy fill
    best_S = _try_add_to_set(conflict_mask, best_S)
    
    # Ensure baseline is at least as good
    if len(best_S) < len(S_baseline):
        best_S = S_baseline
    
    result = sorted(best_S)
    if not result:
        return [1]
    return result
