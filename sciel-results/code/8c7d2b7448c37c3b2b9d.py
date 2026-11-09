"""Rewritten program: simulated annealing with efficient conflict tracking for
sum-free subset maximization, replacing the greedy+branch-and-bound approach."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 145.0
_START = time.time()


def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)


def _build_conflict_structure(vals):
    """Build conflict structure: for each vertex, which other vertices conflict with it."""
    n = len(vals)
    lookup = {v: i for i, v in enumerate(vals)}
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = vals[i] + vals[j]
            if s in lookup:
                k = lookup[s]
                conflicts[i].add(j)
                conflicts[j].add(i)
    # Also handle a=a cases: if 2*vals[i] is in vals
    for i in range(n):
        s = 2 * vals[i]
        if s in lookup and s != vals[i]:
            k = lookup[s]
            conflicts[i].add(k)
            conflicts[k].add(i)
    return conflicts


def _is_sum_free(vals, subset_indices):
    """Verify no a+b=c with a,b,c in subset (a=b allowed)."""
    present = set(subset_indices)
    val_set = set(vals[i] for i in subset_indices)
    for a in val_set:
        for b in val_set:
            if a + b in val_set:
                return False
    return True


def _sa_solve(vals, conflicts, deadline, rng):
    """Simulated annealing to find large sum-free subset."""
    n = len(vals)
    if n == 0:
        return set()
    
    # Initialize: start with a greedy solution
    remaining = set(range(n))
    chosen = set()
    while remaining:
        # Pick vertex with fewest conflicts among remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            d = len(conflicts[v] & remaining)
            if d < best_deg:
                best_deg = d
                best_v = v
        chosen.add(best_v)
        remaining.discard(best_v)
        for u in conflicts[best_v]:
            remaining.discard(u)
    
    # Track conflicts for fast evaluation
    # For each vertex not in chosen, count how many of its conflicts are in chosen
    conflict_count = {}
    for v in range(n):
        if v not in chosen:
            conflict_count[v] = len(conflicts[v] & chosen)
    
    def can_add(v):
        """Check if vertex v can be added to chosen."""
        if v in chosen:
            return False
        # v can be added if it has no conflicts with chosen vertices
        return conflict_count.get(v, len(conflicts[v] & chosen)) == 0
    
    def try_add(v):
        """Add vertex v to chosen, update conflict counts."""
        chosen.add(v)
        for u in conflicts[v]:
            if u not in chosen:
                conflict_count[u] = conflict_count.get(u, 0) + 1
        if v in conflict_count:
            del conflict_count[v]
    
    def try_remove(v):
        """Remove vertex v from chosen, update conflict counts."""
        chosen.discard(v)
        for u in conflicts[v]:
            if u not in chosen:
                conflict_count[u] = max(0, conflict_count.get(u, 0) - 1)
        if v not in chosen:
            conflict_count[v] = len(conflicts[v] & chosen)
    
    best_chosen = set(chosen)
    best_size = len(chosen)
    
    # Simulated annealing parameters
    t_start = 10.0
    t_end = 0.01
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, int((deadline - time.time()) * 50)))
    
    t = t_start
    iteration = 0
    
    while time.time() < deadline - 1.0:
        if t < t_end:
            break
        
        # Choose a random move: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to add a random vertex
            candidates = [v for v in range(n) if v not in chosen]
            if not candidates:
                t *= cooling_rate
                continue
            v = rng.choice(candidates)
            if can_add(v):
                try_add(v)
                new_size = len(chosen)
                if new_size > best_size:
                    best_size = new_size
                    best_chosen = set(chosen)
            else:
                # Can't add, so skip
                pass
        elif move_type < 0.6:
            # Try to remove a random vertex (to allow swaps later)
            if chosen:
                v = rng.choice(list(chosen))
                # Only remove if it allows adding something better
                # For now, just track this as a potential move
                pass
        else:
            # Swap: remove one, try to add one or more
            if chosen:
                v_out = rng.choice(list(chosen))
                # Save state
                saved_chosen = set(chosen)
                saved_conflict_count = dict(conflict_count)
                
                try_remove(v_out)
                
                # Try to add vertices that were previously blocked
                added = []
                for v in range(n):
                    if v not in chosen and v != v_out:
                        if can_add(v):
                            try_add(v)
                            added.append(v)
                            if len(added) >= 1:
                                break
                
                new_size = len(chosen)
                delta = new_size - (best_size if new_size > best_size else len(saved_chosen))
                
                if new_size > len(saved_chosen):
                    # Accept if it's an improvement
                    if new_size > best_size:
                        best_size = new_size
                        best_chosen = set(chosen)
                else:
                    # Metropolis criterion: accept with probability exp(-delta/t)
                    if delta >= 0 or rng.random() < (2.718281828 ** (-delta / t)):
                        pass  # Keep the change
                    else:
                        # Revert
                        chosen = saved_chosen
                        conflict_count = saved_conflict_count
        
        t *= cooling_rate
        iteration += 1
    
    return best_chosen


def _solve_instance(vals, deadline):
    n = len(vals)
    if n == 0:
        return []
    
    conflicts = _build_conflict_structure(vals)
    
    # Try SA with multiple restarts
    best_subset = set()
    best_size = 0
    
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 5.0))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        rng = random.Random(42 + restart)
        alloc_time = deadline - time.time() - 1.0
        if alloc_time < 1.0:
            break
        
        subset = _sa_solve(vals, conflicts, time.time() + alloc_time, rng)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
    
    # Verify and return
    subset_list = [vals[i] for i in best_subset]
    if _is_sum_free(vals, subset_list):
        return subset_list
    
    # Fallback: odds
    return [x for x in vals if x % 2]


def construct():
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > _DEADLINE - 5.0:
            vals = _instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        
        vals = _instance(size, seed)
        remaining = _DEADLINE - time.time() - 3.0
        if size <= 60:
            alloc = min(30.0, remaining)
        elif size <= 120:
            alloc = min(50.0, remaining)
        else:
            alloc = remaining
        alloc = max(alloc, 5.0)
        deadline = time.time() + alloc
        subset = _solve_instance(vals, deadline)
        results.append(subset)
    return results
# EVOLVE-BLOCK-END
