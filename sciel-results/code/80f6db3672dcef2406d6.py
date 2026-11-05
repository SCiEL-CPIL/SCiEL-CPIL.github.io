import time
import itertools
import random
from functools import lru_cache
from math import log, exp

N = 192
MAX_ELEM = 48

def _precompute_conflicts():
    """Precompute conflict bitsets for all pairs of elements.
    
    For each element x, compute a bitset where bit y is set if adding x 
    to a set containing y would create a violation (either x divides some 
    triple sum involving y, or y divides some triple sum involving x).
    
    Actually, we need a more nuanced approach. Let's precompute:
    - For each pair (a, b), whether they can coexist (no direct conflict)
    - For each triple (a, b, c), whether any element divides the sum of the other two plus itself
    
    The key constraint: for selected set S, no x in S divides sum of any 
    3 distinct OTHER elements in S.
    
    So if we have x, u, v, w in S, then x must not divide u+v+w.
    
    Let's precompute for each x, a bitset of "forbidden partners" - but this 
    depends on the full set. Instead, let's use a different approach:
    
    For each element x, and each pair (u, v) where u,v != x, check if x 
    divides u+v+? No, we need triples.
    
    Better approach: Precompute for each x, and for each pair (u,v), 
    whether x divides u+v+w for any w. This is complex.
    
    Alternative: Use a bitset representation where for each x, we store 
    which other elements can "conflict" with x given certain conditions.
    
    Let me think differently. The constraint is:
    For all x in S, for all distinct u,v,w in S\{x}: x does not divide u+v+w
    
    This is equivalent to: for all distinct x,u,v,w in S: x does not divide u+v+w
    
    So we can precompute a 4-way conflict table, but that's too large.
    
    Instead, let's use a greedy+SA approach with efficient conflict checking 
    using precomputed triple sums.
    """
    # Precompute all triple sums for triples (i,j,k) with i<j<k
    # triple_sum[i][j][k] = i+j+k
    # But we need to check divisibility, so let's precompute for each x, 
    # the set of triple sums that are divisible by x
    
    # Actually, for SA we need fast "can I add x to current set S?" checks.
    # Let's precompute:
    # For each x in 1..192:
    #   conflict_with[x] = set of y such that there exist u,v in {1..192}\{x,y} 
    #   where x divides u+v+y OR y divides u+v+x
    #   This is still complex.
    
    # Simpler: just precompute all triple sums and use them in the check.
    # For SA, we'll maintain the current set and check conflicts incrementally.
    
    pass

def _can_add_bitset(x, selected_set, selected_list):
    """Check if x can be added to selected_set using bitset-style logic.
    
    selected_set: set of selected integers
    selected_list: list of selected integers (for iteration)
    """
    n = len(selected_list)
    
    # Check 1: x must not divide sum of any 3 distinct elements in selected_list
    if n >= 3:
        for i in range(n):
            a = selected_list[i]
            for j in range(i+1, n):
                b = selected_list[j]
                s_ab = a + b
                for k in range(j+1, n):
                    c = selected_list[k]
                    if (s_ab + c) % x == 0:
                        return False
    
    # Check 2: For each a in selected_list, a must not divide x + u + v 
    # for any u, v in selected_list \ {a}
    for a_idx, a in enumerate(selected_list):
        others = [selected_list[i] for i in range(n) if i != a_idx]
        m = len(others)
        for i in range(m):
            u = others[i]
            for j in range(i+1, m):
                v = others[j]
                if (x + u + v) % a == 0:
                    return False
    return True

def _greedy_fast(order, deadline):
    """Run greedy in the given order with fast conflict checking."""
    selected = []
    selected_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x not in selected_set and _can_add_bitset(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _simulated_annealing(initial_set, deadline, iterations_per_restart=5000):
    """Run simulated annealing to improve the set.
    
    State: a set of selected elements
    Move: add one element, remove one element, or swap one element
    Objective: maximize |S| (cardinality)
    
    We use a temperature schedule to accept worsening moves.
    """
    current_set = list(initial_set)
    current_set_set = set(current_set)
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Temperature schedule
    t0 = 1.0
    t_final = 0.01
    cooling_rate = (t_final / t0) ** (1.0 / max(1, iterations_per_restart))
    
    t = t0
    rng = random.Random(42)
    
    for iteration in range(iterations_per_restart):
        if time.time() > deadline:
            break
        
        # Choose a random move
        move_type = rng.random()
        new_set = list(current_set)
        new_set_set = set(current_set)
        
        if move_type < 0.3:
            # Add a random element not in current set
            candidates = [x for x in range(1, N+1) if x not in new_set_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if _can_add_bitset(x, new_set_set, new_set):
                new_set.append(x)
                new_set_set.add(x)
        elif move_type < 0.6:
            # Remove a random element
            if not new_set:
                continue
            idx = rng.randint(0, len(new_set)-1)
            x = new_set[idx]
            new_set.pop(idx)
            new_set_set.discard(x)
        else:
            # Swap: remove one, add one
            if not new_set:
                continue
            idx = rng.randint(0, len(new_set)-1)
            x_remove = new_set[idx]
            new_set.pop(idx)
            new_set_set.discard(x_remove)
            
            candidates = [x for x in range(1, N+1) if x not in new_set_set]
            if candidates:
                x_add = rng.choice(candidates)
                if _can_add_bitset(x_add, new_set_set, new_set):
                    new_set.append(x_add)
                    new_set_set.add(x_add)
        
        # Evaluate move
        old_size = len(current_set)
        new_size = len(new_set)
        delta = new_size - old_size
        
        if delta > 0:
            # Accept improving move
            current_set = new_set
            current_set_set = new_set_set
            if new_size > best_size:
                best_set = list(new_set)
                best_size = new_size
        else:
            # Accept worsening move with probability exp(delta/t)
            if t > 0:
                prob = exp(delta / t)
                if rng.random() < prob:
                    current_set = new_set
                    current_set_set = new_set_set
        
        # Cool down
        t *= cooling_rate
    
    return best_set

def _local_search_bitset(selected, deadline):
    """Local search using bitset-style conflict checking."""
    selected = list(selected)
    selected_set = set(selected)
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline - 2:
        improved = False
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
        
        # Try adding any element not in selected
        for x in range(N, 0, -1):
            if time.time() > deadline:
                return selected
            if x not in selected_set and _can_add_bitset(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
                improved = True
                no_improve_count = 0
                break
        
        # Try single swap: remove one, add one
        if not improved:
            best_swap_gain = 0
            best_swap_new_set = None
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                candidate = selected[i]
                selected_without = selected[:i] + selected[i+1:]
                selected_without_set = set(selected_without)
                
                for x in range(N, 0, -1):
                    if x in selected_set:
                        continue
                    if not _can_add_bitset(x, selected_without_set, selected_without):
                        continue
                    
                    new_set = selected_without + [x]
                    new_set_set = set(new_set)
                    
                    # Count how many additional elements can be added
                    extra_count = 0
                    for y in range(N, 0, -1):
                        if time.time() > deadline:
                            break
                        if y not in new_set_set and _can_add_bitset(y, new_set_set, new_set):
                            extra_count += 1
                            if extra_count >= 2:
                                break
                    
                    gain = extra_count
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_new_set = new_set
            
            if best_swap_new_set is not None and best_swap_gain > 0:
                selected = best_swap_new_set
                selected_set = set(selected)
                improved = True
                no_improve_count = 0
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy descending
    result = _greedy_fast(range(N, 0, -1), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 10:
        result = _greedy_fast(range(1, N+1), deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 3: Random shuffles
    random.seed(42)
    num_random = 20
    for i in range(num_random):
        if time.time() > deadline - 20:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        result = _greedy_fast(order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 4: Simulated annealing on best result
    if time.time() < deadline - 15:
        sa_result = _simulated_annealing(best, deadline, iterations_per_restart=3000)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Strategy 5: Local search polish
    if time.time() < deadline - 5:
        best = _local_search_bitset(best, deadline)
        best_size = len(best)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 1:
        in_set = set(best)
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x not in in_set and _can_add_bitset(x, in_set, best):
                best.append(x)
                in_set.add(x)
    
    return best
