import time
import math
import random

# EVOLVE-BLOCK-START

def _precompute(N=180):
    """
    Precompute all forbidden triples and build an adjacency/conflict structure.
    
    A forbidden triple is (a, b, c) with distinct elements where 1/a = 1/b + 1/c.
    This is equivalent to a(b+c) = bc, or c = ab/(b-a) for b > a.
    """
    triples = []
    # For each element x, list of (y, z) pairs such that {x, y, z} is a forbidden triple
    conflict_pairs = {i: [] for i in range(1, N + 1)}
    # For each pair (x, y) with x < y, list of z values that complete a forbidden triple
    pair_conflicts = {}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # We have a forbidden triple {a, b, c}
                    triples.append((a, b, c))
                    # Record conflicts
                    if c > b:
                        # a is smallest, then b, then c
                        conflict_pairs[a].append((b, c))
                        conflict_pairs[b].append((a, c))
                        conflict_pairs[c].append((a, b))
                    else:
                        # a is smallest, then c, then b (c < b)
                        conflict_pairs[a].append((b, c))
                        conflict_pairs[b].append((a, c))
                        conflict_pairs[c].append((a, b))
    
    return triples, conflict_pairs

def _check_valid_fast(S, conflict_pairs):
    """
    Check if set S is valid (no forbidden triples) using conflict_pairs.
    For each element in S, check if any of its conflict pairs are both in S.
    """
    S_set = set(S)
    for x in S:
        for (y, z) in conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
    return True

def _greedy(N, conflict_pairs, order):
    """
    Greedy construction: add elements in given order, skipping any that
    would create a forbidden triple.
    """
    S = set()
    for x in order:
        can_add = True
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    return S

def _local_search(N, S, conflict_pairs, deadline, max_iters=20000):
    """
    Local search: iteratively try to add elements not in S.
    Also try swaps (remove one, add another) to escape local optima.
    """
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    rng = random.Random(42)
    
    all_elements = list(range(1, N + 1))
    
    iter_count = 0
    while iter_count < max_iters:
        if time.time() > deadline:
            break
        iter_count += 1
        
        not_in_S = [x for x in all_elements if x not in S]
        if not not_in_S:
            break
        
        # Try to add a random element
        progress = False
        rng.shuffle(not_in_S)
        for x in not_in_S:
            can_add = True
            for (y, z) in conflict_pairs[x]:
                if y in S and z in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                progress = True
                break
        
        if not progress:
            # Try a swap: remove one element from S, add one from not_in_S
            S_list = list(S)
            rng.shuffle(S_list)
            swap_done = False
            for rem in S_list:
                for add in not_in_S:
                    if add == rem:
                        continue
                    # Check if S - {rem} + {add} is valid
                    new_S = (S - {rem}) | {add}
                    # Quick check: is new_S valid?
                    valid = True
                    for elem in new_S:
                        for (y, z) in conflict_pairs[elem]:
                            if y in new_S and z in new_S:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S = new_S
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        swap_done = True
                        break
                if swap_done:
                    break
            
            # If no progress for a while, try a larger restart
            if not swap_done and iter_count % 500 == 0:
                # Try a random restart with a subset
                pass
    
    return best_S

def _simulated_annealing(N, conflict_pairs, deadline, initial_temp=100.0, cooling=0.999):
    """
    Simulated annealing on set membership.
    State: subset of [1, N].
    Move: add or remove a random element.
    Objective: maximize size while maintaining validity.
    """
    S = set()
    best_S = set()
    best_size = 0
    temp = initial_temp
    rng = random.Random(123)
    
    all_elements = list(range(1, N + 1))
    
    # Start with a random small set
    for _ in range(30):
        x = rng.choice(all_elements)
        can_add = True
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                can_add = False
                break
        if can_add:
            S.add(x)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    iter_count = 0
    max_iters = 200000
    while iter_count < max_iters and temp > 0.1:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Random move: add or remove
        if len(S) < N // 2 or rng.random() < 0.5:
            # Try to add
            x = rng.choice(all_elements)
            if x not in S:
                # Check validity after adding
                can_add = True
                for (y, z) in conflict_pairs[x]:
                    if y in S and z in S:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    # Invalid, accept with probability
                    # We don't accept invalid states; just skip
                    pass
        else:
            # Try to remove
            if S:
                x = rng.choice(list(S))
                S.remove(x)
        
        temp *= cooling
    
    return best_S

def construct():
    """
    Return a nonempty subset of [1,180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a=1/b+1/c. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 180
    triples, conflict_pairs = _precompute(N)
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with various orderings
    # Descending order (large numbers first - less likely to conflict)
    S = _greedy(N, conflict_pairs, list(range(N, 0, -1)))
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Ascending order
    S = _greedy(N, conflict_pairs, list(range(1, N + 1)))
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Random orderings
    rng = random.Random(42)
    for i in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = _greedy(N, conflict_pairs, order)
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Strategy 2: Local search from best greedy result
    if time.time() < deadline - 5:
        improved = _local_search(N, list(best_set), conflict_pairs, deadline, max_iters=10000)
        if len(improved) > best_size:
            best_set = set(improved)
            best_size = len(improved)
    
    # Strategy 3: More random greedy + local search
    while time.time() < deadline - 10:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = _greedy(N, conflict_pairs, order)
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
        
        if time.time() < deadline - 5:
            improved = _local_search(N, list(best_set), conflict_pairs, deadline, max_iters=5000)
            if len(improved) > best_size:
                best_set = set(improved)
                best_size = len(improved)
    
    # Fallback
    if not best_set:
        best_set = set(range(91, N + 1))
    
    return list(best_set)

# EVOLVE-BLOCK-END

