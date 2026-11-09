import random
from math import gcd
import time

N = 128
ALL_MASK = (1 << N) - 1  # bits 1..128 (we use bit i for element i, bit 0 unused)

# Precompute bad triples and conflict structure
def precompute():
    """
    For each x in [1,128], compute:
      - conflict_pairs[x]: a dict mapping (u,v) -> True for each bad pair involving x
      - We also store, for fast bitmask checks, for each x a list of (u, v) pairs
        such that {x, u, v} is a bad triple.
    """
    # bad_triples: list of (a,b,c) with a<b<c and gcd(a,b)==gcd(a,c)==gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad_triples.append((a, b, c))
    
    # For each x, the set of pairs (u,v) that form a bad triple with x
    # Store as a list of (u, v) tuples for iteration
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    # Also store a bitmask of "forbidden partner pairs" for each x:
    # For a given x, if both u and v are in the set, and (u,v) is in conflict_pairs[x],
    # then x cannot be added. We encode each (u,v) pair as a single bit in a 128*128
    # space, but that's too large. Instead, we'll use a different approach:
    # For each x, store a dict: for each u in conflict, the set of v's that conflict with x via u.
    # Then can_add(x, S) checks: for each u in S, if there exists v in S such that (u,v) in conflict_pairs[x].
    # To make this fast, for each x, precompute a bitmask conflict_mask[x] where bit i is set
    # if element i is "conflicting" with x in the sense that there exists some j such that
    # (x, i, j) is a bad triple. But that's not quite right either.
    
    # Better approach: For each x, build a set of "forbidden pairs" as a set of frozensets {u,v}.
    # Then can_add(x, S) = no forbidden pair {u,v} is a subset of S.
    # To speed this up, for each x, build a dict: forbidden_by_u[u] = bitmask of v's such that {u,v} is forbidden.
    # Then can_add(x, S): for each u in S, check if (forbidden_by_u[x].get(u, 0) & S_mask) != 0.
    
    forbidden_by_u = {x: {} for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        for x, u, v in [(a, b, c), (b, a, c), (c, a, b)]:
            if u not in forbidden_by_u[x]:
                forbidden_by_u[x][u] = 0
            forbidden_by_u[x][u] |= (1 << v)
    
    return forbidden_by_u

FORBIDDEN_BY_U = precompute()

def can_add(x, S_mask):
    """
    Check if x can be added to the set represented by S_mask (bitmask).
    S_mask has bit i set iff element i is in the set.
    """
    fb = FORBIDDEN_BY_U[x]
    if not fb:
        return True
    for u, v_mask in fb.items():
        if (S_mask >> u) & 1:  # u is in S
            if v_mask & S_mask:  # some v that conflicts with x via u is also in S
                return False
    return True

def greedy_weighted():
    """
    Greedy: repeatedly pick the element with the fewest remaining conflicts,
    breaking ties by smaller index. This is a degree-based greedy for MIS.
    """
    S_mask = 0
    S_set = set()
    remaining = list(range(1, N + 1))
    
    while remaining:
        # For each candidate, compute its "conflict count" with current set + remaining
        # Actually, for MIS greedy, we pick the vertex with minimum degree in the remaining graph.
        # But our "graph" is a hypergraph. Let's use a simpler heuristic:
        # Pick the element that can be added and has the fewest "conflict pairs" with other remaining elements.
        
        best_x = None
        best_score = None
        
        for x in remaining:
            if not can_add(x, S_mask):
                continue
            # Score: number of remaining elements that would conflict with x
            # i.e., for each u in remaining, check if adding x would create a conflict
            # This is expensive. Let's use a simpler score: len(FORBIDDEN_BY_U[x])
            score = len(FORBIDDEN_BY_U[x])
            if best_score is None or score < best_score or (score == best_score and x < best_x):
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        S_mask |= (1 << best_x)
        S_set.add(best_x)
        remaining.remove(best_x)
    
    return S_mask, S_set

def simulated_annealing(S_mask, S_set, deadline, seed=42):
    """
    Simulated annealing local search with multiple move types.
    Can accept size-decreasing moves to escape local optima.
    """
    rng = random.Random(seed)
    current_mask = S_mask
    current_set = S_set
    current_size = len(S_set)
    best_mask = S_mask
    best_set = S_set
    best_size = current_size
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Choose a random move type
        move_type = rng.choice(['add', 'swap11', 'swap12', 'swap21', 'swap23', 'swap34'])
        
        new_mask = current_mask
        new_set = current_set
        new_size = current_size
        valid = False
        
        if move_type == 'add':
            # Try to add a random element not in set
            candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
            if candidates:
                x = rng.choice(candidates)
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << x)
                    new_set = current_set | {x}
                    new_size = current_size + 1
                    valid = True
        
        elif move_type == 'swap11':
            # Remove one, add one
            if current_set:
                u = rng.choice(list(current_set))
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if candidates:
                    x = rng.choice(candidates)
                    new_mask = current_mask & ~(1 << u)
                    if can_add(x, new_mask):
                        new_mask |= (1 << x)
                        new_set = (current_set - {u}) | {x}
                        new_size = current_size
                        valid = True
        
        elif move_type == 'swap12':
            # Remove one, add two (size +1)
            if current_set:
                u = rng.choice(list(current_set))
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if len(candidates) >= 2:
                    x1, x2 = rng.sample(candidates, 2)
                    new_mask = current_mask & ~(1 << u)
                    if can_add(x1, new_mask):
                        new_mask |= (1 << x1)
                        if can_add(x2, new_mask):
                            new_mask |= (1 << x2)
                            new_set = (current_set - {u}) | {x1, x2}
                            new_size = current_size + 1
                            valid = True
        
        elif move_type == 'swap21':
            # Remove two, add one (size -1, but might escape)
            if len(current_set) >= 2:
                u, v = rng.sample(list(current_set), 2)
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if candidates:
                    x = rng.choice(candidates)
                    new_mask = current_mask & ~(1 << u) & ~(1 << v)
                    if can_add(x, new_mask):
                        new_mask |= (1 << x)
                        new_set = (current_set - {u, v}) | {x}
                        new_size = current_size - 1
                        valid = True
        
        elif move_type == 'swap23':
            # Remove two, add three (size +1)
            if len(current_set) >= 2:
                u, v = rng.sample(list(current_set), 2)
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if len(candidates) >= 3:
                    x1, x2, x3 = rng.sample(candidates, 3)
                    new_mask = current_mask & ~(1 << u) & ~(1 << v)
                    if can_add(x1, new_mask):
                        new_mask |= (1 << x1)
                        if can_add(x2, new_mask):
                            new_mask |= (1 << x2)
                            if can_add(x3, new_mask):
                                new_mask |= (1 << x3)
                                new_set = (current_set - {u, v}) | {x1, x2, x3}
                                new_size = current_size + 1
                                valid = True
        
        elif move_type == 'swap34':
            # Remove three, add four (size +1)
            if len(current_set) >= 3:
                u, v, w = rng.sample(list(current_set), 3)
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if len(candidates) >= 4:
                    x1, x2, x3, x4 = rng.sample(candidates, 4)
                    new_mask = current_mask & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if can_add(x1, new_mask):
                        new_mask |= (1 << x1)
                        if can_add(x2, new_mask):
                            new_mask |= (1 << x2)
                            if can_add(x3, new_mask):
                                new_mask |= (1 << x3)
                                if can_add(x4, new_mask):
                                    new_mask |= (1 << x4)
                                    new_set = (current_set - {u, v, w}) | {x1, x2, x3, x4}
                                    new_size = current_size + 1
                                    valid = True
        
        if not valid:
            T *= cooling
            continue
        
        # Accept/reject
        delta = new_size - current_size
        if delta >= 0:
            # Always accept size-increasing or same-size moves
            current_mask = new_mask
            current_set = new_set
            current_size = new_size
            if current_size > best_size:
                best_mask = current_mask
                best_set = current_set
                best_size = current_size
        else:
            # Size-decreasing move: accept with probability exp(delta / T)
            import math
            prob = math.exp(delta / T)
            if rng.random() < prob:
                current_mask = new_mask
                current_set = new_set
                current_size = new_size
        
        T *= cooling
    
    return best_mask, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_mask = 0
    best_set = set()
    best_size = 0
    
    # 1. Weighted greedy
    if time.time() < deadline:
        mask, s = greedy_weighted()
        if len(s) > best_size:
            best_mask = mask
            best_set = s
            best_size = len(s)
    
    # 2. SA from weighted greedy
    if time.time() < deadline:
        mask, s = simulated_annealing(best_mask, best_set, deadline, seed=1)
        if len(s) > best_size:
            best_mask = mask
            best_set = s
            best_size = len(s)
    
    # 3. Random restarts with SA
    rng = random.Random(12345)
    while time.time() < deadline - 5.0:
        # Random initial set: pick random elements that can be added
        random_mask = 0
        random_set = set()
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if can_add(x, random_mask):
                random_mask |= (1 << x)
                random_set.add(x)
        
        # SA from random start
        mask, s = simulated_annealing(random_mask, random_set, deadline, seed=rng.randint(0, 10**6))
        if len(s) > best_size:
            best_mask = mask
            best_set = s
            best_size = len(s)
        
        # Also try weighted greedy with random tie-breaking
        if time.time() < deadline - 5.0:
            mask2, s2 = greedy_weighted()
            if len(s2) > best_size:
                best_mask = mask2
                best_set = s2
                best_size = len(s2)
    
    return list(best_set)
