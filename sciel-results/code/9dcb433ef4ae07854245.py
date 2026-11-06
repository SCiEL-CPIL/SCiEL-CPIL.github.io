import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    # Build conflict graph: edge (u,v) exists if there's a bad triple containing both u and v
    # We'll use bitmasks: conflict_mask[i] is a bitmask of all j that conflict with i
    conflict_mask = [0] * (N + 1)
    for (a, b, c) in bad_triples:
        # a conflicts with b and c
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)

    # For fast checking: can we add x to a set S (represented as bitmask)?
    # x can be added if no two elements in S both conflict with x AND form a bad triple with x
    # Actually, we need: for all (u,v) in conflicts[x], not (u in S and v in S)
    # Let's precompute for each x the list of conflicting pairs
    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    def is_valid_add(S_mask, x):
        """Check if x can be added to set represented by bitmask S_mask."""
        for (u, v) in conflicts[x]:
            if (S_mask & (1 << u)) and (S_mask & (1 << v)):
                return False
        return True

    # Greedy initial solution with multiple orders
    best_size = 0
    best_mask = 0

    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(conflicts[x])),
    ]

    for _ in range(20):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S_mask = 0
        for x in order:
            if is_valid_add(S_mask, x):
                S_mask |= (1 << x)
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask

    # Branch and Bound for Maximum Independent Set on the conflict hypergraph
    # We'll use a recursive search with pruning

    # Upper bound: greedy coloring on the remaining candidates
    def greedy_upper_bound(candidates_mask, S_mask):
        """
        Compute an upper bound on how many more elements can be added.
        candidates_mask: bitmask of elements not yet decided
        S_mask: current set
        """
        # Simple bound: count of candidates that can individually be added
        count = 0
        temp_mask = candidates_mask
        while temp_mask:
            # Pick the lowest set bit
            lsb = temp_mask & (-temp_mask)
            x = lsb.bit_length() - 1
            if is_valid_add(S_mask, x):
                count += 1
            temp_mask &= temp_mask - 1
        return count

    # Better upper bound using greedy coloring on a derived graph
    def coloring_upper_bound(candidates_mask, S_mask):
        """
        Use a greedy coloring-based upper bound.
        For each candidate, compute its "conflict degree" within candidates.
        """
        # Build adjacency within candidates based on shared bad triples
        # Two candidates u,v are adjacent if there exists a bad triple containing u,v and some w
        # Actually, for the independence number bound, we need a graph where
        # non-adjacent vertices can coexist. But our constraint is 3-uniform.
        
        # Simpler: just use the count of individually addable elements as before,
        # but with a tighter estimate.
        
        # Let's try a different approach: iteratively pick the element with minimum
        # conflicts and estimate how many can be added.
        
        remaining = candidates_mask
        bound = 0
        while remaining and time.time() < deadline:
            # Find element with minimum number of conflicts within remaining
            best_x = -1
            min_conflicts = N + 1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                x = lsb.bit_length() - 1
                # Count how many other candidates conflict with x
                conf_count = 0
                for (u, v) in conflicts[x]:
                    if (remaining & (1 << u)) and (remaining & (1 << v)):
                        conf_count += 1
                    elif (remaining & (1 << u)):
                        conf_count += 0.5
                    elif (remaining & (1 << v)):
                        conf_count += 0.5
                if conf_count < min_conflicts:
                    min_conflicts = conf_count
                    best_x = x
                temp &= temp - 1
            
            if best_x == -1:
                break
            
            # Check if best_x can be added
            if is_valid_add(S_mask, best_x):
                bound += 1
                remaining &= ~(1 << best_x)
            else:
                # Can't add, remove it
                remaining &= ~(1 << best_x)
        
        return bound

    # Recursive branch and bound
    def branch_and_bound(candidates_mask, S_mask, current_size, best_size_ref):
        """
        candidates_mask: bitmask of elements not yet decided
        S_mask: current set of included elements
        current_size: size of current set
        best_size_ref: list containing the best size found so far (mutable)
        """
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
            return
        
        # Upper bound: if current_size + upper_bound <= best_size, prune
        ub = greedy_upper_bound(candidates_mask, S_mask)
        if current_size + ub <= best_size_ref[0]:
            return
        
        # Pick a variable to branch on
        # Strategy: pick the element with the most conflicts (most constrained)
        temp = candidates_mask
        best_x = -1
        max_conf = -1
        while temp:
            lsb = temp & (-temp)
            x = lsb.bit_length() - 1
            # Count conflicts within candidates
            conf = 0
            for (u, v) in conflicts[x]:
                if (candidates_mask & (1 << u)) and (candidates_mask & (1 << v)):
                    conf += 1
            if conf > max_conf:
                max_conf = conf
                best_x = x
            temp &= temp - 1
        
        if best_x == -1:
            return
        
        # Branch 1: include best_x
        if is_valid_add(S_mask, best_x):
            new_S = S_mask | (1 << best_x)
            new_candidates = candidates_mask & ~(1 << best_x)
            branch_and_bound(new_candidates, new_S, current_size + 1, best_size_ref)
        
        # Branch 2: exclude best_x
        new_candidates = candidates_mask & ~(1 << best_x)
        branch_and_bound(new_candidates, S_mask, current_size, best_size_ref)

    # Run B&B starting from greedy solution
    best_size_ref = [best_size]
    
    # We'll do multiple restarts with different initial solutions
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Randomized greedy to get a different starting point
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        S_mask = 0
        for x in random_order:
            if is_valid_add(S_mask, x):
                S_mask |= (1 << x)
        
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
            best_size_ref[0] = size
        
        # Candidates are those not in S
        candidates_mask = 0
        for x in range(1, N + 1):
            if not (S_mask & (1 << x)):
                candidates_mask |= (1 << x)
        
        branch_and_bound(candidates_mask, S_mask, size, best_size_ref)
        
        if best_size_ref[0] > best_size:
            best_size = best_size_ref[0]

    # Convert best_mask to list
    result = [x for x in range(1, N + 1) if best_mask & (1 << x)]
    
    return result if result else [1]
