"""Improved B&B: bitmask-based with tight fractional LP upper bound and branch ordering."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    N = 512

    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime

    is_prime = sieve(511)

    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)

    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)

    # Build conflict graph: two numbers conflict if gcd > 1
    n_nodes = 511
    conflicts = [set() for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if prime_factors[i+1] & prime_factors[j+1]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Sort nodes by score descending
    order = sorted(range(n_nodes), key=lambda x: scores[x+1], reverse=True)
    pos_to_node = order
    node_to_pos = [0] * n_nodes
    for pos, node in enumerate(order):
        node_to_pos[node] = pos

    # Rebuild conflicts in sorted order
    conflicts_sorted = [set() for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in conflicts[i]:
            conflicts_sorted[node_to_pos[i]].add(node_to_pos[j])

    # Precompute scores in sorted order
    scores_sorted = [scores[node+1] for node in order]

    # Build bitmask for conflicts
    conflict_masks = [0] * n_nodes
    for pos in range(n_nodes):
        mask = 0
        for nb in conflicts_sorted[pos]:
            mask |= (1 << nb)
        conflict_masks[pos] = mask

    # Compute an upper bound using LP relaxation
    # The LP relaxation is: maximize sum w_i x_i subject to x_i + x_j <= 1 for all edges
    # A simple upper bound: sum of top k weights where k is a bound on independent set size
    # Better: use the fact that in any independent set, no two adjacent nodes can both be chosen
    # Upper bound: for each connected component, the upper bound is the max weight node + sum of others
    # Even simpler: use greedy coloring to partition into cliques, upper bound = sum of max per clique

    def compute_ub_lp(available_mask):
        """
        Compute upper bound using LP relaxation approximation.
        Strategy: repeatedly find a maximal clique, take the max weight node, 
        remove the clique, accumulate. This gives a valid upper bound because
        in any independent set, at most one node from each clique can be chosen.
        """
        if available_mask == 0:
            return 0.0
        
        ub = 0.0
        remaining = available_mask
        
        while remaining != 0:
            # Find a maximal clique greedily
            # Start with the highest-weight node
            best = None
            best_w = -1
            m = remaining
            while m:
                lsb = m & -m
                pos = lsb.bit_length() - 1
                if scores_sorted[pos] > best_w:
                    best_w = scores_sorted[pos]
                    best = pos
                m -= lsb
            
            # Build clique: start with best, add neighbors that are adjacent to all in clique
            clique = {best}
            candidates = remaining & conflict_masks[best]
            while candidates:
                lsb = candidates & -candidates
                pos = lsb.bit_length() - 1
                # Check if pos is adjacent to all in clique
                is_adj = True
                for c in clique:
                    if not (conflict_masks[pos] & (1 << c)):
                        is_adj = False
                        break
                if is_adj:
                    clique.add(pos)
                    candidates -= (1 << pos)
                candidates -= lsb
            
            # Take max weight from clique
            max_w = 0.0
            for pos in clique:
                if scores_sorted[pos] > max_w:
                    max_w = scores_sorted[pos]
            ub += max_w
            
            # Remove clique from remaining
            for pos in clique:
                remaining &= ~(1 << pos)
        
        return ub

    # Branch and bound
    best_weight = 0.0
    best_solution_mask = 0
    node_count = [0]

    def branch_and_bound(available_mask, current_mask, current_weight):
        nonlocal best_weight, best_solution_mask, node_count
        
        node_count[0] += 1
        
        if time.time() > deadline:
            return
        
        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution_mask = current_mask
            return
        
        # Upper bound check
        ub = compute_ub_lp(available_mask)
        if current_weight + ub <= best_weight + 1e-15:
            return
        
        # Pick the next node to branch on: highest weight available
        candidates = []
        m = available_mask
        while m:
            lsb = m & -m
            pos = lsb.bit_length() - 1
            candidates.append(pos)
            m -= lsb
        
        candidates.sort(key=lambda p: scores_sorted[p], reverse=True)
        
        for pos in candidates:
            if time.time() > deadline:
                return
            
            # Option 1: include pos
            new_current_inc = current_mask | (1 << pos)
            new_available_inc = available_mask & ~(1 << pos) & ~conflict_masks[pos]
            new_weight_inc = current_weight + scores_sorted[pos]
            
            if new_available_inc == 0:
                if new_weight_inc > best_weight:
                    best_weight = new_weight_inc
                    best_solution_mask = new_current_inc
            else:
                ub_inc = compute_ub_lp(new_available_inc)
                if new_weight_inc + ub_inc > best_weight + 1e-15:
                    branch_and_bound(new_available_inc, new_current_inc, new_weight_inc)
            
            # Option 2: exclude pos
            new_available_exc = available_mask & ~(1 << pos)
            if new_available_exc == 0:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_solution_mask = current_mask
            else:
                ub_exc = compute_ub_lp(new_available_exc)
                if current_weight + ub_exc > best_weight + 1e-15:
                    branch_and_bound(new_available_exc, current_mask, current_weight)

    # Get an initial feasible solution using greedy
    greedy_mask = 0
    greedy_weight = 0.0
    remaining = (1 << n_nodes) - 1
    while remaining:
        best = None
        best_w = -1
        m = remaining
        while m:
            lsb = m & -m
            pos = lsb.bit_length() - 1
            if scores_sorted[pos] > best_w:
                best_w = scores_sorted[pos]
                best = pos
            m -= lsb
        greedy_mask |= (1 << best)
        greedy_weight += scores_sorted[best]
        remaining &= ~(1 << best)
        remaining &= ~conflict_masks[best]
    
    best_weight = greedy_weight
    best_solution_mask = greedy_mask
    
    # Run B&B
    all_mask = (1 << n_nodes) - 1
    branch_and_bound(all_mask, 0, 0.0)
    
    # Convert back to original values
    result = []
    m = best_solution_mask
    while m:
        lsb = m & -m
        pos = lsb.bit_length() - 1
        result.append(pos_to_node[pos] + 1)
        m -= lsb
    
    return result
