import time
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using bitmask-based branch-and-bound with
    multiple greedy initializations and local search.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    spf = [0] * N # Smallest Prime Factor
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
        # Find SPF
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                spf[a] = p
                break
        if spf[a] == 0 and a > 1:
            spf[a] = a
        if a == 1:
            spf[a] = 0 # 1 is coprime to everything, treat as unique group or ignore in bound?
            # 1 is coprime to all. It doesn't conflict with anything.
            # In the clique partition bound, 1 is its own clique?
            # If we group by SPF, 1 has no prime factor.
            # We can just add its weight to the bound directly if available, or treat it as a group with max weight = score[1].
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Multiple greedy initializations with different orderings
    def greedy_construct(order):
        chosen = []
        chosen_mask = 0
        for a in order:
            if (factor_masks[a] & chosen_mask) == 0:
                chosen.append(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_mask, sum(scores[a] for a in chosen)
    
    # Try several different greedy orderings
    orders = [
        list(range(N-1, 0, -1)),           # descending by value (highest score first)
        list(range(1, N)),                  # ascending
        sorted(range(1, N), key=lambda x: -scores[x]),  # by score descending
        sorted(range(1, N), key=lambda x: scores[x]),   # by score ascending
        sorted(range(1, N), key=lambda x: factor_masks[x].bit_count(), reverse=True),  # most factors first
        sorted(range(1, N), key=lambda x: factor_masks[x].bit_count()),  # fewest factors first
    ]
    
    best_chosen = []
    best_score = 0.0
    best_mask = 0
    
    for order in orders:
        chosen, mask, score = greedy_construct(order)
        if score > best_score:
            best_score = score
            best_chosen = chosen
            best_mask = mask
    
    # Local search: try to swap elements to improve
    def local_search(chosen, mask, score):
        chosen_set = set(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, a in enumerate(list(chosen)):
                # Try removing a and adding a better b
                new_mask = mask & ~(factor_masks[a])
                # Find best b not in chosen_set that is coprime to new_mask
                best_b = -1
                best_gain = 0.0
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if (factor_masks[b] & new_mask) == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.discard(a)
                    chosen_set.add(best_b)
                    mask = new_mask | factor_masks[best_b]
                    score += best_gain
                    improved = True
        return chosen, mask, score
    
    best_chosen, best_mask, best_score = local_search(best_chosen, best_mask, best_score)
    
    # Branch and bound for further improvement
    # Sort nodes by score descending
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    
    # Precompute conflict masks for each node
    conflict_mask = [0] * N
    for i, a in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            b = nodes[j]
            if (factor_masks[a] & factor_masks[b]) != 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    n = len(nodes)
    
    best_solution = list(best_chosen)
    best_val = best_score
    
    # Compute upper bound using SPF clique partition
    def compute_upper_bound(available_mask, current_score):
        """
        Upper bound: Sum of max weight in each SPF group among available nodes.
        Nodes with SPF p form a clique, so we can pick at most 1 from each.
        Nodes with SPF 0 (i.e., 1) form their own group.
        """
        max_weight_by_spf = {}
        
        for i in range(n):
            if (available_mask >> i) & 1:
                node_val = nodes[i]
                p = spf[node_val]
                w = scores[node_val]
                if p not in max_weight_by_spf:
                    max_weight_by_spf[p] = w
                else:
                    if w > max_weight_by_spf[p]:
                        max_weight_by_spf[p] = w
        
        bound = current_score
        for p, w in max_weight_by_spf.items():
            bound += w
            
        return bound
    
    # Branch and bound search
    def branch_and_bound(available_mask, current_score, chosen_indices):
        nonlocal best_solution, best_val
        
        if time.time() > deadline:
            return
        
        # Update best if we have a valid solution
        if current_score > best_val:
            best_val = current_score
            best_solution = [nodes[i] for i in chosen_indices]
        
        # If no available nodes, return
        if available_mask == 0:
            return
        
        # Compute upper bound
        ub = compute_upper_bound(available_mask, current_score)
        if ub <= best_val + 1e-12:
            return
        
        # Pick the node with highest score from available
        best_i = -1
        best_score_i = -1.0
        for i in range(n):
            if (available_mask >> i) & 1:
                if scores[nodes[i]] > best_score_i:
                    best_score_i = scores[nodes[i]]
                    best_i = i
        
        if best_i == -1:
            return
        
        # Branch 1: include best_i
        new_available = available_mask & ~(1 << best_i)
        # Remove all nodes that conflict with best_i
        cm = conflict_mask[best_i]
        j = 0
        while cm:
            if cm & 1:
                new_available &= ~(1 << j)
            cm >>= 1
            j += 1
        
        branch_and_bound(new_available, current_score + best_score_i, 
                        chosen_indices + [best_i])
        
        # Branch 2: exclude best_i
        new_available = available_mask & ~(1 << best_i)
        branch_and_bound(new_available, current_score, chosen_indices)
    
    # Start branch and bound from the best greedy solution
    # Build initial available mask excluding already chosen nodes
    chosen_set = set(best_chosen)
    initial_mask = 0
    for i, a in enumerate(nodes):
        if a not in chosen_set:
            initial_mask |= (1 << i)
    
    # Also remove nodes that conflict with chosen ones
    for a in best_chosen:
        for i, b in enumerate(nodes):
            if b in chosen_set:
                continue
            if (factor_masks[a] & factor_masks[b]) != 0:
                initial_mask &= ~(1 << i)
    
    branch_and_bound(initial_mask, best_val, [])
    
    # Final local search on the best solution
    if time.time() < deadline:
        final_mask = 0
        for a in best_solution:
            final_mask |= factor_masks[a]
        best_solution, final_mask, best_val = local_search(best_solution, final_mask, best_val)
    
    return best_solution
