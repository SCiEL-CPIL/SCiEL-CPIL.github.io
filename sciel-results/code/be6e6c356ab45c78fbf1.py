import time
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using branch-and-bound.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
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
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Greedy construction: Pick highest a first if coprime to all picked
    def greedy_construct():
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        # Iterate from largest to smallest
        for a in range(N-1, 0, -1):
            if (factor_masks[a] & chosen_mask) == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask, sum(scores[a] for a in chosen)
    
    best_chosen, best_chosen_set, best_chosen_mask, best_score = greedy_construct()
    
    # Branch and bound for maximum weight independent set
    # Nodes are integers 1..511, edge if not coprime
    # We want to find maximum weight independent set
    
    # Sort nodes by score descending (higher score = larger a)
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    
    # Precompute conflict masks for each node (which other nodes conflict with it)
    conflict_mask = [0] * N
    for i, a in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            b = nodes[j]
            if (factor_masks[a] & factor_masks[b]) != 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Branch and bound
    best_solution = list(best_chosen)
    best_val = best_score
    
    # Use bitsets for efficiency
    n = len(nodes)
    
    # Upper bound: fractional knapsack on remaining nodes
    # Sort by score/weight ratio (all weights are 1, so just by score)
    # But we need to account for conflicts
    
    def compute_upper_bound(available_mask, current_score):
        """
        Compute an upper bound on the maximum additional score we can get
        from the available nodes.
        
        Simple bound: sum of top k scores where k is the size of maximum
        independent set in the conflict graph induced by available nodes.
        
        For a tighter bound, we can use a greedy coloring bound or
        fractional knapsack.
        """
        # Simple bound: sum of all scores in available set
        # This is valid but not tight
        total = 0.0
        for i in range(n):
            if (available_mask >> i) & 1:
                total += scores[nodes[i]]
        return current_score + total
    
    # Better bound: use the fact that we can't pick conflicting nodes
    # Greedy coloring bound
    def compute_tight_bound(available_mask, current_score):
        """
        Compute a tighter upper bound using a greedy approach.
        """
        # Get list of available nodes
        avail = []
        for i in range(n):
            if (available_mask >> i) & 1:
                avail.append(i)
        
        if not avail:
            return current_score
        
        # Sort by score descending
        avail.sort(key=lambda i: -scores[nodes[i]])
        
        # Greedy: pick nodes in order, skip if conflicts with already picked
        bound = current_score
        picked_mask = 0
        for i in avail:
            if (conflict_mask[i] & picked_mask) == 0:
                bound += scores[nodes[i]]
                picked_mask |= (1 << i)
        
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
        ub = compute_tight_bound(available_mask, current_score)
        if ub <= best_val + 1e-12:
            return
        
        # Pick the node with highest score from available
        best_i = -1
        best_score_i = -1
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
        for j in range(n):
            if (conflict_mask[best_i] >> j) & 1:
                new_available &= ~(1 << j)
        
        branch_and_bound(new_available, current_score + best_score_i, 
                        chosen_indices + [best_i])
        
        # Branch 2: exclude best_i
        new_available = available_mask & ~(1 << best_i)
        branch_and_bound(new_available, current_score, chosen_indices)
    
    # Start branch and bound
    initial_mask = (1 << n) - 1
    branch_and_bound(initial_mask, 0.0, [])
    
    return best_solution
