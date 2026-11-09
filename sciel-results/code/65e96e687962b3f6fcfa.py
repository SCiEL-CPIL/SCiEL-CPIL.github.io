import time
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using bitmask-based branch-and-bound with
    multiple greedy initializations and enhanced local search.
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
        # Additional orderings: random-like via bit patterns
        sorted(range(1, N), key=lambda x: ((x * 7919) % 9973)),
        sorted(range(1, N), key=lambda x: ((x * 104729) % 99991)),
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
    
    # Enhanced local search: 1-swap and 2-swap improvements
    def local_search(chosen, mask, score):
        chosen_set = set(chosen)
        improved = True
        pass_count = 0
        while improved and time.time() < deadline:
            improved = False
            pass_count += 1
            
            # 1-swap: replace one element with another
            for i, a in enumerate(list(chosen)):
                new_mask = mask & ~(factor_masks[a])
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
            
            # 2-swap: replace two elements with two better ones
            if pass_count % 2 == 0:  # Do 2-swap every other pass to save time
                for i in range(len(chosen)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(chosen)):
                        if time.time() > deadline:
                            break
                        a = chosen[i]
                        b = chosen[j]
                        # Remove both a and b
                        new_mask = mask & ~(factor_masks[a]) & ~(factor_masks[b])
                        base_score = score - scores[a] - scores[b]
                        
                        # Find best pair (c, d) not in chosen_set, coprime to new_mask, and coprime to each other
                        best_c = -1
                        best_d = -1
                        best_pair_gain = 0.0
                        best_pair_score = -1.0
                        
                        for c in range(1, N):
                            if c in chosen_set:
                                continue
                            if (factor_masks[c] & new_mask) != 0:
                                continue
                            c_mask = factor_masks[c]
                            for d in range(1, N):
                                if d in chosen_set or d == c:
                                    continue
                                if (factor_masks[d] & new_mask) != 0:
                                    continue
                                if (factor_masks[d] & c_mask) != 0:
                                    continue
                                pair_score = scores[c] + scores[d]
                                gain = pair_score - scores[a] - scores[b]
                                if gain > best_pair_gain:
                                    best_pair_gain = gain
                                    best_pair_score = pair_score
                                    best_c = c
                                    best_d = d
                        
                        if best_c != -1 and best_d != -1 and best_pair_gain > 1e-12:
                            chosen[i] = best_c
                            chosen[j] = best_d
                            chosen_set.discard(a)
                            chosen_set.discard(b)
                            chosen_set.add(best_c)
                            chosen_set.add(best_d)
                            mask = new_mask | factor_masks[best_c] | factor_masks[best_d]
                            score += best_pair_gain
                            improved = True
                            break  # Restart 2-swap after one improvement
                    if improved:
                        break
        return chosen, mask, score
    
    best_chosen, best_mask, best_score = local_search(best_chosen, best_mask, best_score)
    
    # Branch and bound for further improvement
    # Sort nodes by score descending
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    
    # Precompute conflict masks for each node
    n = len(nodes)
    conflict_mask = [0] * n
    for i in range(n):
        a = nodes[i]
        for j in range(i+1, n):
            b = nodes[j]
            if (factor_masks[a] & factor_masks[b]) != 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    best_solution = list(best_chosen)
    best_val = best_score
    
    # Precompute sorted node indices for faster upper bound
    node_indices_sorted = list(range(n))  # Already sorted by score descending
    
    # Compute tighter upper bound using Lagrangian relaxation
    def compute_upper_bound(available_mask, current_score):
        """
        Upper bound: sum of top scores we could possibly pick, accounting for conflicts.
        Uses a greedy approach with conflict-aware selection.
        """
        bound = current_score
        picked_mask = 0
        remaining = available_mask
        
        while remaining:
            # Find the highest-score node in remaining that doesn't conflict with picked
            best_i = -1
            best_score_i = -1.0
            m = remaining
            i = 0
            while m:
                if m & 1:
                    if (conflict_mask[i] & picked_mask) == 0:
                        if scores[nodes[i]] > best_score_i:
                            best_score_i = scores[nodes[i]]
                            best_i = i
                m >>= 1
                i += 1
            
            if best_i == -1:
                break
            
            bound += best_score_i
            picked_mask |= (1 << best_i)
            remaining &= ~(1 << best_i)
            # Remove all nodes conflicting with best_i
            cm = conflict_mask[best_i]
            j = 0
            while cm:
                if cm & 1:
                    remaining &= ~(1 << j)
                cm >>= 1
                j += 1
        
        return bound
    
    # Branch and bound search with iterative deepening
    def branch_and_bound(available_mask, current_score, chosen_indices, depth=0):
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
        m = available_mask
        i = 0
        while m:
            if m & 1:
                if scores[nodes[i]] > best_score_i:
                    best_score_i = scores[nodes[i]]
                    best_i = i
            m >>= 1
            i += 1
        
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
                        chosen_indices + [best_i], depth + 1)
        
        # Branch 2: exclude best_i
        new_available = available_mask & ~(1 << best_i)
        branch_and_bound(new_available, current_score, chosen_indices, depth + 1)
    
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
    
    # Final enhanced local search on the best solution
    if time.time() < deadline:
        final_mask = 0
        for a in best_solution:
            final_mask |= factor_masks[a]
        best_solution, final_mask, best_val = local_search(best_solution, final_mask, best_val)
    
    # One more pass of 2-swap if time allows
    if time.time() < deadline - 10:
        best_solution, best_mask, best_val = local_search(best_solution, best_mask, best_val)
    
    return best_solution
