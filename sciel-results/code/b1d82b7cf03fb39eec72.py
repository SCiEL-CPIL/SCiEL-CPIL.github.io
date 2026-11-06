"""Optimized solver for large reciprocal weight under LCM separation."""
# EVOLVE-BLOCK-START
import math
from math import gcd
import random

def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    """
    MAX_N = 256
    N = MAX_N - 1  # Number of candidates: 2..256
    
    # Map value to index 0..254
    # val = index + 2
    
    # Precompute incompatibility bitmasks
    # incompat[i] is an integer bitmask where bit j is 1 if i and j are incompatible (lcm <= 256)
    incompat = [0] * N
    for i in range(N):
        vi = i + 2
        for j in range(i + 1, N):
            vj = j + 2
            g = gcd(vi, vj)
            if vi // g * vj <= 256:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
                
    # Precompute reciprocals
    recips = [1.0 / (i + 2) for i in range(N)]
    
    def get_score(state_bits):
        """Calculate score from a bitmask state."""
        score = 0.0
        # Iterate through set bits
        temp = state_bits
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            score += recips[idx]
            temp ^= lsb
        return score

    def is_compatible(node, state_bits):
        """Check if node can be added to state_bits."""
        return (incompat[node] & state_bits) == 0

    def greedy_construct(order):
        """Greedy construction using bitsets."""
        state_bits = 0
        for node in order:
            if is_compatible(node, state_bits):
                state_bits |= (1 << node)
        return state_bits

    def local_search(state_bits):
        """
        Iterative Local Search with 1-for-1 and 1-for-2 moves.
        Returns improved state_bits.
        """
        improved = True
        current_score = get_score(state_bits)
        
        while improved:
            improved = False
            
            # 1. Try single additions
            # Find all nodes not in state that are compatible
            # We iterate through all N nodes.
            for i in range(N):
                if (state_bits >> i) & 1:
                    continue
                if is_compatible(i, state_bits):
                    new_state = state_bits | (1 << i)
                    new_score = current_score + recips[i]
                    if new_score > current_score:
                        state_bits = new_state
                        current_score = new_score
                        improved = True
                        # Restart loop to check for further additions
                        break 

            if improved:
                continue

            # 2. Try 1-for-1 swaps
            best_swap_gain = 0
            best_swap_remove = -1
            best_swap_add = -1
            
            for i in range(N):
                if not ((state_bits >> i) & 1):
                    continue
                # i is in state. Try removing it.
                state_without_i = state_bits & ~(1 << i)
                score_without_i = current_score - recips[i]
                
                for j in range(N):
                    if (state_bits >> j) & 1:
                        continue
                    # j is not in state. Check if j compatible with state_without_i
                    if is_compatible(j, state_without_i):
                        gain = recips[j] - recips[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_remove = i
                            best_swap_add = j
            
            if best_swap_remove != -1:
                state_bits = (state_bits & ~(1 << best_swap_remove)) | (1 << best_swap_add)
                current_score += best_swap_gain
                improved = True
                continue

            # 3. Try 1-for-2 swaps
            # Remove i, add j and k.
            # Gain = recips[j] + recips[k] - recips[i]
            # Conditions:
            # - i in state
            # - j, k not in state
            # - j, k compatible with each other
            # - j, k compatible with state_without_i
            
            best_12_gain = 0
            best_12_remove = -1
            best_12_add1 = -1
            best_12_add2 = -1
            
            for i in range(N):
                if not ((state_bits >> i) & 1):
                    continue
                state_without_i = state_bits & ~(1 << i)
                score_without_i = current_score - recips[i]
                
                # Iterate j
                for j in range(N):
                    if (state_bits >> j) & 1:
                        continue
                    if not is_compatible(j, state_without_i):
                        continue
                    
                    # Iterate k > j to avoid duplicates
                    for k in range(j + 1, N):
                        if (state_bits >> k) & 1:
                            continue
                        if not is_compatible(k, state_without_i):
                            continue
                        # Check j and k compatibility
                        if (incompat[j] >> k) & 1:
                            continue
                        
                        gain = recips[j] + recips[k] - recips[i]
                        if gain > best_12_gain:
                            best_12_gain = gain
                            best_12_remove = i
                            best_12_add1 = j
                            best_12_add2 = k
            
            if best_12_remove != -1:
                state_bits = (state_bits & ~(1 << best_12_remove)) | (1 << best_12_add1) | (1 << best_12_add2)
                current_score += best_12_gain
                improved = True
                continue

        return state_bits

    # Main Search Loop: Iterated Local Search
    best_state = 0
    best_score = 0.0
    
    random.seed(42)
    
    # Initial greedy starts
    # 1. Ascending
    order_asc = list(range(N))
    s1 = greedy_construct(order_asc)
    s1 = local_search(s1)
    sc1 = get_score(s1)
    if sc1 > best_score:
        best_score = sc1
        best_state = s1
        
    # 2. Descending
    order_desc = list(range(N-1, -1, -1))
    s2 = greedy_construct(order_desc)
    s2 = local_search(s2)
    sc2 = get_score(s2)
    if sc2 > best_score:
        best_score = sc2
        best_state = s2
        
    # 3. Iterated Local Search with Random Restarts
    # We will perform many restarts. Each restart starts from a random greedy solution 
    # or a perturbed best solution.
    
    num_restarts = 200
    
    for r in range(num_restarts):
        # Generate a random order for greedy
        order_rand = list(range(N))
        random.shuffle(order_rand)
        
        # Sometimes start from best_state with perturbation (ILS style)
        if r > 20 and random.random() < 0.5:
            # Perturb best_state
            current_state = best_state
            # Remove 1-3 random elements
            num_removals = random.randint(1, 3)
            temp = current_state
            for _ in range(num_removals):
                if temp == 0: break
                # Pick a random set bit
                # Convert to list for random selection, or use bit tricks
                # For simplicity, just pick a random index and if it's set, remove it
                # This is inefficient but N is small.
                # Better: get list of indices
                indices = []
                t = temp
                while t:
                    lsb = t & -t
                    idx = lsb.bit_length() - 1
                    indices.append(idx)
                    t ^= lsb
                if indices:
                    idx_to_remove = random.choice(indices)
                    current_state &= ~(1 << idx_to_remove)
            # Then run local search
            current_state = local_search(current_state)
            sc = get_score(current_state)
            if sc > best_score:
                best_score = sc
                best_state = current_state
        else:
            # Standard greedy restart
            current_state = greedy_construct(order_rand)
            current_state = local_search(current_state)
            sc = get_score(current_state)
            if sc > best_score:
                best_score = sc
                best_state = current_state

    # Convert best_state bitmask to list of values
    result = []
    temp = best_state
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(idx + 2)
        temp ^= lsb
        
    if not result:
        # Fallback to a single valid element, e.g., 256
        return [256]
        
    return result

# EVOLVE-BLOCK-END
