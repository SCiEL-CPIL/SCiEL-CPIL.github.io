import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Precompute all 4-APs involving each element.
    # conflicts[i] is a list of lists, where each sublist contains the indices
    # of the other 3 elements in a 4-AP with i.
    conflicts = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            a0 = a - 1
            a1 = a + d - 1
            a2 = a + 2 * d - 1
            a3 = a + 3 * d - 1
            
            conflicts[a0].append([a1, a2, a3])
            conflicts[a1].append([a0, a2, a3])
            conflicts[a2].append([a0, a1, a3])
            conflicts[a3].append([a0, a1, a2])

    def is_valid_add(S_mask, z_idx):
        """
        Check if adding z_idx to S_mask creates a 4-AP.
        S_mask is an integer bitmask.
        """
        for triple in conflicts[z_idx]:
            # Check if all 3 are in S
            # Using bitwise operations
            mask = 0
            for idx in triple:
                mask |= (1 << idx)
            if (S_mask & mask) == mask:
                return False
        return True

    def greedy_refill(S_mask, order=None):
        """
        Greedily add valid elements to S_mask.
        Returns updated S_mask and size.
        """
        if order is None:
            order = list(range(N))
            random.shuffle(order)
        
        size = bin(S_mask).count('1')
        # We can optimize this by keeping track of size, but bin().count is fast enough for N=512
        # However, doing it inside the loop is bad.
        
        for z_idx in order:
            if time.time() > deadline:
                break
            if not ((S_mask >> z_idx) & 1):
                if is_valid_add(S_mask, z_idx):
                    S_mask |= (1 << z_idx)
                    size += 1
        return S_mask, size

    # 1. Initial Greedy Phase
    best_S = 0
    best_size = 0
    
    # Run multiple random greedy initializations
    for _ in range(20):
        S_curr = 0
        S_curr, size_curr = greedy_refill(S_curr)
        if size_curr > best_size:
            best_size = size_curr
            best_S = S_curr
            
    # 2. Local Search / Iterated Local Search
    # Strategy: 
    # - Start with best_S
    # - Perturb: Remove k random elements
    # - Refill: Greedy refill
    # - If improved, update best
    # - To escape local optima, we can also try 1-swap moves (remove i, add j)
    #   But greedy refill after removal is often stronger.
    
    # Let's implement a more robust local search:
    # 1. Greedy Refill until no more can be added (using random order)
    # 2. If stuck, try to remove a small set of elements and refill.
    # 3. To improve further, we can try "kick" moves: remove 1-3 random elements, refill.
    
    current_S = best_S
    current_size = best_size
    
    # Optimization: Precompute bit masks for conflicts to speed up is_valid_add?
    # is_valid_add iterates ~N/3 times. Each iteration does a few bit ops.
    # It's reasonably fast.
    
    while time.time() < deadline:
        # Perturb
        S_trial = current_S
        
        # Randomly remove 1 to 5 elements
        num_remove = random.randint(1, 5)
        S_bits = [i for i in range(N) if (S_trial >> i) & 1]
        
        if len(S_bits) <= num_remove:
            continue
            
        to_remove = random.sample(S_bits, num_remove)
        for r_idx in to_remove:
            S_trial &= ~(1 << r_idx)
            
        # Refill
        S_trial, size_trial = greedy_refill(S_trial)
        
        if size_trial > best_size:
            best_size = size_trial
            best_S = S_trial
            current_S = S_trial
            current_size = size_trial
            # If we made progress, we might want to restart the perturbation size?
            # Or just continue.
            
    # Final Polish: Try to add any remaining elements with a fixed order (1 to N)
    # Sometimes random order misses easy additions? Unlikely, but a final pass with sorted order 
    # might catch something if the random order was unlucky in the last few steps? 
    # Actually, greedy_refill already tries to add all possible valid elements in the given order.
    # If an element is valid, it gets added. The order only matters for *which* valid elements are added
    # if there are conflicts between valid elements? No, if X is valid given S, adding X doesn't make Y invalid 
    # unless Y forms an AP with X and other elements in S.
    # Wait, adding X can make Y invalid.
    # So the order matters.
    # A final pass with a different order might help.
    
    # Let's do a few more random refills on the best_S found, just in case.
    for _ in range(10):
        if time.time() > deadline:
            break
        # Try to add elements not in best_S
        # We can't just refill best_S because it's already "full" relative to the previous random order.
        # But a different random order might allow a different set of additions?
        # No, if S is maximal (no valid additions), no order helps.
        # Is best_S maximal?
        # greedy_refill returns when it has scanned the entire list.
        # So best_S is maximal with respect to the specific random order used in the last call.
        # However, if we change the order, we might find a different maximal set of the same size, 
        # or potentially a larger one if the previous greedy choice was suboptimal (greedy is not optimal).
        # But greedy_refill adds *any* valid element. It doesn't "choose" between valid elements in a way that 
        # blocks future valid elements *unless* the valid elements conflict with each other.
        # If X and Y are both valid for S, but X and Y form a 4-AP with some Z in S, then we can only add one.
        # Greedy picks the first one in the order.
        # So yes, different orders yield different maximal sets.
        # So running greedy_refill on a *subset* of best_S (perturbation) is the way to go.
        # The loop above does this.
        
        # Let's try a slightly different perturbation: remove 1 element, refill.
        # This is a 1-exchange move essentially.
        S_trial = best_S
        S_bits = [i for i in range(N) if (S_trial >> i) & 1]
        if S_bits:
            r_idx = random.choice(S_bits)
            S_trial &= ~(1 << r_idx)
            S_trial, size_trial = greedy_refill(S_trial)
            if size_trial > best_size:
                best_size = size_trial
                best_S = S_trial

    # Convert bitmask to list of values (1-indexed)
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
    
    return sorted(result)
