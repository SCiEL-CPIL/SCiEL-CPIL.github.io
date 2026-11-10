import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 512
    MAX_K = 48
    
    def is_valid(set_vals):
        """
        Check if no element is the mean of any subset of at least two other elements.
        Uses integer bitsets for subset sums.
        """
        m = len(set_vals)
        if m < 3:
            return True
        
        vals = sorted(set_vals)
        
        # For each a in vals, check if k*a is a subset sum of size k from vals \ {a}
        for i, a in enumerate(vals):
            # Construct list of others
            # To avoid creating new lists repeatedly, we can pass indices
            # But for m=48, creating lists is fine.
            others = vals[:i] + vals[i+1:]
            n_others = m - 1
            
            # dp[k] is a bitset of sums achievable with k elements
            # Max sum: n_others * 512
            dp = [0] * (n_others + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for v in others:
                # Update in reverse order to avoid using v multiple times in same step
                for k in range(n_others, 0, -1):
                    dp[k] |= dp[k-1] << v
            
            # Check if k*a is achievable for any k >= 2
            # k*a can be at most n_others * 512
            for k in range(2, n_others + 1):
                target = k * a
                # If target is too large, it can't be in the bitset
                if target <= n_others * 512:
                    if (dp[k] >> target) & 1:
                        return False
        return True

    def greedy_construct(random_order=False):
        selected = []
        candidates = list(range(1, N + 1))
        if random_order:
            random.shuffle(candidates)
        
        for x in candidates:
            # Quick check: if x is too small or large?
            # Just try to add
            candidate_set = selected + [x]
            if is_valid(candidate_set):
                selected.append(x)
                if len(selected) >= MAX_K:
                    break
        return selected

    # Try multiple greedy strategies
    best_set = []
    best_size = 0
    
    # 1. Standard increasing greedy
    current = greedy_construct(random_order=False)
    best_set = current[:]
    best_size = len(current)
    
    if best_size >= MAX_K:
        return best_set

    # 2. Randomized greedy restarts
    # Use a fixed seed for reproducibility in the function, but vary it per restart?
    # The task says "random" is available. We should probably use a fixed seed for determinism if required, 
    # but the prompt implies we can use random.
    # Let's use a few restarts.
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Vary the random seed or shuffle
        # We can shuffle the candidates
        current = greedy_construct(random_order=True)
        if len(current) > best_size:
            best_set = current[:]
            best_size = len(current)
            if best_size >= MAX_K:
                return best_set

    # 3. Local search to improve the best set found
    # Try to swap elements to increase size or just maintain size but allow different configurations?
    # Since we want to maximize size, if we are at size K < 48, we try to add elements.
    # If we are stuck at K, we try to swap an element to "free up" space for a new one?
    # Actually, if a set of size K is valid, and we can't add any element, 
    # maybe swapping an element allows adding two?
    
    # Simple local search: try to add a random element. If it fails, try to swap one out.
    current_set = set(best_set)
    
    # If we have space, try to fill it
    if len(current_set) < MAX_K:
        for _ in range(1000): # Limit iterations
            if time.time() > deadline - 5:
                break
            
            # Try to add a random element
            if len(current_set) < MAX_K:
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                if candidates:
                    x = random.choice(candidates)
                    new_set = current_set | {x}
                    if is_valid(list(new_set)):
                        current_set = new_set
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                            if best_size >= MAX_K:
                                return best_set
            
            # Try to swap
            if current_set:
                a = random.choice(list(current_set))
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                if candidates:
                    b = random.choice(candidates)
                    new_set = (current_set - {a}) | {b}
                    if is_valid(list(new_set)):
                        current_set = new_set
                        # Size is same, but might enable future additions?
                        # We don't track "potential", so we just keep the best size.
                        # If size is same, we don't update best_set unless we want to explore more.
                        # Let's update current_set to explore from this new state.

    return best_set
