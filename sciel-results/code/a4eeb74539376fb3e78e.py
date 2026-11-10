import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140  # 10 second margin before 150s kill
    
    best_size = 0
    best_set = []
    
    def greedy_build(order, counts_init=None, S_init=None):
        """Build a set greedily from order, optionally starting from initial state."""
        if counts_init is not None:
            counts = list(counts_init)
        else:
            counts = [0] * (2 * N + 1)
        if S_init is not None:
            S = list(S_init)
        else:
            S = []
        
        for x in order:
            if x in S:
                continue
            valid = True
            # Check sums x + y for y in S
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            # Check sum 2x
            if valid:
                s = 2 * x
                if counts[s] >= 2:
                    valid = False
            
            if valid:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S, counts
    
    def try_order(order):
        nonlocal best_size, best_set
        S, _ = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    def simulated_annealing(S, counts, steps=10000):
        """Use simulated annealing to improve the set."""
        if time.time() > deadline - 5:
            return S, counts
        
        S = list(S)
        counts = list(counts)
        in_set = set(S)
        remaining = [x for x in range(1, N + 1) if x not in in_set]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        current_size = len(S)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Random move: swap an element in S with one in remaining
            if not remaining or not S:
                break
            
            i = random.randint(0, len(S) - 1)
            j = random.randint(0, len(remaining) - 1)
            
            out_elem = S[i]
            in_elem = remaining[j]
            
            # Try the swap
            new_S = S[:]
            new_S[i] = in_elem
            new_counts = counts[:]
            
            # Remove out_elem's contributions
            for y in new_S:
                if y != in_elem:
                    s = out_elem + y
                    if new_counts[s] > 0:
                        new_counts[s] -= 1
            s2 = 2 * out_elem
            if new_counts[s2] > 0:
                new_counts[s2] -= 1
            
            # Add in_elem's contributions
            valid = True
            for y in new_S:
                if y != in_elem:
                    s = in_elem + y
                    if new_counts[s] >= 2:
                        valid = False
                        break
            if valid:
                s2 = 2 * in_elem
                if new_counts[s2] >= 2:
                    valid = False
            
            if valid:
                for y in new_S:
                    if y != in_elem:
                        new_counts[in_elem + y] += 1
                new_counts[2 * in_elem] += 1
                
                # Accept if better or with probability exp(-delta/T)
                delta = current_size - len(new_S)  # 0 if same size, negative if worse
                if delta <= 0 or random.random() < (2.718281828 ** (-delta / T)):
                    S = new_S
                    counts = new_counts
                    in_set = set(S)
                    remaining = [x for x in range(1, N + 1) if x not in in_set]
                    current_size = len(S)
                    
                    if len(S) > best_size:
                        nonlocal best_size, best_set
                        best_size = len(S)
                        best_set = list(S)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return S, counts
    
    def local_search(S, counts, max_rounds=5):
        """Improve set by trying to swap out elements that block many others."""
        if time.time() > deadline - 5:
            return S, counts
        
        S = list(S)
        counts = list(counts)
        in_set = set(S)
        remaining = [x for x in range(1, N + 1) if x not in in_set]
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            
            # Try removing each element and see if we can add more
            for i in range(len(S)):
                if time.time() > deadline - 5:
                    break
                elem = S[i]
                
                # Compute counts after removing elem
                new_counts = list(counts)
                new_S = S[:i] + S[i+1:]
                
                # Remove elem's contributions
                for y in new_S:
                    s = elem + y
                    if new_counts[s] > 0:
                        new_counts[s] -= 1
                s2 = 2 * elem
                if new_counts[s2] > 0:
                    new_counts[s2] -= 1
                
                # Try to add elements from remaining
                added_any = False
                for cand in remaining:
                    if cand in in_set:
                        continue
                    valid = True
                    for y in new_S:
                        s = cand + y
                        if new_counts[s] >= 2:
                            valid = False
                            break
                    if valid:
                        s2 = 2 * cand
                        if new_counts[s2] >= 2:
                            valid = False
                    
                    if valid:
                        for y in new_S:
                            new_counts[cand + y] += 1
                        new_counts[2 * cand] += 1
                        new_S.append(cand)
                        in_set.add(cand)
                        added_any = True
                
                # If we added at least one element, this swap is beneficial
                if added_any and len(new_S) > len(S):
                    S = new_S
                    counts = new_counts
                    improved = True
                    # Update remaining
                    in_set = set(S)
                    remaining = [x for x in range(1, N + 1) if x not in in_set]
                    break  # Restart the loop since S changed
        
        return S, counts
    
    # 1. Natural order
    try_order(list(range(1, N + 1)))
    
    # 2. Reverse order
    if time.time() < deadline - 30:
        try_order(list(range(N, 0, -1)))
    
    # 3. Primes first, then composites
    if time.time() < deadline - 30:
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        primes = [x for x in range(1, N+1) if is_prime(x)]
        composites = [x for x in range(1, N+1) if not is_prime(x)]
        try_order(primes + composites)
    
    # 4. Random shuffles with different seeds
    rng = random.Random(42)
    shuffle_count = 0
    while time.time() < deadline - 20:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        try_order(order)
        shuffle_count += 1
        if shuffle_count >= 50:
            break
    
    # 5. Local search on the best set found
    if time.time() < deadline - 10 and best_set:
        # Rebuild counts for best_set
        counts = [0] * (2 * N + 1)
        for i, x in enumerate(best_set):
            for j in range(i + 1):
                s = best_set[i] + best_set[j]
                counts[s] += 1
        
        improved_S, improved_counts = local_search(best_set, counts)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_set = improved_S
    
    # 6. Simulated annealing
    if time.time() < deadline - 5 and best_set:
        # Rebuild counts for best_set
        counts = [0] * (2 * N + 1)
        for i, x in enumerate(best_set):
            for j in range(i + 1):
                s = best_set[i] + best_set[j]
                counts[s] += 1
        
        improved_S, improved_counts = simulated_annealing(best_set, counts)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_set = improved_S
    
    return best_set
