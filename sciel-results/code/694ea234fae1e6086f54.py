import time
import random

def construct():
    """
    Build a set of distinct integers in 1..256 maximizing size such that
    no subset sum is achievable by subsets of different cardinalities.
    
    Uses Simulated Annealing with bitset-based validity checks.
    """
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    def is_done():
        return elapsed() > DEADLINE

    def compute_sums_by_k(A):
        """Compute sums_by_k for set A using bitsets. sums_by_k[k] is a bitmask of sums."""
        n = len(A)
        sums_by_k = [0] * (n + 1)
        sums_by_k[0] = 1  # Bit 0 set, sum 0
        for a in A:
            for k in range(n - 1, -1, -1):
                if sums_by_k[k]:
                    sums_by_k[k + 1] |= (sums_by_k[k] << a)
        return sums_by_k

    def is_valid_sums(sums_by_k):
        """Check if no sum appears in two different cardinality classes."""
        n = len(sums_by_k)
        for i in range(n):
            for j in range(i + 1, n):
                if sums_by_k[i] & sums_by_k[j]:
                    return False
        return True

    def greedy_build(order, max_size=60):
        """Build a valid set by greedily adding elements in the given order."""
        A = []
        sums_by_k = [1]  # Bit 0 set
        
        for candidate in order:
            if len(A) >= max_size:
                break
            n = len(A)
            # Try adding candidate
            new_sums_by_k = [0] * (n + 2)
            for k in range(n + 1):
                new_sums_by_k[k] = sums_by_k[k]
            
            for k in range(n, -1, -1):
                if sums_by_k[k]:
                    new_sums_by_k[k + 1] |= (sums_by_k[k] << candidate)
            
            if is_valid_sums(new_sums_by_k):
                A.append(candidate)
                sums_by_k = new_sums_by_k
            if is_done():
                break
        
        return A

    def simulated_annealing(A_init, iterations=100000):
        """
        Improve set A using Simulated Annealing.
        State: List of integers.
        Objective: Maximize size.
        Moves: Add, Remove, Swap.
        """
        A = list(A_init)
        best_A = list(A)
        best_size = len(A)
        
        # Precompute candidates
        all_candidates = list(range(1, 257))
        
        # Initial temperature
        T = 1.0
        cooling = 0.9995
        
        for i in range(iterations):
            if is_done():
                break
            
            n = len(A)
            A_set = set(A)
            
            # Choose a move type
            # Bias towards adding if possible
            move_type = random.choices(['add', 'swap', 'remove'], weights=[0.6, 0.3, 0.1])[0]
            
            new_A = None
            
            if move_type == 'add':
                if n >= 60:
                    continue
                # Pick a random candidate not in A
                candidates_not_in_A = [c for c in all_candidates if c not in A_set]
                if not candidates_not_in_A:
                    continue
                c = random.choice(candidates_not_in_A)
                new_A = A + [c]
                
            elif move_type == 'swap':
                if n == 0:
                    continue
                # Pick element to remove
                idx = random.randint(0, n - 1)
                removed = A[idx]
                A_temp = A[:idx] + A[idx+1:]
                A_temp_set = set(A_temp)
                
                # Pick element to add
                candidates_not_in_A_temp = [c for c in all_candidates if c not in A_temp_set]
                if not candidates_not_in_A_temp:
                    continue
                c = random.choice(candidates_not_in_A_temp)
                new_A = A_temp + [c]
                
            else: # remove
                if n == 0:
                    continue
                idx = random.randint(0, n - 1)
                new_A = A[:idx] + A[idx+1:]
            
            # Check validity of new_A
            if new_A is None:
                continue
            
            # Fast validity check
            if len(new_A) > 0:
                sums_new = compute_sums_by_k(new_A)
                valid = is_valid_sums(sums_new)
            else:
                valid = True
            
            if valid:
                # Accept if size is larger, or with probability exp(-delta/T)
                size_diff = len(new_A) - n
                if size_diff >= 0:
                    A = new_A
                    if len(A) > best_size:
                        best_size = len(A)
                        best_A = list(A)
                else:
                    # Accept worse move (smaller size) to escape local minima
                    # Delta is negative (size decrease), so -delta/T is positive
                    # We want to accept with prob exp(size_diff / T)
                    # If T is small, this prob is very small for size_diff < 0
                    if random.random() < pow(2.71828, size_diff / T):
                        A = new_A
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_A

    # Strategy 1: Greedy from small to large
    best_A = greedy_build(range(1, 257))
    
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Strategy 2: Greedy from large to small
    A2 = greedy_build(range(256, 0, -1))
    if len(A2) > len(best_A):
        best_A = A2
    
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Strategy 3: Simulated Annealing on best found
    # Run SA with a large number of iterations, bounded by time
    A_sa = simulated_annealing(best_A, iterations=200000)
    if len(A_sa) > len(best_A):
        best_A = A_sa
        
    if is_done():
        return [int(x) for x in best_A] if best_A else [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Strategy 4: Random restarts with SA
    random.seed(12345)
    
    for restart in range(10):
        if is_done():
            break
        if len(best_A) >= 60:
            break
        
        # Create a random order
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        A_rand = greedy_build(candidates)
        
        # Short SA on random start
        A_rand_sa = simulated_annealing(A_rand, iterations=50000)
        
        if len(A_rand_sa) > len(best_A):
            best_A = A_rand_sa

    # Final validation
    if not best_A:
        best_A = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Ensure all are Python ints in a list
    return [int(x) for x in best_A]
