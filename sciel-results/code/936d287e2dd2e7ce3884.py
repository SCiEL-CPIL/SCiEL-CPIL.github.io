import time
import random

def construct():
    deadline = time.time() + 85  # Leave 5s margin
    
    N = 512
    MAX_K = 48
    
    def is_valid_addition(x, dp_main, dp_removed, S):
        """
        Check if x can be added to S.
        dp_main: DP for S
        dp_removed: dict mapping y in S to DP for S \ {y}
        S: list of elements in S
        """
        # 1. Check if x is the mean of any subset of S
        # x = sum(A) / |A| => sum(A) = k * x
        # A is subset of S, |A| >= 2
        for k in range(2, len(S) + 1):
            target_sum = k * x
            if target_sum > 24576: # Max possible sum
                break
            if (dp_main[k] >> target_sum) & 1:
                return False
        
        # 2. Check if adding x makes any y in S a mean
        # y = sum(A) / |A| where A is subset of S U {x}, |A| >= 2, x in A
        # Let A = {x} U B, where B is subset of S \ {y}
        # y = (x + sum(B)) / (1 + |B|)
        # (1 + |B|) * y = x + sum(B)
        # sum(B) = (1 + |B|) * y - x
        # Let m = 1 + |B|. Since |A| >= 2, m >= 2.
        # |B| = m - 1. Since B is subset of S \ {y}, and we need to form a mean of *at least two other elements*,
        # the total subset size is m. The elements are from S \ {y}. One is x, m-1 are from S \ {y, x}.
        # Wait, the condition is "subset of at least two OTHER elements".
        # The other elements are S \ {y}.
        # The subset A must be a subset of S \ {y}.
        # If x is in A, then A contains x and some elements from S \ {y, x}.
        # Let B be the part of A from S \ {y, x}.
        # A = {x} U B.
        # |A| = 1 + |B|.
        # Condition: |A| >= 2 => 1 + |B| >= 2 => |B| >= 1.
        # So we need to find if there exists B subset of S \ {y} with |B| = m-1 (where m >= 2)
        # such that sum(B) = m * y - x.
        
        for y in S:
            dp_y = dp_removed[y]
            # m ranges from 2 to len(S) (since B is subset of S\{y}, max size of B is len(S)-1, so max m is len(S))
            # Actually, B is subset of S \ {y}. Max size of B is len(S)-1.
            # So m-1 <= len(S)-1 => m <= len(S).
            for m in range(2, len(S) + 1):
                target_sum = m * y - x
                if target_sum < 0:
                    continue
                if target_sum > 24576:
                    break
                # Check if target_sum is achievable by m-1 elements from S \ {y}
                if (dp_y[m-1] >> target_sum) & 1:
                    return False
        return True

    def add_element(x, S, dp_main, dp_removed):
        """
        Add x to S and update DP structures.
        """
        new_S = S + [x]
        new_dp_main = [0] * (len(new_S) + 1)
        new_dp_removed = {}
        
        # Update dp_main
        for k in range(len(new_S), 0, -1):
            new_dp_main[k] = dp_main[k] | (dp_main[k-1] << x)
        new_dp_main[0] = 1
        
        # Update dp_removed
        # For each y in old S, new S \ {y} = old S \ {y} U {x}
        for y in S:
            old_dp_y = dp_removed[y]
            new_dp_y = [0] * (len(new_S)) # Size is len(old S) because we remove y from new S of size len(old S)+1
            # Wait, new S has size N+1. S \ {y} has size N.
            # old S had size N-1. old S \ {y} had size N-2.
            # So new_dp_y needs to support up to N-1 elements.
            # dp arrays are 1-indexed for count.
            # old_dp_y had size (N-1) + 1 = N.
            # new_dp_y needs size N + 1? No.
            # Max elements in new S \ {y} is N-1.
            # So we need indices 0..N-1.
            # Let's just allocate len(new_S) which is N+1. Safe.
            
            # Copy old values
            for k in range(len(old_dp_y)):
                new_dp_y[k] = old_dp_y[k]
            
            # Add x
            for k in range(len(new_S) - 1, 0, -1): # Max elements is len(new_S)-1
                 if new_dp_y[k-1]:
                     new_dp_y[k] |= (new_dp_y[k-1] << x)
            
            new_dp_removed[y] = new_dp_y
        
        # For x itself, new S \ {x} = old S
        new_dp_removed[x] = dp_main[:] # Copy old dp_main
        
        return new_S, new_dp_main, new_dp_removed

    best_size = 0
    best_set = []
    
    # Strategy: Multiple random restarts with greedy construction
    # Use diverse orderings
    orderings = []
    
    # 1. Random
    for seed in range(20):
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orderings.append(order)
        
    # 2. Primes
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True
    primes = [x for x in range(2, N + 1) if is_prime(x)]
    non_primes = [x for x in range(1, N + 1) if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # 3. Powers of 2
    p2 = [2**i for i in range(1, 10) if 2**i <= N]
    non_p2 = [x for x in range(1, N + 1) if x not in set(p2)]
    orderings.append(p2 + non_p2)
    
    # 4. Modular
    for mod in [3, 5, 7]:
        mod_order = []
        for r in range(mod):
            mod_order.extend([x for x in range(1, N + 1) if x % mod == r])
        orderings.append(mod_order)
        
    # 5. Ascending/Descending
    orderings.append(list(range(1, N + 1)))
    orderings.append(list(range(N, 0, -1)))
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
            
        S = []
        dp_main = [0] * (MAX_K + 1)
        dp_main[0] = 1
        dp_removed = {}
        
        for x in order:
            if time.time() > deadline - 10:
                break
            if len(S) >= MAX_K:
                break
            
            if is_valid_addition(x, dp_main, dp_removed, S):
                S, dp_main, dp_removed = add_element(x, S, dp_main, dp_removed)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            if best_size == MAX_K:
                return best_set
                
    # Local Search on best_set
    if best_size < MAX_K:
        S = best_set[:]
        # Rebuild DP for S
        dp_main = [0] * (len(S) + 1)
        dp_main[0] = 1
        for val in S:
            for k in range(len(S), 0, -1):
                if dp_main[k-1]:
                    dp_main[k] |= (dp_main[k-1] << val)
                    
        dp_removed = {}
        for y in S:
            others = [v for v in S if v != y]
            dp_y = [0] * (len(others) + 1)
            dp_y[0] = 1
            for val in others:
                for k in range(len(others), 0, -1):
                    if dp_y[k-1]:
                        dp_y[k] |= (dp_y[k-1] << val)
            dp_removed[y] = dp_y
            
        remaining = [x for x in range(1, N + 1) if x not in set(S)]
        rng = random.Random(123)
        rng.shuffle(remaining)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in remaining:
                if time.time() > deadline - 5:
                    break
                if is_valid_addition(x, dp_main, dp_removed, S):
                    S, dp_main, dp_removed = add_element(x, S, dp_main, dp_removed)
                    remaining.remove(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = S[:]
                    if best_size == MAX_K:
                        return best_set
                    break # Restart loop with new state
                    
    return best_set

construct()
