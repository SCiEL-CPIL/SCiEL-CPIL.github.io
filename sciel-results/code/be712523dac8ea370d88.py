import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_size = N + 1
    best_set = None
    
    def can_add(S, sum_set, x):
        """
        Check if x can be added to Sidon set S (with precomputed sum_set).
        """
        if 2 * x in sum_set:
            return False
        for s in S:
            if x + s in sum_set:
                return False
        return True

    def is_maximal(S, sum_set):
        """
        Check if Sidon set S is inclusion-maximal in [1, N].
        S is maximal if no y in [1, N] \ S can be added.
        """
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            if can_add(S, sum_set, y):
                return False
        return True

    def generate_random_maximal():
        """
        Generate a random inclusion-maximal Sidon set.
        """
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        S = []
        sum_set = set()
        for x in perm:
            if can_add(S, sum_set, x):
                S.append(x)
                sum_set.add(2 * x)
                for s in S[:-1]:
                    sum_set.add(x + s)
        return S, sum_set

    # Main search loop
    while time.time() < deadline:
        S, sum_set = generate_random_maximal()
        
        # Local search: attempt to reduce size while maintaining maximality
        improved = True
        while improved:
            improved = False
            # Shuffle current set to try different removal orders
            S_shuffled = S[:]
            random.shuffle(S_shuffled)
            
            for x in S_shuffled:
                # Candidate set without x
                S_try = [s for s in S if s != x]
                
                # Compute new sum set
                # Remove sums that involved x
                # Sums to remove: {2*x} U {x+s | s in S_try}
                sums_to_remove = {2 * x}
                for s in S_try:
                    sums_to_remove.add(x + s)
                
                sum_set_try = sum_set.copy()
                for s_val in sums_to_remove:
                    sum_set_try.discard(s_val)
                
                # Check if the reduced set is still maximal
                if is_maximal(S_try, sum_set_try):
                    # Accept the removal
                    S = S_try
                    sum_set = sum_set_try
                    improved = True
                    # Break to restart the shuffle/pass with the new set
                    break
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
