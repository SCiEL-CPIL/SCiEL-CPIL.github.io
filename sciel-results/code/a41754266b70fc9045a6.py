import time
import random

N = 256

def construct():
    """
    Returns an inclusion-maximal Sidon set in [1, 256] with minimal cardinality.
    """
    deadline = time.time() + 28.0
    
    def can_add(x, S, sums):
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        return True

    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    def is_maximal(S):
        if not S:
            return False
        S_set = set(S)
        sums = get_sums(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add(x, S, sums):
                return False
        return True

    def search_maximal_k(k, time_limit):
        """
        Attempt to find a maximal Sidon set of size k using backtracking.
        """
        start_time = time.time()
        
        def backtrack(current_S, current_sums, depth):
            if time.time() > start_time + time_limit:
                return None
            
            if depth == k:
                if is_maximal(current_S):
                    return list(current_S)
                return None
            
            min_val = current_S[-1] + 1 if current_S else 1
            candidates = []
            for x in range(min_val, N + 1):
                if can_add(x, current_S, current_sums):
                    candidates.append(x)
            
            if not candidates:
                return None
                
            random.shuffle(candidates)
            
            for x in candidates:
                new_S = current_S + [x]
                new_sums = set(current_sums)
                new_sums.add(2 * x)
                for a in current_S:
                    new_sums.add(x + a)
                
                result = backtrack(new_S, new_sums, depth + 1)
                if result:
                    return result
                    
            return None

        return backtrack([], set(), 0)

    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            if can_add(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S

    def prune_aggressive(S):
        S = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline:
                    break
                new_S = S[:idx] + S[idx+1:]
                if not new_S:
                    continue
                if is_maximal(new_S):
                    S = new_S
                    improved = True
                    break
        return S

    random.seed(42)
    
    # Phase 1: Try to find a very small set via backtracking
    # Start from k=10
    for k in range(10, 13):
        if time.time() > deadline - 5.0:
            break
        # Give 2 seconds to each k
        found = search_maximal_k(k, 2.0)
        if found:
            if is_maximal(found):
                return found

    # Phase 2: Multi-start Greedy + Prune
    best_set = None
    best_size = N + 1
    
    # Use remaining time for multi-start
    while time.time() < deadline - 2.0:
        # Random permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_construct(order)
        S = prune_aggressive(S)
        
        if S:
            if len(S) < best_size:
                best_size = len(S)
                best_set = S
        
        # If we found a set of size 10 or 11, we can stop?
        # We already tried backtracking for 10-12.
        # If greedy finds 11, it's good.
        if best_size <= 11:
            break

    if best_set:
        return best_set
    
    # Fallback
    order = list(range(N, 0, -1))
    return greedy_construct(order)

# EVOLVE-BLOCK-END
