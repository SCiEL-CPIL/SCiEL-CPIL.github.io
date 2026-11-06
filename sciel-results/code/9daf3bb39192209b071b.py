import time
import random

N = 256

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    def compute_sums(S):
        """Compute all pairwise sums a+b for a<=b in S."""
        s = set()
        for i in range(len(S)):
            s.add(2 * S[i])
            for j in range(i + 1, len(S)):
                s.add(S[i] + S[j])
        return s
    
    def is_valid_add(x, S, sums):
        """Check if x can be added to Sidon set S."""
        if 2 * x in sums:
            return False
        for s in S:
            if x + s in sums:
                return False
        return True
    
    def is_maximal(S, sums):
        """Check if S is inclusion-maximal: no element from [1,N]\S can be added."""
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            if is_valid_add(y, S, sums):
                return False
        return True
    
    def local_search_reduce(S, sums):
        """Try to remove elements while maintaining maximality.
        Returns a smaller maximal Sidon set if found."""
        S = list(S)
        sums = compute_sums(S)
        improved = True
        while improved:
            improved = False
            # Try removing each element
            for i in range(len(S)):
                # Temporarily remove S[i]
                S_temp = S[:i] + S[i+1:]
                sums_temp = compute_sums(S_temp)
                if is_maximal(S_temp, sums_temp):
                    S = S_temp
                    sums = sums_temp
                    improved = True
                    break
        return S, sums
    
    def greedy_construct(order):
        """Greedy construct Sidon set following given order."""
        S = []
        sums = set()
        for x in order:
            if is_valid_add(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for s in S[:-1]:
                    sums.add(x + s)
        return S, sums
    
    # Strategy: Multiple approaches
    # 1. Random greedy with local search reduction
    # 2. Greedy with prioritized blocking (elements that block many others first)
    # 3. Random restarts with different strategies
    
    for iteration in range(5000):
        if time.time() >= deadline:
            break
        
        strategy = iteration % 3
        
        if strategy == 0:
            # Random order greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy == 1:
            # Prioritize elements that create many conflicts (high blocking power)
            # Elements near N/2 create more distinct sums
            order = list(range(1, N + 1))
            # Sort by "blocking potential" - elements that conflict with many others
            # Approximate: elements in middle range block more
            random.shuffle(order)
            # But bias towards middle values
            order.sort(key=lambda x: -abs(x - N/2) + random.random() * 50)
        else:
            # Hybrid: start with some structured elements, then random
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Put some spread-out elements first
            spread = [1, N//4, N//2, 3*N//4, N]
            for s in spread:
                if s in order:
                    order.remove(s)
            order = spread + order
        
        S, sums = greedy_construct(order)
        
        if len(S) < best_size and is_maximal(S, sums):
            # Try to reduce further
            S_reduced, sums_reduced = local_search_reduce(S, sums)
            if len(S_reduced) < best_size:
                best_size = len(S_reduced)
                best_set = S_reduced[:]
        
        # Also try: construct a maximal set, then aggressively reduce
        if len(S) >= 15 and time.time() < deadline:
            # Check if current S is maximal
            if is_maximal(S, sums):
                S_reduced, sums_reduced = local_search_reduce(S, sums)
                if len(S_reduced) < best_size:
                    best_size = len(S_reduced)
                    best_set = S_reduced[:]
    
    # Final attempt: try to reduce the best set found further
    if best_set:
        sums_best = compute_sums(best_set)
        S_final, sums_final = local_search_reduce(best_set, sums_best)
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = S_final[:]
    
    return best_set
