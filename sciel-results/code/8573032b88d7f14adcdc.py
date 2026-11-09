import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    N = 284
    prescribed = [1, 2, 5, 11]
    
    # The maximum size for a Sidon set in [1, 284] is known to be 20.
    # We try to find a set of size 20 containing the prescribed elements.
    # If not found, we try 19, 18, etc.
    
    best_solution = prescribed[:]
    
    def is_valid_solution(sol):
        if not set(prescribed).issubset(set(sol)):
            return False
        if len(sol) != len(set(sol)):
            return False
        if any(x < 1 or x > N for x in sol):
            return False
        sums = set()
        n = len(sol)
        for i in range(n):
            for j in range(i, n):
                s = sol[i] + sol[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def backtrack_search(target_size, time_limit):
        """
        Attempts to find a Sidon set of size `target_size` containing `prescribed`.
        Uses backtracking with increasing order of elements.
        """
        current_set = prescribed[:]
        current_sums = set()
        
        # Pre-compute sums for initial set
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                current_sums.add(current_set[i] + current_set[j])
                
        # Sort current set to maintain order
        current_set.sort()
        
        # We will add elements greater than the last one to avoid permutations
        # This forces the set to be strictly increasing.
        
        solution_found = [None]
        
        def dfs(last_val, depth):
            if time.time() > time_limit:
                return False
            
            if depth == target_size:
                solution_found[0] = current_set[:]
                return True
                
            # Pruning: Can we possibly reach target_size?
            # Max value is N. We need (target_size - depth) more elements.
            # The smallest possible next elements are last_val+1, last_val+2, ...
            # So the last element would be at least last_val + (target_size - depth)
            if last_val + (target_size - depth) > N:
                return False
                
            # Candidate range
            # Start from last_val + 1
            # We can also use a heuristic: maybe start from a higher value to spread out?
            # Standard Sidon construction often benefits from spreading.
            # Let's iterate from last_val + 1 to N.
            
            for x in range(last_val + 1, N + 1):
                # Check validity of x
                valid = True
                # Check 2x
                if 2 * x in current_sums:
                    valid = False
                    if not valid:
                        continue
                
                # Check x + s for all s in current_set
                if valid:
                    for s in current_set:
                        if (x + s) in current_sums:
                            valid = False
                            break
                
                if not valid:
                    continue
                    
                # Add x
                current_set.append(x)
                new_sums = set()
                for s in current_set[:-1]: # All previous elements
                    new_sums.add(x + s)
                new_sums.add(2 * x)
                current_sums.update(new_sums)
                
                if dfs(x, depth + 1):
                    return True
                    
                # Backtrack
                current_set.pop()
                for s in new_sums:
                    current_sums.remove(s)
                    
            return False

        dfs(current_set[-1], len(current_set))
        return solution_found[0]

    # Try sizes from 20 down to 5
    # We give more time to larger sizes initially, or just a fixed time per attempt?
    # Let's give 15s for size 20, 10s for 19, 5s for 18, etc.
    # If we find one, we stop and return it.
    
    # To handle cases where simple increasing backtracking gets stuck in bad branches,
    # we can use randomized restarts or randomize the candidate order.
    # However, for N=284, the increasing order is usually effective if we start with the right base.
    
    # Let's try a few different strategies for the search order if simple backtracking fails.
    
    # Strategy 1: Standard increasing backtracking
    for target_size in range(20, 4, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on size
        if target_size == 20:
            t_limit = min(deadline - 10, time.time() + 20.0)
        elif target_size == 19:
            t_limit = min(deadline - 5, time.time() + 15.0)
        else:
            t_limit = min(deadline - 2, time.time() + 5.0)
            
        sol = backtrack_search(target_size, t_limit)
        if sol:
            if is_valid_solution(sol):
                return sol
                
    # If standard backtracking failed (likely due to getting stuck in a local trap or just slow),
    # we try a randomized backtracking approach.
    
    def randomized_backtrack_search(target_size, time_limit, seed):
        random.seed(seed)
        current_set = prescribed[:]
        current_sums = set()
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                current_sums.add(current_set[i] + current_set[j])
        current_set.sort()
        
        solution_found = [None]
        
        def dfs(last_val, depth):
            if time.time() > time_limit:
                return False
            
            if depth == target_size:
                solution_found[0] = current_set[:]
                return True
                
            if last_val + (target_size - depth) > N:
                return False
                
            # Generate candidates
            candidates = list(range(last_val + 1, N + 1))
            # Shuffle to avoid systematic bias
            random.shuffle(candidates)
            
            for x in candidates:
                valid = True
                if 2 * x in current_sums:
                    continue
                for s in current_set:
                    if (x + s) in current_sums:
                        valid = False
                        break
                if not valid:
                    continue
                
                current_set.append(x)
                new_sums = set()
                for s in current_set[:-1]:
                    new_sums.add(x + s)
                new_sums.add(2 * x)
                current_sums.update(new_sums)
                
                if dfs(x, depth + 1):
                    return True
                    
                current_set.pop()
                for s in new_sums:
                    current_sums.remove(s)
                    
            return False

        dfs(current_set[-1], len(current_set))
        return solution_found[0]

    # Try randomized restarts for size 20 and 19 if not found
    if time.time() < deadline - 5:
        for target_size in [20, 19]:
            if time.time() > deadline - 5:
                break
            # Try a few random seeds
            for seed in range(10):
                if time.time() > deadline - 5:
                    break
                t_limit = min(deadline - 5, time.time() + 5.0)
                sol = randomized_backtrack_search(target_size, t_limit, seed)
                if sol and is_valid_solution(sol):
                    return sol
                    
    return best_solution
