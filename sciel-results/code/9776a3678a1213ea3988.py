"""Improved Sidon set construction with backtracking and pruning."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    
    # Helper to check if adding x to current set keeps all pairwise sums distinct
    def can_add(x, current_set, sums_set):
        """Check if x can be added to current_set without creating duplicate pairwise sums."""
        # Check x with each element in current_set
        for y in current_set:
            s = x + y
            if s in sums_set:
                return False
        # Check 2*x (pair of x with itself)
        if 2 * x in sums_set:
            return False
        return True
    
    def get_new_sums(x, current_set):
        """Get the new sums created by adding x."""
        return {x + y for y in current_set} | {2 * x}
    
    # Initial greedy pass to get a decent starting point
    def greedy_extend(base_set):
        """Extend base_set greedily."""
        a = list(base_set)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for x in range(12, 285):
            new_sums = get_new_sums(x, a)
            if len(new_sums) == len(a) + 1 and not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        return a, sums
    
    # Get initial greedy solution
    greedy_set, greedy_sums = greedy_extend(base)
    best_set = list(greedy_set)
    best_size = len(best_set)
    
    # Backtracking search to find larger sets
    def backtrack(current_set, sums_set, target_size, depth=0):
        """Backtracking search to build a Sidon set of target_size."""
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
        
        if len(current_set) >= target_size:
            if len(current_set) > best_size:
                best_set = list(current_set)
                best_size = len(current_set)
            return True
        
        # Try to add more elements
        # Start from the last element + 1 to maintain sorted order
        start_val = current_set[-1] + 1 if current_set else 1
        
        # Pruning: upper bound on how many more elements we can add
        # In [1, 284], max Sidon set size is about 20
        remaining_slots = target_size - len(current_set)
        
        # Try candidates in a smart order
        candidates = list(range(start_val, 285))
        
        # Heuristic: try values that are far from existing elements first
        # to spread out the sums
        def heuristic_score(x):
            min_dist = min(abs(x - y) for y in current_set) if current_set else 0
            return min_dist
        
        # Sort by heuristic (larger distances first) but keep some randomness
        random.shuffle(candidates)
        candidates.sort(key=heuristic_score, reverse=True)
        
        for x in candidates:
            if time.time() > deadline:
                return False
            
            if can_add(x, current_set, sums_set):
                new_sums = get_new_sums(x, current_set)
                current_set.append(x)
                sums_set |= new_sums
                
                # Recursive call
                if backtrack(current_set, sums_set, target_size, depth + 1):
                    return True
                
                # Backtrack
                current_set.pop()
                sums_set -= new_sums
        
        return False
    
    # Try to find larger sets using backtracking
    # Start from greedy size and try to improve
    for target_size in range(best_size + 1, 21):
        if time.time() > deadline:
            break
        
        # Reset random seed for reproducibility within each attempt
        random.seed(42 + target_size)
        
        success = backtrack(list(base), set(), target_size)
        if success:
            best_size = target_size
            # Continue trying for even larger sets
    
    # If backtracking didn't improve, try local search / simulated annealing
    if best_size < 20 and time.time() < deadline - 10:
        def local_search(initial_set):
            """Local search to try to improve the set."""
            nonlocal best_set, best_size
            
            a = list(initial_set)
            sums = set()
            for i, x in enumerate(a):
                for y in a[i:]:
                    sums.add(x + y)
            
            current_size = len(a)
            no_improvement = 0
            max_no_improvement = 50
            
            while time.time() < deadline - 5 and no_improvement < max_no_improvement:
                # Try to add a random element
                added = False
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    x = random.randint(1, 284)
                    if x not in a:
                        if can_add(x, a, sums):
                            new_sums = get_new_sums(x, a)
                            a.append(x)
                            sums |= new_sums
                            a.sort()
                            added = True
                            if len(a) > best_size:
                                best_set = list(a)
                                best_size = len(a)
                            no_improvement = 0
                            break
                
                if not added:
                    no_improvement += 1
                
                # Occasionally try to remove and re-add to escape local optima
                if no_improvement % 10 == 0 and len(a) > 4:
                    # Remove a random non-base element
                    removable = [x for x in a if x not in base]
                    if removable:
                        x_to_remove = random.choice(removable)
                        a.remove(x_to_remove)
                        # Recompute sums
                        sums = set()
                        for i, x in enumerate(a):
                            for y in a[i:]:
                                sums.add(x + y)
        
        local_search(greedy_set)
    
    # Final validation
    def validate(s):
        """Check if s is a valid Sidon set containing base elements."""
        if not all(b in s for b in base):
            return False
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 284 for x in s):
            return False
        sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                if x + y in sums:
                    return False
                sums.add(x + y)
        return True
    
    if not validate(best_set):
        # Fallback to greedy
        best_set = list(greedy_set)
    
    return best_set
