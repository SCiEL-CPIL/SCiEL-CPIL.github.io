import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def get_subset_sums(S):
        """Get all non-empty subset sums of S."""
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums |= new_sums
        return sums

    def is_valid(S):
        """Check if all non-empty subset sums are divisibility-incomparable."""
        sums = get_subset_sums(S)
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    def is_valid_with_sums(sums):
        """Check divisibility-incomparability given a set of sums."""
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b < 2 * a:
                    continue
                if b % a == 0:
                    return False
        return True

    def add_element_check(current_sums, x):
        """Check if adding x keeps validity; return new sums or None."""
        new_sums = {x}
        for s in current_sums:
            new_sums.add(x + s)
        all_sums = current_sums | new_sums
        if is_valid_with_sums(all_sums):
            return all_sums
        return None

    # Start with a known valid set
    best = [128, 129, 131]
    best_size = 3
    best_sums = get_subset_sums(best)

    # Candidate pool: large numbers to minimize divisibility conflicts
    # Numbers in [128, 256]
    candidates = list(range(128, 257))

    # Backtracking search with pruning
    def backtrack(current_set, current_sums, candidates_list, depth=0):
        nonlocal best, best_size, best_sums
        if time.time() > deadline - 2.0:
            return
        
        if len(current_set) > best_size:
            best = list(current_set)
            best_size = len(best)
            best_sums = set(current_sums)
        
        # If we've used all candidates or can't improve, stop
        if not candidates_list:
            return
        
        # Try adding each candidate
        for i, x in enumerate(candidates_list):
            if time.time() > deadline - 2.0:
                return
            
            new_sums = add_element_check(current_sums, x)
            if new_sums is not None:
                backtrack(current_set + [x], new_sums, candidates_list[i+1:], depth+1)
            
            # If we have a good set, try to add more from remaining
            if len(current_set) >= best_size - 1:
                # Try to find one more element that fits
                for j in range(i+1, len(candidates_list)):
                    if time.time() > deadline - 2.0:
                        return
                    x2 = candidates_list[j]
                    new_sums2 = add_element_check(current_sums, x2)
                    if new_sums2 is not None:
                        backtrack(current_set + [x2], new_sums2, candidates_list[j+1:], depth+1)

    # Strategy 1: Greedy with multiple orderings
    for order in ['desc', 'asc', 'random']:
        if time.time() > deadline - 5.0:
            break
        
        cand = list(candidates)
        if order == 'desc':
            cand.sort(reverse=True)
        elif order == 'asc':
            cand.sort()
        else:
            random.seed(42)
            random.shuffle(cand)
        
        current_set = []
        current_sums = set()
        
        for x in cand:
            if time.time() > deadline - 5.0:
                break
            new_sums = add_element_check(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best = list(current_set)
            best_size = len(best)
            best_sums = set(current_sums)

    # Strategy 2: Randomized restarts with different starting points
    random.seed(123)
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Start with 2-3 random large numbers
        start_size = random.randint(2, 4)
        start_set = random.sample(candidates, start_size)
        start_set.sort()
        
        start_sums = get_subset_sums(start_set)
        if not is_valid_with_sums(start_sums):
            continue
        
        # Try to extend
        remaining = [x for x in candidates if x not in start_set]
        random.shuffle(remaining)
        
        current_set = list(start_set)
        current_sums = set(start_sums)
        
        for x in remaining:
            if time.time() > deadline - 5.0:
                break
            new_sums = add_element_check(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best = list(current_set)
            best_size = len(best)
            best_sums = set(current_sums)

    # Strategy 3: Local search - try swapping elements to enable growth
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_set = list(best)
            best_sums = get_subset_sums(best_set)
            
            for i in range(len(best_set)):
                if time.time() > deadline - 10.0:
                    break
                
                for x in candidates:
                    if x in best_set:
                        continue
                    
                    new_set = best_set[:i] + [x] + best_set[i+1:]
                    new_sums = get_subset_sums(new_set)
                    if not is_valid_with_sums(new_sums):
                        continue
                    
                    # Try to extend from this swapped set
                    remaining = [y for y in candidates if y not in new_set]
                    random.shuffle(remaining)
                    
                    current_set = list(new_set)
                    current_sums = set(new_sums)
                    
                    for y in remaining:
                        if time.time() > deadline - 10.0:
                            break
                        new_sums_y = add_element_check(current_sums, y)
                        if new_sums_y is not None:
                            current_set.append(y)
                            current_sums = new_sums_y
                    
                    if len(current_set) > best_size:
                        best = list(current_set)
                        best_size = len(best)
                        best_sums = set(current_sums)
                        improved = True
                        break
                
                if improved:
                    break

    # Strategy 4: Focused search on promising ranges
    # Numbers close together have subset sums that are also close, reducing divisibility risk
    for base in range(128, 250, 5):
        if time.time() > deadline - 3.0:
            break
        
        # Try sets of numbers in a narrow window
        window = list(range(base, min(base + 20, 257)))
        
        # Greedy from this window
        current_set = []
        current_sums = set()
        
        for x in window:
            if time.time() > deadline - 3.0:
                break
            new_sums = add_element_check(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best = list(current_set)
            best_size = len(best)
            best_sums = set(current_sums)

    # Final verification
    if not is_valid(best):
        best = [128, 129, 131]

    return list(best)
