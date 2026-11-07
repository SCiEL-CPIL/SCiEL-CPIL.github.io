import time
import random
from functools import lru_cache

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s kill
    
    def compute_sums(s):
        """Compute all subset sums indexed by cardinality."""
        dp = [set() for _ in range(len(s) + 1)]
        dp[0].add(0)
        for x in s:
            for c in range(len(s) - 1, -1, -1):
                if dp[c]:
                    dp[c + 1].update(sv + x for sv in dp[c])
        return dp
    
    def is_valid(s):
        """Check if subset sums of different cardinalities are disjoint."""
        dp = compute_sums(s)
        all_sums = set()
        for c in range(len(s) + 1):
            if not all_sums.isdisjoint(dp[c]):
                return False
            all_sums.update(dp[c])
        return True
    
    def count_conflicts(s, x):
        """Count how many new conflicts adding x would create."""
        dp = compute_sums(s)
        new_dp = [set() for _ in range(len(s) + 2)]
        for c in range(len(s) + 1):
            new_dp[c] = set(dp[c])
        for c in range(len(s), -1, -1):
            for sv in dp[c]:
                new_dp[c + 1].add(sv + x)
        
        # Count conflicts: sums in new_dp that appear in multiple cardinalities
        all_sums = {}
        for c in range(len(s) + 2):
            for sv in new_dp[c]:
                if sv in all_sums:
                    all_sums[sv].append(c)
                else:
                    all_sums[sv] = [c]
        
        conflicts = 0
        for sv, cs in all_sums.items():
            if len(cs) > 1:
                conflicts += 1
        return conflicts
    
    # Backtracking search with conflict-based ordering
    def backtrack_search(current_set, max_size, time_limit):
        """Search for a valid set of size max_size extending current_set."""
        if time.time() > time_limit:
            return None
        if len(current_set) == max_size:
            return current_set[:]
        
        # Generate candidates: all numbers 1..256 not in current_set
        current_set_sorted = sorted(current_set)
        used = set(current_set)
        candidates = [x for x in range(1, 257) if x not in used]
        
        if not candidates:
            return None
        
        # Order candidates by conflict count (fewest conflicts first)
        # This is expensive, so we limit to a subset for larger sets
        if len(current_set) < 6:
            scored_candidates = []
            for x in candidates:
                conflicts = count_conflicts(current_set, x)
                scored_candidates.append((conflicts, x))
            scored_candidates.sort()
            ordered_candidates = [x for _, x in scored_candidates]
        else:
            # For larger sets, just use natural order but try smaller first
            ordered_candidates = sorted(candidates)
        
        for x in ordered_candidates:
            new_set = current_set + [x]
            if is_valid(new_set):
                result = backtrack_search(new_set, max_size, time_limit)
                if result is not None:
                    return result
        return None
    
    # Start with known valid set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    best_size = len(best_set)
    
    # Try to find larger sets using backtracking
    # Try sizes 9, 10, 11, ...
    for target_size in range(best_size + 1, 15):
        if time.time() > deadline - 10:
            break
        
        # Try multiple starting points
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Start from best known set
                initial_set = best_set[:]
            else:
                # Try random initial sets of size target_size - 5
                rng = random.Random(attempt * 1000 + target_size)
                initial_size = min(target_size - 5, 5)
                initial_set = rng.sample(range(1, 257), initial_size)
                if not is_valid(initial_set):
                    continue
            
            result = backtrack_search(initial_set, target_size, deadline - 5)
            if result is not None:
                best_set = result
                best_size = target_size
                found = True
                break
        
        if not found:
            # If we couldn't extend, try a different approach
            # Randomized local search
            if time.time() > deadline - 5:
                break
            
            # Try random restarts with greedy + local search
            for rs in range(50):
                if time.time() > deadline - 3:
                    break
                
                rng = random.Random(rs * 7919 + target_size * 13)
                s = []
                # Greedy build with randomness
                for _ in range(target_size):
                    valid_candidates = []
                    for x in range(1, 257):
                        if x in s:
                            continue
                        new_s = s + [x]
                        if is_valid(new_s):
                            valid_candidates.append(x)
                    if not valid_candidates:
                        break
                    # Pick with bias towards smaller
                    if rng.random() < 0.7:
                        x = min(valid_candidates)
                    else:
                        x = rng.choice(valid_candidates)
                    s.append(x)
                
                if len(s) > best_size:
                    best_set = s
                    best_size = len(s)
                
                # Local search: try to swap elements to improve
                if len(s) == target_size:
                    for _ in range(100):
                        if time.time() > deadline - 3:
                            break
                        i = rng.randrange(len(s))
                        old_x = s[i]
                        candidates = [x for x in range(1, 257) if x != old_x and x not in s]
                        if not candidates:
                            continue
                        new_x = rng.choice(candidates)
                        s[i] = new_x
                        if is_valid(s):
                            # Keep it
                            pass
                        else:
                            s[i] = old_x
    
    # Final validation
    if not is_valid(best_set):
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return [int(x) for x in best_set]
