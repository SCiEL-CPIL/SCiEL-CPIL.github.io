"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def time_up():
        return elapsed() > deadline
    
    def get_diffs(s):
        """Get all positive differences from sorted set s"""
        diffs = []
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.append(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if set is Sidon (all positive differences distinct)"""
        diffs = get_diffs(s)
        return len(set(diffs)) == len(diffs)
    
    def covers_all(s):
        """Check if all differences 1-30 are covered"""
        diffs = set(get_diffs(s))
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    def missing_diffs(s):
        """Get which differences 1-30 are missing"""
        diffs = set(get_diffs(s))
        return [d for d in range(1, 31) if d not in diffs]
    
    def diameter(s):
        return max(s) - min(s)
    
    # Phase 1: Greedy construction to get initial solution
    def greedy_construct():
        a = [0]
        covered = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            
            # Try to find a new element that covers d and maintains Sidon
            found = False
            # Try adding element near existing ones
            for existing in sorted(a):
                for candidate in [existing + d, existing - d]:
                    if candidate < 0 or candidate in a:
                        continue
                    trial = sorted(a + [candidate])
                    if is_sidon(trial):
                        new_diffs = set(get_diffs(trial))
                        if len(new_diffs & set(range(1, 31))) > len(covered & set(range(1, 31))):
                            a = trial
                            covered = new_diffs & set(range(1, 31))
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # Add element far away
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        covered = set(get_diffs(trial)) & set(range(1, 31))
                        break
                    y += 1
        
        return a
    
    # Phase 2: Local search to improve
    def local_search(s, max_iterations=50000):
        best = list(s)
        best_diam = diameter(best)
        
        for _ in range(max_iterations):
            if time_up():
                break
            
            # Try to remove an element and re-add to improve
            # Or try swapping elements
            improved = False
            
            # Try removing each element and see if we can do better
            for i in range(len(best)):
                candidate = best[:i] + best[i+1:]
                if len(candidate) < 2:
                    continue
                if not is_sidon(candidate):
                    continue
                if not covers_all(candidate):
                    continue
                if diameter(candidate) < best_diam:
                    best = candidate
                    best_diam = diameter(best)
                    improved = True
                    break
            
            if not improved:
                # Try adding a new element and removing another
                max_val = max(best)
                for new_val in range(1, max_val + 10):
                    if new_val in best:
                        continue
                    trial = sorted(best + [new_val])
                    if not is_sidon(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    # Try removing one element
                    for i in range(len(trial)):
                        candidate = trial[:i] + trial[i+1:]
                        if len(candidate) < 2:
                            continue
                        if not is_sidon(candidate):
                            continue
                        if not covers_all(candidate):
                            continue
                        if diameter(candidate) < best_diam:
                            best = candidate
                            best_diam = diameter(candidate)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try random perturbation
                if random.random() < 0.1:
                    i = random.randint(0, len(best) - 1)
                    old_val = best[i]
                    for delta in [1, -1, 2, -2, 3, -3]:
                        new_val = old_val + delta
                        if new_val < 0 or new_val in best:
                            continue
                        trial = best[:i] + [new_val] + best[i+1:]
                        trial = sorted(trial)
                        if is_sidon(trial) and covers_all(trial):
                            if diameter(trial) < best_diam:
                                best = trial
                                best_diam = diameter(trial)
                                improved = True
                                break
                if not improved and random.random() < 0.01:
                    break
        
        return best
    
    # Phase 3: More aggressive search with different starting points
    def diverse_search():
        best_solution = greedy_construct()
        best_diam = diameter(best_solution)
        
        # Try multiple random restarts
        for restart in range(20):
            if time_up():
                break
            
            # Create a perturbed starting point
            a = [0]
            random.shuffle(list(range(1, 31)))
            for d in range(1, 31):
                if d in set(get_diffs(a)):
                    continue
                found = False
                for existing in sorted(a):
                    for candidate in [existing + d, existing - d]:
                        if candidate < 0 or candidate in a:
                            continue
                        trial = sorted(a + [candidate])
                        if is_sidon(trial):
                            a = trial
                            found = True
                            break
                    if found:
                        break
                if not found:
                    y = 2 * max(a) + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if is_sidon(trial):
                            a = trial
                            break
                        y += 1
            
            if covers_all(a):
                a = local_search(a, max_iterations=10000)
                if diameter(a) < best_diam:
                    best_solution = a
                    best_diam = diameter(a)
        
        return best_solution
    
    # Run the search
    solution = greedy_construct()
    solution = local_search(solution, max_iterations=100000)
    
    if not time_up():
        solution2 = diverse_search()
        if diameter(solution2) < diameter(solution):
            solution = solution2
    
    # Final verification
    solution = sorted(set(solution))
    if not is_sidon(solution) or not covers_all(solution):
        # Fallback to greedy
        solution = greedy_construct()
    
    return solution
# EVOLVE-BLOCK-END
