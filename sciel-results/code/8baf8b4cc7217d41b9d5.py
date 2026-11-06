"""Optimized Sidon set construction with randomized search."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def is_sidon(a):
        """Check if sorted list a is a Sidon set (all positive differences distinct)."""
        n = len(a)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in seen:
                    return False
                seen.add(d)
        return True

    def diffs_of(a):
        """Return set of all positive differences."""
        n = len(a)
        return set(a[j] - a[i] for i in range(n) for j in range(i + 1, n))

    def valid(a):
        """Check if a is a valid Sidon set covering 1..30."""
        if not is_sidon(a):
            return False
        d = diffs_of(a)
        return all(x in d for x in range(1, 31))

    # Start with a good initial solution from greedy
    def greedy_construct():
        a = [0]
        for d in range(1, 31):
            if d in diffs_of(a):
                continue
            found = False
            # Try adding element to create difference d
            for v in a:
                for x in [v + d, v - d]:
                    if x < 0 or x in a:
                        continue
                    trial = sorted(a + [x])
                    if is_sidon(trial):
                        a = trial
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try brute force for this difference
                max_a = max(a)
                for x in range(1, max_a + d + 200):
                    if x in a:
                        continue
                    trial = sorted(a + [x])
                    if is_sidon(trial) and d in diffs_of(trial):
                        a = trial
                        found = True
                        break
                if not found:
                    # Add two elements
                    y = 2 * max_a + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if is_sidon(trial):
                            a = trial
                            break
                        y += 1
        return a

    best = greedy_construct()
    best_max = max(best)

    # Try to reduce size if possible (we may have more than 9 elements)
    # Try removing elements while maintaining validity
    while len(best) > 9:
        improved = False
        for i in range(len(best)):
            trial = best[:i] + best[i+1:]
            if valid(trial):
                best = trial
                best_max = max(best)
                improved = True
                break
        if not improved:
            break

    # Randomized local search to minimize max element
    random.seed(12345)
    
    def local_search(current, max_iters=5000):
        """Try to improve current solution by swapping elements."""
        curr = sorted(current)
        curr_max = max(curr)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try swapping a random element with a smaller value
            i = random.randint(1, len(curr) - 1)  # don't swap 0
            old_val = curr[i]
            
            # Try replacing with a smaller random value
            if old_val > 1:
                new_val = random.randint(1, old_val - 1)
                if new_val not in curr:
                    trial = sorted(curr[:i] + [new_val] + curr[i+1:])
                    if valid(trial):
                        if max(trial) < curr_max:
                            curr = trial
                            curr_max = max(curr)
            
            # Try swapping two non-zero elements
            if len(curr) >= 3:
                i = random.randint(1, len(curr) - 1)
                j = random.randint(1, len(curr) - 1)
                if i != j:
                    trial = list(curr)
                    trial[i], trial[j] = trial[j], trial[i]
                    trial = sorted(trial)
                    if valid(trial):
                        if max(trial) < curr_max:
                            curr = trial
                            curr_max = max(curr)
        
        return curr, curr_max

    # Multiple restarts with perturbation
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Perturb the best solution
        curr = list(best)
        # Randomly perturb a few elements
        for _ in range(random.randint(1, 3)):
            i = random.randint(1, len(curr) - 1)
            old = curr[i]
            new = random.randint(1, old + 20)
            if new not in curr:
                curr[i] = new
        curr = sorted(curr)
        
        if not valid(curr):
            # Try to fix
            fixed = False
            for _ in range(100):
                # Try adjusting elements
                i = random.randint(1, len(curr) - 1)
                old = curr[i]
                for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                    new = old + delta
                    if new <= 0 or new in curr:
                        continue
                    trial = sorted(curr[:i] + [new] + curr[i+1:])
                    if valid(trial):
                        curr = trial
                        fixed = True
                        break
                if fixed:
                    break
            if not valid(curr):
                continue
        
        # Local search
        curr, curr_max = local_search(curr, max_iters=2000)
        
        if curr_max < best_max:
            best = curr
            best_max = curr_max

    # Final attempt: try to construct a 9-element set directly with small max
    if time.time() < deadline - 20:
        # Try systematic search for 9-element Sidon sets with small max
        # Use backtracking
        def backtrack_search(target_size=9, max_bound=200):
            """Backtracking search for a Sidon set of given size covering 1..30."""
            if time.time() > deadline - 5:
                return None
            
            def solve(a, remaining_diffs, max_allowed):
                if time.time() > deadline - 5:
                    return None
                
                if len(a) == target_size:
                    if all(d in diffs_of(a) for d in range(1, 31)):
                        return a
                    return None
                
                if not a:
                    a = [0]
                    remaining = set(range(1, 31))
                else:
                    remaining = set(range(1, 31)) - diffs_of(a)
                
                if not remaining:
                    # Need to fill up to target_size while maintaining Sidon
                    pass
                
                # Try adding elements
                max_a = max(a) if a else 0
                candidates = list(range(1, max_allowed + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    if x in a:
                        continue
                    trial = sorted(a + [x])
                    if is_sidon(trial):
                        result = solve(trial, remaining, max_allowed)
                        if result is not None:
                            return result
                
                return None
            
            # Try increasing bounds
            for bound in range(50, max_bound + 1, 10):
                result = solve([0], set(range(1, 31)), bound)
                if result is not None and max(result) < best_max:
                    return result
            
            return None
        
        result = backtrack_search()
        if result is not None:
            best = result
            best_max = max(best)

    return sorted(best)

# EVOLVE-BLOCK-END
