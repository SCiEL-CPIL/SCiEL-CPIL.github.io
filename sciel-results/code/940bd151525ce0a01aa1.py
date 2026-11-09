import time
import random

def construct():
    start = time.time()
    deadline = 115.0  # safety margin before 120s
    
    target = set(range(1, 31))
    
    def get_diffs(S):
        """Return set of all positive differences in sorted set S."""
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs
    
    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        n = len(S)
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all(S):
        """Check if all differences 1..30 are present."""
        diffs = get_diffs(S)
        return target.issubset(diffs)
    
    def cost(S):
        """Cost function: penalize missing diffs, duplicate diffs, and large max."""
        n = len(S)
        expected_diffs = n * (n - 1) // 2
        diffs = get_diffs(S)
        
        # Penalty for duplicate differences
        dup_penalty = expected_diffs - len(diffs)
        
        # Penalty for missing target differences
        missing = target - diffs
        missing_penalty = len(missing) * 10
        
        # Penalty for max element
        max_penalty = max(S)
        
        # Bonus for covering more of the target
        covered = len(target & diffs)
        
        return max_penalty + missing_penalty + dup_penalty * 5 - covered
    
    def is_valid(S):
        return is_sidon(S) and covers_all(S)
    
    best_set = None
    best_max = float('inf')
    
    # Initial greedy construction to get a starting point
    def greedy_construct():
        S = [0]
        diffs = set()
        for _ in range(50):
            if target.issubset(diffs):
                break
            missing = target - diffs
            best_c = None
            best_score = -1
            for d in missing:
                for x in S:
                    for c in (x + d, x - d):
                        if c < 0 or c in S:
                            continue
                        new_S = sorted(S + [c])
                        if not is_sidon(new_S):
                            continue
                        new_diffs = get_diffs(new_S)
                        score = len(target & new_diffs)
                        if score > best_score:
                            best_score = score
                            best_c = c
            if best_c is None:
                break
            S.append(best_c)
            S.sort()
            diffs = get_diffs(S)
        return S
    
    # Try greedy first
    greedy_S = greedy_construct()
    if is_valid(greedy_S):
        best_set = greedy_S[:]
        best_max = max(best_set)
    
    # Simulated Annealing
    def simulated_annealing(initial_S, iterations_per_restart=5000, max_temp=100.0, min_temp=0.01):
        nonlocal best_set, best_max
        S = sorted(initial_S[:])
        T = max_temp
        cooling = (min_temp / max_temp) ** (1.0 / iterations_per_restart)
        
        current_cost = cost(S)
        best_local = S[:]
        best_local_cost = current_cost
        
        for i in range(iterations_per_restart):
            if time.time() - start > deadline:
                break
            
            # Pick a random element to move (not 0)
            idx = random.randint(1, len(S) - 1)
            old_val = S[idx]
            
            # Move by a random amount
            delta = random.randint(-20, 20)
            new_val = old_val + delta
            if new_val <= 0:
                new_val = old_val + random.randint(1, 5)
            if new_val >= 1000000:
                new_val = old_val - random.randint(1, 5)
            
            if new_val in S:
                continue
            
            S[idx] = new_val
            S.sort()
            
            new_cost = cost(S)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / max(T, 0.001)):
                current_cost = new_cost
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local = S[:]
            else:
                S[idx] = old_val
                S.sort()
            
            T *= cooling
        
        # Check if best local is valid
        if is_valid(best_local):
            m = max(best_local)
            if m < best_max:
                best_max = m
                best_set = best_local[:]
        
        return best_local
    
    # Run multiple SA restarts
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() - start > deadline:
            break
        
        if r == 0 and best_set is not None:
            init = best_set[:]
        else:
            # Random initialization
            init = [0] + sorted(random.sample(range(1, 150), 8))
        
        result = simulated_annealing(init, iterations_per_restart=3000)
        if time.time() - start > deadline:
            break
    
    # Local search refinement on best found
    if best_set is not None:
        improved = True
        while improved and time.time() - start < deadline:
            improved = False
            for i in range(1, len(best_set)):
                orig = best_set[i]
                for new_val in range(max(1, orig - 3), orig):
                    if new_val in best_set:
                        continue
                    new_S = sorted([new_val if x == orig else x for x in best_set])
                    if is_valid(new_S) and max(new_S) < max(best_set):
                        best_set = new_S
                        best_max = max(best_set)
                        improved = True
                        break
                if improved:
                    break
    
    # Fallback: if no valid set found, use greedy
    if best_set is None:
        best_set = greedy_S
    
    # Final validation
    if not is_valid(best_set):
        # Emergency fallback: construct a valid set
        S = [0]
        diffs = set()
        for d in range(1, 31):
            if d in diffs:
                continue
            found = False
            for x in S:
                for c in (x + d, x - d):
                    if c < 0 or c in S:
                        continue
                    new_S = sorted(S + [c])
                    if is_sidon(new_S):
                        S = new_S
                        diffs = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add two elements to create difference d
                y = 2 * max(S) + 3 * d + 1
                while True:
                    new_S = sorted(S + [y, y + d])
                    if is_sidon(new_S):
                        S = new_S
                        diffs = get_diffs(S)
                        break
                    y += 1
        best_set = S
    
    return best_set
