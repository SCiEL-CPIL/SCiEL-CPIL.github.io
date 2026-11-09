import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill

    target_diffs = set(range(1, 31))

    def get_diffs(S):
        """Get all positive differences from sorted set S"""
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)"""
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2

    def compute_energy(S, target_diffs):
        """Energy = max(S) + 100 * num_missing_diffs + 1000 * num_conflicts"""
        if not S:
            return float('inf')
        S_sorted = sorted(S)
        n = len(S_sorted)
        diffs = get_diffs(S_sorted)
        missing = target_diffs - diffs
        # Count conflicts: differences that appear more than once
        from collections import Counter
        diff_counts = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                diff_counts[S_sorted[j] - S_sorted[i]] += 1
        conflicts = sum(1 for v in diff_counts.values() if v > 1)
        energy = max(S_sorted) + 100 * len(missing) + 1000 * conflicts
        return energy

    def is_valid(S, target_diffs):
        """Check if S is a valid Sidon set covering all target differences"""
        if not S or len(S) > 32:
            return False
        S_sorted = sorted(S)
        if 0 not in S_sorted:
            return False
        if not is_sidon(S_sorted):
            return False
        diffs = get_diffs(S_sorted)
        return target_diffs.issubset(diffs)

    def greedy_construct(rng, max_steps=200):
        """Greedy construction of a valid Sidon set"""
        S = [0]
        covered = set()
        used_diffs = set()
        
        for step in range(max_steps):
            missing = target_diffs - covered
            if not missing:
                break
            
            best_x = None
            best_coverage = -1
            best_diffs = None
            
            for d in missing:
                for s in S:
                    x = s + d
                    if x <= max(S) or x > 10**6:
                        continue
                    
                    new_diffs = set()
                    valid = True
                    for t in S:
                        diff = x - t
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    
                    if valid:
                        coverage = len((covered | new_diffs) & target_diffs)
                        if coverage > best_coverage or (coverage == best_coverage and (best_x is None or x < best_x)):
                            best_coverage = coverage
                            best_x = x
                            best_diffs = new_diffs
            
            if best_x is None:
                # Try larger values
                found = False
                base = max(S) + 1
                for offset in range(1, 100):
                    x = base + offset
                    new_diffs = set()
                    valid = True
                    for t in S:
                        diff = x - t
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    
                    if valid:
                        S.append(x)
                        S.sort()
                        covered = get_diffs(S)
                        used_diffs = get_diffs(S)
                        found = True
                        break
                
                if not found:
                    break
            else:
                S.append(best_x)
                S.sort()
                covered = get_diffs(S)
                used_diffs = get_diffs(S)
        
        return S if is_valid(S, target_diffs) else None

    def simulated_annealing(initial_S, target_diffs, time_budget):
        """Simulated annealing to minimize max element"""
        rng = random.Random(42)
        S = list(initial_S)
        best_S = S[:]
        best_energy = compute_energy(S, target_diffs)
        best_max = max(S)
        
        T = 500.0  # Initial temperature
        cooling = 0.9995
        min_T = 0.1
        
        while T > min_T and time.time() - start_time < deadline - 5:
            S_copy = S[:]
            
            # Propose a move
            move_type = rng.random()
            if move_type < 0.3 and len(S_copy) > 2:
                # Remove an element (not 0)
                non_zero = [i for i, x in enumerate(S_copy) if x != 0]
                if non_zero:
                    idx = non_zero[rng.randint(0, len(non_zero) - 1)]
                    S_copy.pop(idx)
            elif move_type < 0.6:
                # Swap: remove one non-zero element, add another
                non_zero = [i for i, x in enumerate(S_copy) if x != 0]
                if non_zero:
                    idx = non_zero[rng.randint(0, len(non_zero) - 1)]
                    removed = S_copy.pop(idx)
                    # Try to add a new element
                    max_val = max(S_copy) if S_copy else 0
                    candidates = []
                    for d in target_diffs:
                        for s in S_copy:
                            x = s + d
                            if x > max_val and x <= 10**6:
                                candidates.append(x)
                            x2 = s - d
                            if x2 > 0 and x2 <= max_val:
                                candidates.append(x2)
                    if not candidates:
                        candidates = list(range(1, max_val + 1))
                    if candidates:
                        new_val = candidates[rng.randint(0, len(candidates) - 1)]
                        if new_val not in S_copy:
                            S_copy.append(new_val)
                            S_copy.sort()
            elif move_type < 0.8:
                # Add a new element
                if len(S_copy) < 32:
                    max_val = max(S_copy)
                    candidates = []
                    for d in target_diffs:
                        for s in S_copy:
                            x = s + d
                            if x > max_val and x <= 10**6:
                                candidates.append(x)
                    if not candidates:
                        candidates = list(range(max_val + 1, max_val + 50))
                    if candidates:
                        new_val = candidates[rng.randint(0, len(candidates) - 1)]
                        if new_val not in S_copy:
                            S_copy.append(new_val)
                            S_copy.sort()
            else:
                # Move an element to a nearby position
                non_zero = [i for i, x in enumerate(S_copy) if x != 0]
                if non_zero:
                    idx = non_zero[rng.randint(0, len(non_zero) - 1)]
                    old_val = S_copy[idx]
                    delta = rng.randint(-5, 5)
                    new_val = old_val + delta
                    if new_val > 0 and new_val != old_val:
                        # Check if new_val is not already in S_copy
                        if new_val not in S_copy:
                            S_copy[idx] = new_val
                            S_copy.sort()
            
            if S_copy == S:
                T *= cooling
                continue
            
            E_old = compute_energy(S, target_diffs)
            E_new = compute_energy(S_copy, target_diffs)
            delta_E = E_new - E_old
            
            if delta_E < 0 or rng.random() < pow(2.718281828, -delta_E / max(T, 0.01)):
                S = S_copy
                if E_new < best_energy:
                    best_energy = E_new
                    best_S = S[:]
                    best_max = max(S)
            
            T *= cooling
        
        return best_S

    # Main execution
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Generate initial solutions via greedy
    initial_solutions = []
    for trial in range(20):
        if time.time() - start_time > deadline - 30:
            break
        rng = random.Random(trial * 1000 + 42)
        S = greedy_construct(rng)
        if S is not None:
            initial_solutions.append(S)
            score = max(S)
            if score < best_score:
                best_score = score
                best_result = S[:]
    
    # If no valid solution found, use fallback
    if best_result is None:
        # Guaranteed construction
        S = [0]
        for d in range(1, 31):
            found = False
            for s in S:
                x = s + d
                if x not in S:
                    # Check Sidon
                    test_S = S + [x]
                    test_S.sort()
                    if is_sidon(test_S):
                        S = test_S
                        found = True
                        break
            if not found:
                # Try larger values
                base = max(S) + 1
                for offset in range(1, 200):
                    x = base + offset
                    test_S = S + [x]
                    test_S.sort()
                    if is_sidon(test_S):
                        S = test_S
                        found = True
                        break
                if not found:
                    # Force add
                    x = max(S) + 1
                    while True:
                        test_S = S + [x]
                        test_S.sort()
                        if is_sidon(test_S):
                            S = test_S
                            break
                        x += 1
        best_result = S
        best_score = max(S)
    
    # Phase 2: Simulated annealing on best and other initial solutions
    for S_init in initial_solutions[:5]:
        if time.time() - start_time > deadline - 10:
            break
        time_budget = deadline - 10 - (time.time() - start_time)
        if time_budget < 5:
            break
        optimized = simulated_annealing(S_init, target_diffs, time_budget)
        if optimized is not None and is_valid(optimized, target_diffs):
            score = max(optimized)
            if score < best_score:
                best_score = score
                best_result = optimized[:]
    
    # Final verification
    if best_result is None or not is_valid(best_result, target_diffs):
        # Last resort
        S = [0]
        for d in range(1, 31):
            found = False
            for s in S:
                x = s + d
                if x not in S:
                    test_S = S + [x]
                    test_S.sort()
                    if is_sidon(test_S):
                        S = test_S
                        found = True
                        break
            if not found:
                base = max(S) + 1
                for offset in range(1, 500):
                    x = base + offset
                    test_S = S + [x]
                    test_S.sort()
                    if is_sidon(test_S):
                        S = test_S
                        found = True
                        break
        best_result = S
    
    return best_result
