"""Rewritten program for maximizing sum of pairwise-coprime integers from 1..256."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    
    # Precompute compatibility matrix
    compat = [[True] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            c = not (prime_factors[i] & prime_factors[j])
            compat[i][j] = c
            compat[j][i] = c
    
    def is_compatible_with_all(x, solution_list):
        for b in solution_list:
            if not compat[x][b]:
                return False
        return True
    
    def greedy_start(rng=None):
        """Generate a greedy initial solution."""
        solution = [1]
        solution_set = {1}
        
        candidates = list(range(N, 1, -1))
        if rng is not None:
            # Randomize the order somewhat but keep bias toward larger numbers
            # Use weighted random selection
            weights = [1.0 / (N - c + 1) for c in candidates]
            # Actually, let's just shuffle with a bias
            # Simpler: shuffle and then sort with a random key
            # Or just shuffle the candidates
            rng.shuffle(candidates)
            # Re-sort to maintain some structure? No, let's just use shuffled
            # But we want to prefer larger numbers. Let's use a different approach.
            # Pick random candidates but with probability proportional to value
            # Actually, let's just shuffle and try to add
            pass
        
        # For deterministic start, try in descending order
        # For randomized start, shuffle the candidates
        for a in candidates:
            if a in solution_set:
                continue
            if is_compatible_with_all(a, solution):
                solution.append(a)
                solution_set.add(a)
        
        return solution, solution_set, sum(solution)
    
    def local_search(solution, solution_set, current_sum, deadline):
        """Perform local search with various moves."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1-for-1 swaps: replace a in solution with b not in solution
            for a in list(solution):
                # Try all b > a not in solution
                for b in range(a + 1, N + 1):
                    if b in solution_set:
                        continue
                    if not compat[a][b]:
                        continue
                    # Check if b is compatible with all other elements
                    others = [x for x in solution if x != a]
                    if all(compat[b][x] for x in others):
                        # This improves sum since b > a
                        solution.remove(a)
                        solution.append(b)
                        solution_set.discard(a)
                        solution_set.add(b)
                        current_sum += b - a
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 2-for-1: replace two elements a1, a2 with one element b
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    a1, a2 = solution[i], solution[j]
                    # Try b > a1 + a2
                    for b in range(a1 + a2 + 1, N + 1):
                        if b in solution_set:
                            continue
                        # Check b compatible with all other elements
                        others = [x for k, x in enumerate(solution) if k != i and k != j]
                        if all(compat[b][x] for x in others):
                            solution.remove(a1)
                            solution.remove(a2)
                            solution.append(b)
                            solution_set.discard(a1)
                            solution_set.discard(a2)
                            solution_set.add(b)
                            current_sum += b - a1 - a2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 1-for-2: replace one element a with two elements b, c
            for a in list(solution):
                found = False
                for b in range(a + 1, N + 1):
                    if b in solution_set:
                        continue
                    if not compat[a][b]:
                        continue
                    others = [x for x in solution if x != a]
                    if not all(compat[b][x] for x in others):
                        continue
                    for c in range(b + 1, N + 1):
                        if c in solution_set or c == b:
                            continue
                        if not compat[a][c]:
                            continue
                        if not compat[b][c]:
                            continue
                        if not all(compat[c][x] for x in others):
                            continue
                        if b + c > a:
                            solution.remove(a)
                            solution.append(b)
                            solution.append(c)
                            solution_set.discard(a)
                            solution_set.add(b)
                            solution_set.add(c)
                            current_sum += b + c - a
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        return solution, solution_set, current_sum
    
    # Multi-start local search
    best_solution = None
    best_sum = -1
    
    # First, deterministic greedy start
    sol, sol_set, cur_sum = greedy_start(None)
    sol, sol_set, cur_sum = local_search(sol, sol_set, cur_sum, deadline)
    best_solution = list(sol)
    best_sum = cur_sum
    
    # Multiple randomized restarts
    rng = random.Random(42)
    while time.time() < deadline - 10:
        sol, sol_set, cur_sum = greedy_start(rng)
        sol, sol_set, cur_sum = local_search(sol, sol_set, cur_sum, deadline)
        
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_solution = list(sol)
    
    # Final high-precision polish on the best solution
    if best_solution is not None and time.time() < deadline - 2:
        best_sol_set = set(best_solution)
        best_solution, best_sol_set, best_sum = local_search(
            best_solution, best_sol_set, best_sum, deadline
        )
    
    # Final validation
    result = list(best_solution)
    assert len(result) == len(set(result)), "Duplicates found"
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in result), "Non-int found"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) == 1, f"GCD not 1: {result[i]}, {result[j]}"
    
    return result
# EVOLVE-BLOCK-END
