import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        n = len(A)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    def local_search(A, deadline):
        """Local search via swaps to minimize objective."""
        A = sorted(A)
        current_score = evaluate(A)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(A)
            # Try all swaps
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        return A, current_score
                    old_i, old_j = A[i], A[j]
                    A[i], A[j] = A[j], A[i]
                    new_score = evaluate(A)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        A[i], A[j] = old_i, old_j
            # Try removing one element and adding a new one
            for i in range(n):
                if time.time() > deadline:
                    return A, current_score
                old_val = A[i]
                # Try several replacement values
                for attempt in range(50):
                    if time.time() > deadline:
                        return A, current_score
                    new_val = random.randint(1, 10000)
                    if new_val in A:
                        continue
                    A[i] = new_val
                    new_score = evaluate(A)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        A[i] = old_val
        return A, current_score
    
    def generate_geometric(a, r, n=20, max_val=10000):
        """Generate geometric-like sequence."""
        result = []
        for i in range(n):
            val = int(a * (r ** i))
            if val > max_val or val <= 0:
                break
            if val not in result:
                result.append(val)
        return result
    
    def generate_arithmetic(a, d, n=20, max_val=10000):
        """Generate arithmetic sequence."""
        result = []
        for i in range(n):
            val = a + i * d
            if val > max_val or val <= 0:
                break
            if val not in result:
                result.append(val)
        return result
    
    def generate_mixed(n=20, max_val=10000):
        """Generate a mixed structured set."""
        # Use a base geometric sequence with perturbations
        a = random.randint(1, 5)
        r = random.choice([1.5, 1.8, 2.0, 2.2, 2.5])
        result = []
        for i in range(n):
            val = int(a * (r ** i))
            # Add small perturbation
            perturbation = random.randint(-2, 2)
            val += perturbation
            if val < 1:
                val = 1
            if val > max_val:
                val = max_val
            if val not in result:
                result.append(val)
        # If we don't have enough, fill with random
        while len(result) < n:
            val = random.randint(1, max_val)
            if val not in result:
                result.append(val)
        return result[:n]
    
    # Generate initial solutions
    initial_solutions = []
    
    # Pure geometric sequences
    for a in [1, 2, 3]:
        for r in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
            sol = generate_geometric(a, r)
            if len(sol) >= 20:
                initial_solutions.append(sol[:20])
    
    # Pure arithmetic sequences
    for a in [1, 2, 3, 5]:
        for d in [1, 2, 3, 4, 5, 7, 10]:
            sol = generate_arithmetic(a, d)
            if len(sol) >= 20:
                initial_solutions.append(sol[:20])
    
    # Mixed sequences
    for _ in range(20):
        initial_solutions.append(generate_mixed())
    
    # Random sets
    for _ in range(10):
        sol = random.sample(range(1, 10001), 20)
        initial_solutions.append(sol)
    
    # Evaluate and local search each initial solution
    for sol in initial_solutions:
        if time.time() > deadline:
            break
        sol = sorted(sol)
        score = evaluate(sol)
        if score < best_score:
            best_score = score
            best_set = list(sol)
        
        # Local search
        improved_sol, improved_score = local_search(list(sol), deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_set = list(improved_sol)
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 5.0:
        sol = random.sample(range(1, 10001), 20)
        sol = sorted(sol)
        score = evaluate(sol)
        if score < best_score:
            best_score = score
            best_set = list(sol)
        
        improved_sol, improved_score = local_search(list(sol), deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_set = list(improved_sol)
    
    if best_set is None:
        best_set = list(range(1, 21))
    
    return best_set
# EVOLVE-BLOCK-END
