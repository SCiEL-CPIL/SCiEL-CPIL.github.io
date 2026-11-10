"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Generate all odd divisors of base greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order
    n_divs = len(divisors)
    div_to_idx = {d: i for i, d in enumerate(divisors)}

    def score(classes):
        """Compute number of uncovered residues modulo base."""
        covered = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered[r::m] = True
        return base - int(np.sum(covered))

    def greedy_construct(order=None, cost_weighted=False):
        """Greedy: moduli in specified order.
        If cost_weighted, select r that maximizes new_coverage / cost where cost = 1.0/m.
        Otherwise, select r that maximizes new_coverage (count-based).
        """
        if order is None:
            order = divisors
        covered = np.zeros(base, dtype=bool)
        classes = []
        for m in order:
            # Find r that maximizes coverage of currently uncovered residues
            best_r = 0
            best_val = -1
            for r in range(m):
                new_cov = int(np.sum(~covered[r::m]))
                if cost_weighted:
                    val = new_cov / (1.0 / m)  # = new_cov * m
                else:
                    val = new_cov
                if val > best_val:
                    best_val = val
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = True
        return classes, score(classes)

    def local_search(classes, max_iter=30, time_budget=None):
        """Local search: try swapping residues for each modulus to improve score."""
        classes = [list(c) for c in classes]
        current_score = score(classes)
        n = len(classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if time_budget is not None and time.time() > time_budget:
                break
            
            improved = False
            for i in range(n):
                m, r = classes[i]
                # Compute covered without this class
                covered_without = np.zeros(base, dtype=bool)
                for j in range(n):
                    if j != i:
                        mj, rj = classes[j]
                        covered_without[rj::mj] = True
                
                # Find best residue for m given covered_without
                best_r = r
                best_new_covers = -1
                
                for r_new in range(m):
                    new_covers = int(np.sum(~covered_without[r_new::m]))
                    if new_covers > best_new_covers:
                        best_new_covers = new_covers
                        best_r = r_new
                
                if best_r != r:
                    classes[i][1] = best_r
                    current_score = score(classes)
                    improved = True
            
            if not improved:
                break
        
        return classes, current_score

    def simulated_annealing(classes, start_temp=1.0, end_temp=0.01, max_iter=200):
        """Simulated annealing on residue assignments."""
        classes = [list(c) for c in classes]
        n = len(classes)
        current_score = score(classes)
        best_classes = [list(c) for c in classes]
        best_score = current_score
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Cooling schedule
            progress = iteration / max_iter
            temp = start_temp * (end_temp / start_temp) ** progress
            
            # Pick a random class to perturb
            i = random.randint(0, n - 1)
            m, r = classes[i]
            old_r = r
            new_r = random.randint(0, m - 1)
            classes[i][1] = new_r
            
            new_score = score(classes)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [list(c) for c in classes]
            else:
                classes[i][1] = old_r
        
        return best_classes, best_score

    # Initial greedy (count-based)
    classes, current_score = greedy_construct()
    best_classes = [list(c) for c in classes]
    best_score = current_score

    # Local search on initial
    classes, current_score = local_search(classes, max_iter=30)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Try cost-weighted greedy
    classes, current_score = greedy_construct(cost_weighted=True)
    classes, current_score = local_search(classes, max_iter=20)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Multiple random restarts
    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1
        # Perturbed greedy: randomize the order of moduli
        random_divs = list(divisors)
        random.shuffle(random_divs)
        
        # Mix: sometimes use cost-weighted
        use_weighted = random.random() < 0.3
        classes, current_score = greedy_construct(order=random_divs, cost_weighted=use_weighted)
        
        # Local search
        classes, current_score = local_search(classes, max_iter=15)
        
        # Occasionally try simulated annealing
        if random.random() < 0.2 and time.time() < deadline - 20.0:
            classes, current_score = simulated_annealing(classes, max_iter=100)
        
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    # Final polish: local search on best solution with more iterations
    if time.time() < deadline - 10.0:
        classes, current_score = local_search(best_classes, max_iter=50)
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    return best_classes
# EVOLVE-BLOCK-END
