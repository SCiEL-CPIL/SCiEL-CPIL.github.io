import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    N = 51975
    
    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()
    
    # We'll use a subset of these divisors. Start with all, then optimize.
    # Use numpy for fast coverage computation
    
    def compute_uncovered(classes):
        """Compute number of uncovered residues given list of [m, r] classes."""
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(np.sum(covered == 0))
    
    # Greedy initialization: sort moduli, pick best residue for each
    def greedy_init(moduli_list):
        classes = []
        covered = np.zeros(N, dtype=np.uint8)
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                # Count uncovered in this class
                mask = covered[r::m] == 0
                cnt = int(np.sum(mask))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, int(np.sum(covered == 0))
    
    # Start with all divisors in increasing order
    initial_classes, initial_score = greedy_init(divisors)
    best_classes = initial_classes
    best_score = initial_score
    
    # Local search: try swapping residues or changing which moduli to use
    def local_search(classes, score, time_limit):
        classes = [c[:] for c in classes]
        score = score
        covered = np.zeros(N, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                # Try all possible residues for this modulus
                best_r = r
                best_score = score
                # Remove current class from coverage
                covered[r::m] = 0
                current_uncovered = int(np.sum(covered == 0))
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    covered[new_r::m] = 1
                    new_uncovered = int(np.sum(covered == 0))
                    if new_uncovered < best_score:
                        best_score = new_uncovered
                        best_r = new_r
                    covered[new_r::m] = 0
                
                # Apply best
                if best_r != r:
                    classes[i] = [m, best_r]
                    covered[best_r::m] = 1
                    score = best_score
                    improved = True
                else:
                    covered[r::m] = 1
        
        return classes, score
    
    # Run local search with time limit
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 5:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 2)
    
    # Try removing some moduli if it helps (unlikely but check)
    # Also try different orderings
    
    # Simulated annealing-like approach: try random perturbations
    if (time.time() - start_time) < deadline - 10:
        import random
        random.seed(42)
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        
        for iteration in range(1000):
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Perturb: change one residue
            idx = random.randint(0, len(current_classes) - 1)
            m, r = current_classes[idx]
            old_r = r
            new_r = random.randint(0, m - 1)
            
            if new_r == old_r:
                continue
            
            # Compute new score efficiently
            covered = np.zeros(N, dtype=np.uint8)
            for i, (mm, rr) in enumerate(current_classes):
                if i == idx:
                    rr = new_r
                covered[rr::mm] = 1
            new_score = int(np.sum(covered == 0))
            
            if new_score < current_score:
                current_classes[idx] = [m, new_r]
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in current_classes]
            else:
                # Revert
                pass
        
        # Final local search
        if (time.time() - start_time) < deadline - 2:
            best_classes, best_score = local_search(best_classes, best_score, deadline - 1)
    
    return best_classes
