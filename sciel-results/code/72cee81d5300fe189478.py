"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer before 120s limit
    
    # Generate divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute residue lists for each divisor
    # For each divisor m, precompute the indices for each residue class
    residue_indices = {}
    for m in divisors:
        residue_indices[m] = {}
        for r in range(m):
            # Indices: r, r+m, r+2m, ...
            residue_indices[m][r] = np.arange(r, base, m)
    
    # Greedy initial solution
    covered_count = np.zeros(base, dtype=np.int32)
    classes = []
    for m in divisors:
        best_r = -1
        best_count = -1
        for r in range(m):
            idx = residue_indices[m][r]
            # Count how many are currently uncovered (count == 0)
            count = np.count_nonzero(covered_count[idx] == 0)
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        idx = residue_indices[m][best_r]
        covered_count[idx] += 1
    
    # Calculate initial score
    best_score = np.count_nonzero(covered_count == 0)
    best_classes = [c[:] for c in classes]
    best_covered_count = covered_count.copy()
    
    # Local search
    # We maintain covered_count for the current best solution
    # We try to improve by swapping residues
    
    while time.time() < deadline:
        # Try to improve current best
        improved = False
        classes = [c[:] for c in best_classes]
        covered_count = best_covered_count.copy()
        current_score = best_score
        
        # Try changing residue for each class
        # We iterate through classes multiple times (multi-pass) within a time limit
        for pass_num in range(20):
            if time.time() > deadline:
                break
            for i, (m, r) in enumerate(classes):
                if time.time() > deadline:
                    break
                for new_r in range(m):
                    if new_r == r:
                        continue
                    old_idx = residue_indices[m][r]
                    new_idx = residue_indices[m][new_r]
                    
                    # Calculate delta
                    # Gains: residues in new_idx that are currently uncovered (count == 0)
                    gains = np.count_nonzero(covered_count[new_idx] == 0)
                    # Losses: residues in old_idx that are currently covered only by this class (count == 1)
                    # Note: We must be careful if old_idx and new_idx overlap, but since they are distinct residues mod m, they are disjoint.
                    losses = np.count_nonzero(covered_count[old_idx] == 1)
                    
                    delta = losses - gains
                    if delta < 0:
                        # Accept move
                        current_score += delta
                        covered_count[old_idx] -= 1
                        covered_count[new_idx] += 1
                        classes[i] = [m, new_r]
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_classes = [c[:] for c in classes]
                            best_covered_count = covered_count.copy()
        
        # If no improvement, try a random restart with perturbation
        if not improved:
            # Perturb: randomly change some residues
            classes = [c[:] for c in best_classes]
            covered_count = best_covered_count.copy()
            for i in range(len(classes)):
                if random.random() < 0.3:
                    m, r = classes[i]
                    old_idx = residue_indices[m][r]
                    new_r = random.randint(0, m-1)
                    if new_r != r:
                        new_idx = residue_indices[m][new_r]
                        covered_count[old_idx] -= 1
                        covered_count[new_idx] += 1
                        classes[i] = [m, new_r]
            
            new_score = np.count_nonzero(covered_count == 0)
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in classes]
                best_covered_count = covered_count.copy()
    
    return best_classes
# EVOLVE-BLOCK-END
