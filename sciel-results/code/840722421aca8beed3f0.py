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
    
    # Greedy initial solution - process larger moduli first (they cover fewer residues per class)
    # Actually, let's try multiple orderings and keep the best
    best_initial_score = float('inf')
    best_initial_classes = None
    best_initial_covered = None
    
    orderings = [
        'desc',  # larger moduli first
        'asc',   # smaller moduli first
        'random1',
        'random2'
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        if order == 'desc':
            ordered_divs = sorted(divisors, reverse=True)
        elif order == 'asc':
            ordered_divs = sorted(divisors)
        elif order == 'random1':
            ordered_divs = divisors[:]
            random.shuffle(ordered_divs)
        else:  # random2
            ordered_divs = divisors[:]
            random.shuffle(ordered_divs)
        
        covered_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in ordered_divs:
            best_r = -1
            best_count = -1
            for r in range(m):
                idx = residue_indices[m][r]
                count = np.count_nonzero(covered_count[idx] == 0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            idx = residue_indices[m][best_r]
            covered_count[idx] += 1
        
        score = np.count_nonzero(covered_count == 0)
        if score < best_initial_score:
            best_initial_score = score
            best_initial_classes = [c[:] for c in classes]
            best_initial_covered = covered_count.copy()
    
    best_score = best_initial_score
    best_classes = best_initial_classes
    best_covered_count = best_initial_covered
    
    # Local search with faster evaluation
    # Key insight: precompute for each modulus m, for each residue r, the index array
    # We can vectorize the gain/loss computation across all residues for a given m
    
    while time.time() < deadline:
        improved = False
        classes = [c[:] for c in best_classes]
        covered_count = best_covered_count.copy()
        current_score = best_score
        
        # Multi-pass local search
        for pass_num in range(30):
            if time.time() > deadline:
                break
            
            # Process classes in random order for this pass
            class_order = list(range(len(classes)))
            random.shuffle(class_order)
            
            for i in class_order:
                if time.time() > deadline:
                    break
                
                m, r = classes[i]
                old_idx = residue_indices[m][r]
                
                # Vectorized evaluation: compute gains and losses for all new_r at once
                # For each new_r, gains = count of uncovered in new_idx
                # For each new_r, losses = count of singly-covered in old_idx (constant for all new_r)
                losses = np.count_nonzero(covered_count[old_idx] == 1)
                
                # Compute gains for all residues of m at once
                # Build a 2D array: shape (m, len(old_idx)) where each row is a residue class
                # But this might be too memory intensive for large m
                # Instead, loop but with early termination
                best_new_r = -1
                best_delta = 0  # we need delta < 0 to improve
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_idx = residue_indices[m][new_r]
                    gains = np.count_nonzero(covered_count[new_idx] == 0)
                    delta = losses - gains
                    if delta < best_delta:
                        best_delta = delta
                        best_new_r = new_r
                
                if best_new_r >= 0 and best_delta < 0:
                    # Accept move
                    current_score += best_delta
                    new_idx = residue_indices[m][best_new_r]
                    covered_count[old_idx] -= 1
                    covered_count[new_idx] += 1
                    classes[i] = [m, best_new_r]
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                        best_covered_count = covered_count.copy()
        
        # If no improvement, try perturbation
        if not improved:
            # Try different perturbation strengths
            for strength in [0.2, 0.3, 0.5]:
                if time.time() > deadline:
                    break
                
                classes = [c[:] for c in best_classes]
                covered_count = best_covered_count.copy()
                for i in range(len(classes)):
                    if random.random() < strength:
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
                    improved = True
    
    return best_classes
# EVOLVE-BLOCK-END
