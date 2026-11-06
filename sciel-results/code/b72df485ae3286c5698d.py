"""Optimized odd covering system construction for moduli dividing 51975."""
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for i in range(2, base + 1):
        if base % i == 0:
            divisors.append(i)
    
    # Sort by size descending - larger moduli cover fewer residues but are more "efficient"
    # Actually, for covering, we want to think about which residues are uncovered.
    # Let's use a smarter approach: process moduli in an order that maximizes coverage gain.
    
    # Start with all residues uncovered
    covered = np.zeros(base, dtype=bool)
    classes = []
    
    # We'll use a greedy approach but with a twist:
    # For each modulus, find the residue class that covers the most currently uncovered residues.
    # Process moduli in order of increasing size (smaller moduli cover more residues per class)
    
    # Sort divisors by size ascending
    divisors.sort()
    
    # Precompute for each modulus the count of uncovered residues per residue class
    # This is expensive, so let's be smart about it.
    
    # Actually, let's think differently. The key insight is that we want to minimize
    # uncovered residues. Let's use a different strategy:
    
    # Strategy: Use the Chinese Remainder Theorem structure.
    # 51975 = 27 * 25 * 7 * 11
    # We can build a covering system by combining congruences.
    
    # Alternative approach: Randomized local search
    # Start with a random assignment of residue classes to moduli, then improve.
    
    import random
    
    def count_uncovered(classes):
        """Count uncovered residues given a list of [m, r] classes."""
        covered_arr = np.zeros(base, dtype=bool)
        for m, r in classes:
            covered_arr[r::m] = True
        return np.sum(~covered_arr)
    
    def random_assignment():
        """Create a random assignment of residue classes to all divisors."""
        classes = []
        for m in divisors:
            r = random.randint(0, m - 1)
            classes.append([m, r])
        return classes
    
    # Try multiple random starts and local search
    best_classes = None
    best_score = base  # worst case: all uncovered
    
    n_attempts = 50
    max_iterations = 200
    
    for attempt in range(n_attempts):
        # Start with a random or structured initial solution
        if attempt % 2 == 0:
            # Random start
            current_classes = random_assignment()
        else:
            # Structured start: for each modulus, pick residue 0
            current_classes = [[m, 0] for m in divisors]
        
        current_score = count_uncovered(current_classes)
        
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in current_classes]
        
        # Local search: try changing one class at a time
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and current_score > 0:
            improved = False
            iterations += 1
            
            # Shuffle to avoid bias
            order = list(range(len(current_classes)))
            random.shuffle(order)
            
            for idx in order:
                m, r = current_classes[idx]
                best_r = r
                best_local_score = current_score
                
                # Try all possible residues for this modulus
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute new score
                    # We need to compute the change in coverage
                    # Old coverage from this class: residues r, r+m, r+2m, ...
                    # New coverage: residues new_r, new_r+m, new_r+2m, ...
                    
                    # To compute the change efficiently:
                    # Count how many residues were ONLY covered by this class (old)
                    # and how many new residues are covered (new)
                    
                    # For simplicity, just recompute (this is O(base) per change)
                    # But that's too slow. Let's use a smarter method.
                    
                    # Actually, let's just recompute the whole thing for correctness,
                    # but only for promising candidates.
                    
                    # Quick estimate: the number of residues covered by class (m, r) is base//m
                    # The intersection with other classes is complex.
                    
                    # Let's just do the full recomputation for now, but limit it.
                    new_classes = [c[:] for c in current_classes]
                    new_classes[idx] = [m, new_r]
                    new_score = count_uncovered(new_classes)
                    
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_r = new_r
                
                if best_r != r:
                    current_classes[idx] = [m, best_r]
                    current_score = best_local_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                    if current_score == 0:
                        break
        
        if best_score == 0:
            break
    
    return best_classes
