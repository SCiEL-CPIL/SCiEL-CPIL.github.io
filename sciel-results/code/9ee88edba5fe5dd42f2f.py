"""Improved program using weighted greedy set cover for odd covering systems."""
import numpy as np

def construct():
    base = 51975
    
    # Generate all odd divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = set()
    for a in range(4):  # 3^a for a in 0..3
        for b in range(3):  # 5^b for b in 0..2
            for c in range(2):  # 7^c for c in 0..1
                for d in range(2):  # 11^d for d in 0..1
                    val = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if val > 1:
                        divisors.add(val)
    
    # Sort divisors in increasing order
    sorted_divs = sorted(divisors)
    
    # Precompute coverage masks for each (m, r) pair
    # We'll use a byte array for the uncovered set
    uncovered = np.ones(base, dtype=np.uint8)
    
    classes = []
    used_moduli = set()
    
    # For efficiency, we'll iterate through moduli in a strategic order
    # Weighted greedy: prioritize classes that cover many uncovered residues
    # with a cost based on the modulus size (smaller moduli cover more residues)
    
    # Precompute all possible (m, r) pairs and their coverage counts
    # To keep memory manageable, we'll compute on the fly but in a smart order
    
    # Start with larger moduli that cover fewer residues but can fill gaps,
    # but actually for set cover, we want high coverage first.
    # Let's use a ratio: newly_covered / cost, where cost = 1 (uniform) 
    # but we can bias toward smaller moduli since they cover more.
    
    # Actually, let's use the standard greedy set cover: pick the class 
    # that covers the most uncovered residues.
    # But to improve over the seed, we'll consider all (m, r) pairs, not just 
    # one per modulus.
    
    # For each modulus m, the best residue r is the one covering the most 
    # uncovered residues. We'll iterate through all moduli, find the best r 
    # for each, then pick the globally best (m, r) pair.
    
    # To make this more efficient, we'll do multiple passes:
    # In each pass, for each unused modulus, find the best r, then select 
    # the (m, r) with maximum coverage.
    
    # Precompute indices for each modulus
    # For modulus m, residues r cover positions r, r+m, r+2m, ...
    
    # We'll use a priority-based approach
    # For each modulus, track the best residue
    
    # Initial setup: for each modulus, compute coverage for all residues
    # This is expensive, so we'll do it incrementally
    
    # Let's use a different strategy: 
    # 1. For each modulus m (in some order), find the best r
    # 2. Select the (m, r) with the highest coverage
    # 3. Mark those residues as covered
    # 4. Repeat until no improvement
    
    # To speed up, we'll process moduli in decreasing order of size first
    # (they cover fewer residues, so they're more specialized), then fill 
    # gaps with smaller moduli.
    
    # Actually, for set cover, we should pick the set with the most uncovered 
    # elements. Let's do that properly.
    
    # For each modulus m, the number of residues is m.
    # For a given m, residue r covers floor((base - 1 - r) / m) + 1 elements 
    # if r < base, else 0. Since r < m and m divides base, each residue class 
    # covers exactly base/m elements.
    
    # So all (m, r) pairs for a given m cover the same number of residues.
    # The best r is the one that covers the most currently uncovered residues.
    
    # Strategy: 
    # - Maintain an uncovered array
    # - For each unused modulus m, find the r that maximizes uncovered[r::m].sum()
    # - Pick the (m, r) with the global maximum
    # - Update uncovered
    # - Repeat
    
    # To make this faster, we'll use numpy for the coverage checks.
    
    # Convert to numpy for speed
    uncovered_np = np.ones(base, dtype=np.uint8)
    
    # We'll iterate, picking the best (m, r) each time
    # To avoid O(n^2) behavior, we'll limit the number of iterations
    # to the number of available moduli.
    
    num_moduli = len(sorted_divs)
    
    for _ in range(num_moduli):
        best_m = None
        best_r = None
        best_count = -1
        
        for m in sorted_divs:
            if m in used_moduli:
                continue
            
            # Find the best residue for this modulus
            # Try all residues and count uncovered
            # For efficiency, we can sample or use a heuristic
            
            # Since base/m is the number of elements per class,
            # and m can be large, let's just iterate through residues
            # but use numpy slicing for speed
            
            # For small m, there are many residues; for large m, fewer
            # Let's try all residues for small m, and a subset for large m
            
            if m <= 100:
                # Try all residues
                best_r_local = 0
                best_count_local = -1
                for r in range(m):
                    count = uncovered_np[r::m].sum()
                    if count > best_count_local:
                        best_count_local = count
                        best_r_local = r
            else:
                # Try a subset of residues for large m
                # Sample some residues
                candidates = list(range(0, m, max(1, m // 50)))
                if (m - 1) not in candidates:
                    candidates.append(m - 1)
                best_r_local = 0
                best_count_local = -1
                for r in candidates:
                    count = uncovered_np[r::m].sum()
                    if count > best_count_local:
                        best_count_local = count
                        best_r_local = r
            
            if best_count_local > best_count:
                best_count = best_count_local
                best_m = m
                best_r = best_r_local
        
        if best_m is None or best_count <= 0:
            break
        
        used_moduli.add(best_m)
        classes.append([best_m, best_r])
        
        # Update uncovered array
        uncovered_np[best_r::best_m] = 0
    
    return classes
