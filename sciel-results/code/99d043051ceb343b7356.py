import numpy as np
from itertools import combinations

def construct():
    # We aim for n=34 or higher.
    # Use finite field F_37
    q = 37
    
    # Helper to check quadratic residue
    # x is QR if x^( (q-1)/2 ) == 1
    # x is QNR if x^( (q-1)/2 ) == -1 (i.e., q-1)
    def is_quadratic_residue(x, q):
        if x == 0:
            return False
        return pow(x, (q - 1) // 2, q) == 1

    # Generate colors for all triples in 0..q-1
    # n_max = q
    n_max = q
    colors = []
    # Precompute for speed?
    # We need colors for combinations(range(n_max), 3)
    
    # Let's generate the full hypergraph for q=37
    # Map triple (a,b,c) to index
    # We will store colors in a list corresponding to lexicographic order of triples of 0..n-1
    # But first, let's just generate a dict or array for all triples of 0..36
    
    # For efficiency, let's compute the "type" of each triple
    # Red (0) if QR, Blue (1) if QNR
    # Note: The problem says 0 is Red, 1 is Blue.
    
    triple_colors = {}
    for a in range(q):
        for b in range(a+1, q):
            for c in range(b+1, q):
                # Determinant of Vandermonde
                # (a-b)(a-c)(b-c)
                d = ((a - b) * (a - c) * (b - c)) % q
                if is_quadratic_residue(d, q):
                    triple_colors[(a, b, c)] = 0  # Red
                else:
                    triple_colors[(a, b, c)] = 1  # Blue

    # Check if the full set 0..36 is valid
    def is_valid(vertices):
        n = len(vertices)
        v_set = set(vertices)
        
        # Check Red K4
        for comb4 in combinations(vertices, 4):
            is_red_k4 = True
            for triple in combinations(comb4, 3):
                # Sort triple to match key
                t = tuple(sorted(triple))
                if triple_colors[t] != 0:
                    is_red_k4 = False
                    break
            if is_red_k4:
                return False, "Red K4 found: " + str(comb4)
                
        # Check Blue K5
        for comb5 in combinations(vertices, 5):
            is_blue_k5 = True
            for triple in combinations(comb5, 3):
                t = tuple(sorted(triple))
                if triple_colors[t] != 1:
                    is_blue_k5 = False
                    break
            if is_blue_k5:
                return False, "Blue K5 found: " + str(comb5)
                
        return True, "Valid"

    valid, msg = is_valid(list(range(q)))
    
    if valid:
        n = q
        vertices = list(range(q))
    else:
        # Greedy removal
        vertices = list(range(q))
        
        # To improve results, we can try multiple random removals
        # But deterministic greedy is safer for reproducibility
        # We remove vertices from the "smallest" forbidden clique found?
        # Or just the first one found.
        
        import random
        best_vertices = None
        best_n = 0
        
        # Try a few random seeds for the removal process to see if we can hit 34
        for seed in range(10):
            random.seed(seed)
            v = list(range(q))
            while True:
                valid, msg = is_valid(v)
                if valid:
                    break
                
                if "Red K4" in msg:
                    # Extract the clique
                    # msg format: "Red K4 found: (a, b, c, d)"
                    start = msg.find('(')
                    end = msg.find(')')
                    clique_str = msg[start+1:end]
                    clique = tuple(map(int, clique_str.split(', ')))
                    # Remove a random vertex from the clique
                    v.remove(random.choice(clique))
                else: # Blue K5
                    start = msg.find('(')
                    end = msg.find(')')
                    clique_str = msg[start+1:end]
                    clique = tuple(map(int, clique_str.split(', ')))
                    v.remove(random.choice(clique))
            
            if len(v) > best_n:
                best_n = len(v)
                best_vertices = v.copy()
                
        if best_n < 34:
             # Fallback: Maybe the algebraic construction on F_37 isn't optimal.
             # Try F_32? No, too small.
             # Try a different coloring rule?
             # Or just return the best we found.
             # But we know 34 exists.
             # If greedy fails to find 34, it might be stuck in a local minimum.
             # We can try to "repair" or use a different base.
             
             # Let's try to explicitly construct the known 34-vertex solution if we can find it online?
             # No, we must generate it.
             
             # Alternative: Use the construction on F_31?
             # If F_37 greedy gives < 34, try F_41?
             
             pass

        vertices = best_vertices
        n = len(vertices)

    # Relabel vertices to 0..n-1
    # Create a mapping from old label to new label
    old_to_new = {v: i for i, v in enumerate(vertices)}
    
    # Reconstruct colors for the new vertex set
    # We need colors for all triples of 0..n-1
    # The triples in the new set correspond to triples in the old set
    # We need to list them in lexicographic order of the NEW labels
    
    final_colors = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                # Map back to old labels
                old_a = vertices[a]
                old_b = vertices[b]
                old_c = vertices[c]
                t = tuple(sorted((old_a, old_b, old_c)))
                # Look up color
                color = triple_colors[t]
                final_colors.append(color)
                
    return {'n': n, 'colors': final_colors}
