"""Hypergraph Ramsey R(4,5;3) construction: maximize n with valid 3-colouring."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 150s wall-clock limit
    
    # Precompute triple indices for various n
    # For n vertices, triples are (a,b,c) with a<b<c in lex order
    # Index: we can map (a,b,c) to a flat index
    
    def get_triples(n):
        """Return list of (a,b,c) triples in lex order."""
        return list(combinations(range(n), 3))
    
    def count_violations(n, colors):
        """Count red K4^3 (4-sets with all 4 triples red) and blue K5^3 (5-sets with all 10 triples blue)."""
        triples = get_triples(n)
        # Build a dict mapping (a,b,c) -> color
        # For efficiency, use a lookup by (a,b,c)
        color_map = {}
        for i, (a, b, c) in enumerate(triples):
            color_map[(a, b, c)] = colors[i]
        
        red_violations = 0
        blue_violations = 0
        
        # Check all 4-sets for red K4^3
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # The 4 triples within this quad
            tris = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            if all(color_map[t] == 0 for t in tris):
                red_violations += 1
        
        # Check all 5-sets for blue K5^3
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            tris = [(a,b,c),(a,b,d),(a,b,e),(a,c,d),(a,c,e),(a,d,e),(b,c,d),(b,c,e),(b,d,e),(c,d,e)]
            if all(color_map[t] == 1 for t in tris):
                blue_violations += 1
        
        return red_violations, blue_violations
    
    def is_valid(n, colors):
        r, b = count_violations(n, colors)
        return r == 0 and b == 0
    
    def random_coloring(n):
        """Generate a random coloring."""
        num_triples = len(get_triples(n))
        return np.random.randint(0, 2, size=num_triples)
    
    def simulated_annealing(n, max_iters=50000, initial_temp=1.0, cooling_rate=0.9995):
        """Run simulated annealing to find a valid coloring."""
        triples = get_triples(n)
        num_triples = len(triples)
        
        # Initialize with random coloring
        colors = random_coloring(n)
        
        # Precompute for each vertex, which triples it participates in
        # For incremental updates, we need to know which 4-sets/5-sets are affected
        # when we flip a triple
        
        # Build vertex-to-triples mapping
        vert_triples = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(triples):
            vert_triples[a].append(i)
            vert_triples[b].append(i)
            vert_triples[c].append(i)
        
        # For each 4-set, store the 4 triple indices
        quad_triples = {}
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            tris = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            indices = []
            for t in tris:
                # Find index of this triple
                # We need a fast lookup
                idx = triples.index(t)  # This is O(n^3) which is slow
                indices.append(idx)
            quad_triples[quad] = indices
        
        # This approach is too slow for large n. Let's use a different strategy.
        # Instead, let's use a direct approach with numpy for speed.
        
        # Alternative: use a bitset-based approach
        # For each 4-set, we need to check if all 4 triples are red
        # For each 5-set, we need to check if all 10 triples are blue
        
        # Let's try a simpler approach: just count violations from scratch each time
        # but do it efficiently with numpy
        
        def compute_violations(colors_arr):
            """Compute (red_violations, blue_violations) efficiently."""
            # Build a 3D array or use the triples list
            # For n up to 34, num_triples = C(34,3) = 5984
            # Number of 4-sets = C(34,4) = 46376
            # Number of 5-sets = C(34,5) = 278256
            
            # This might be too slow in pure Python. Let's use numpy.
            
            # Create a lookup: for triple (a,b,c), index = a*n*(n-1)//2 + b*(n-1) + c - something
            # Actually, let's just use a dict for small n, or precompute indices
            
            # For n <= 20, this is manageable. For larger n, we need optimization.
            
            red_v = 0
            blue_v = 0
            
            # For 4-sets
            for a in range(n-3):
                for b in range(a+1, n-2):
                    for c in range(b+1, n-1):
                        for d in range(c+1, n):
                            # Check if all 4 triples are red
                            # Triple indices: need to map (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                            # We need a fast way to get the color of a triple
                            pass
            
            return red_v, blue_v
        
        # This is getting complex. Let's use a pragmatic approach:
        # For small n (<= 15), do full verification
        # For larger n, use approximate methods
        
        # Actually, let's just do a simple local search for smaller n and use the known result for larger
        pass
    
    # Given the complexity, let's use a simpler strategy:
    # 1. Start with n=9 (known valid from seed)
    # 2. Try to grow n by adding vertices and re-coloring
    # 3. Use local search to fix violations
    
    # For n=9, the seed coloring is valid
    def seed_coloring(n):
        """Color triple red if sum of vertices divisible by 3."""
        return np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)])
    
    # Start with the seed
    n_current = 9
    colors_current = seed_coloring(n_current)
    best_n = n_current
    best_colors = colors_current.copy()
    
    # Try to increase n
    for target_n in range(10, 35):
        if time.time() > deadline - 5:
            break
            
        # Extend the coloring to target_n vertices
        # We need to add C(target_n, 3) - C(target_n-1, 3) new triples
        # New triples involve at least one vertex >= target_n-1
        
        # Strategy: start with seed coloring for target_n, then fix violations
        colors = seed_coloring(target_n)
        
        # Check if already valid
        r, b = count_violations(target_n, colors.tolist())
        if r == 0 and b == 0:
            best_n = target_n
            best_colors = colors.copy()
            continue
        
        # Try local search to fix violations
        # Flip individual triples to reduce violations
        improved = True
        max_iters = 5000
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            
            r, b = count_violations(target_n, colors.tolist())
            if r == 0 and b == 0:
                best_n = target_n
                best_colors = colors.copy()
                break
            
            # Try flipping each triple and see if it reduces violations
            best_delta = 0
            best_flip_idx = -1
            
            for i in range(len(colors)):
                colors[i] = 1 - colors[i]
                r2, b2 = count_violations(target_n, colors.tolist())
                delta = (r + b) - (r2 + b2)
                if delta > best_delta:
                    best_delta = delta
                    best_flip_idx = i
                colors[i] = 1 - colors[i]
            
            if best_flip_idx >= 0 and best_delta > 0:
                colors[best_flip_idx] = 1 - colors[best_flip_idx]
                improved = True
        
        if best_n >= target_n:
            continue
        
        # If we couldn't reach this n, try with random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            colors = random_coloring(target_n)
            r, b = count_violations(target_n, colors.tolist())
            
            if r == 0 and b == 0:
                best_n = target_n
                best_colors = colors.copy()
                break
            
            # Local search
            for _ in range(2000):
                if time.time() > deadline - 5:
                    break
                r, b = count_violations(target_n, colors.tolist())
                if r == 0 and b == 0:
                    best_n = target_n
                    best_colors = colors.copy()
                    break
                
                # Random flip
                idx = np.random.randint(0, len(colors))
                colors[idx] = 1 - colors[idx]
                r2, b2 = count_violations(target_n, colors.tolist())
                if r2 + b2 > r + b:
                    colors[idx] = 1 - colors[idx]  # Revert
    
    # Return the best result found
    return {'n': best_n, 'colors': best_colors.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
