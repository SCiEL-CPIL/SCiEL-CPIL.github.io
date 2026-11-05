import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    # Field F_25 arithmetic
    # Elements are (a, b) representing a + b*alpha, where alpha^2 = -1 = 4
    def add(x, y):
        return ((x[0] + y[0]) % 5, (x[1] + y[1]) % 5)
    
    def sub(x, y):
        return ((x[0] - y[0]) % 5, (x[1] - y[1]) % 5)
    
    def mul(x, y):
        # (a + b*alpha) * (c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # alpha^2 = 4
        a, b = x
        c, d = y
        real = (a*c + 4*b*d) % 5
        imag = (a*d + b*c) % 5
        return (real, imag)
    
    def inv(x):
        # Inverse of a + b*alpha is (a - b*alpha) / (a^2 + b^2)
        a, b = x
        norm = (a*a + b*b) % 5
        if norm == 0:
            return (0, 0) # Should not happen for F*
        inv_norm = pow(norm, 3, 5) # Fermat's little theorem: x^(p-2)
        real = (a * inv_norm) % 5
        imag = (4 * b * inv_norm) % 5 # -b
        return (real, imag)
    
    # Generate all elements of F_25
    elements = [(a, b) for a in range(5) for b in range(5)]
    # Identify zero
    zero = (0, 0)
    # Generate F_25^*
    f_star = [e for e in elements if e != zero]
    
    # Check if a set S is a Sidon set in F_25
    # Condition: For all x != 0, |S ∩ (S+x)| <= 1
    def is_sidon(S):
        S_set = set(S)
        for x in f_star:
            count = 0
            for s in S_set:
                target = add(s, x)
                if target in S_set:
                    count += 1
                    if count > 1:
                        return False
        return True

    # Backtracking to partition F_25^* into 5 Sidon sets
    # We want sets S0..S4 such that union is F_25^* and each is Sidon.
    # Max size of Sidon set in F_25 is 5.
    # 24 elements / 5 sets. Sizes must be e.g., 5,5,5,5,4.
    
    # Pre-compute Sidon status for small sets? 
    # Better: Greedy with backtracking.
    
    remaining = set(f_star)
    sets = [set() for _ in range(5)]
    
    # Order elements to help search? Random shuffle
    elems_list = list(f_star)
    random.seed(42)
    random.shuffle(elems_list)
    
    def backtrack(idx, sets):
        if idx == len(elems_list):
            # Check if all sets are Sidon
            for s in sets:
                if not is_sidon(s):
                    return False
            return True
        
        e = elems_list[idx]
        for i in range(5):
            # Heuristic: Don't make sets too big (max 5)
            if len(sets[i]) >= 5:
                continue
            
            # Try adding e to sets[i]
            sets[i].add(e)
            
            # Pruning: Check if sets[i] is still "potentially" Sidon
            # We can just check if it's Sidon now. If not, backtrack.
            # Note: Adding an element might break Sidon property.
            if is_sidon(sets[i]):
                if backtrack(idx + 1, sets):
                    return True
            
            sets[i].remove(e)
        
        return False

    # To speed up, we can fix the first few elements or use a better heuristic.
    # Given the small size, this should work.
    
    # Optimization: Sort sets by size in descending order during backtracking?
    # Or just rely on the random shuffle and recursion limit.
    
    # Increase recursion limit if needed
    import sys
    sys.setrecursionlimit(10000)
    
    if backtrack(0, sets):
        # We found a partition
        # Map F_25 elements to indices 0..24
        # We need a consistent mapping
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        # Precompute color for each pair in F_25^*
        color_map = {}
        for i, s_i in enumerate(sets):
            for e in s_i:
                color_map[e] = i
        
        n = 26
        colors = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if j == 25:
                    # Edge (i, infinity)
                    # Vertex i corresponds to elements[i]
                    d = elements[i]
                    c = color_map[d]
                    colors.append(c)
                else:
                    # Edge (i, j)
                    # d = elements[j] - elements[i]
                    d = sub(elements[j], elements[i])
                    c = color_map[d]
                    colors.append(c)
        
        return {'n': n, 'colors': colors}
    else:
        # Fallback to a known smaller construction if partition fails (unlikely)
        # But 26 is known to exist.
        # Return n=11 construction as fallback
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
