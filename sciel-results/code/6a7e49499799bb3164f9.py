import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Target n
    n = 34
    
    # 1. Define difference classes
    # A triple is characterized by the sorted tuple of its cyclic side lengths
    # d1, d2, d3 such that d1+d2+d3 = n and di >= 1
    classes = []
    class_map = {}
    for d1 in range(1, n - 1):
        for d2 in range(1, n - d1):
            d3 = n - d1 - d2
            if d3 <= 0:
                continue
            # Sort to get canonical representation
            key = tuple(sorted([d1, d2, d3]))
            if key not in class_map:
                class_map[key] = len(classes)
                classes.append(key)
    
    num_classes = len(classes)
    
    # 2. Map every triple (a,b,c) to its class ID
    # We need to generate all triples for n=34
    # triple_to_class: dict mapping (a,b,c) to class_id
    triple_to_class = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                d1 = b - a
                d2 = c - b
                d3 = n - (d1 + d2)
                key = tuple(sorted([d1, d2, d3]))
                triple_to_class[(a, b, c)] = class_map[key]
                
    # 3. Identify K4 and K5 structures and their class compositions
    # A K4 is a set of 4 vertices. It contains 4 triples.
    # If all 4 triples are Red (color 0), it's a violation.
    # A K5 is a set of 5 vertices. It contains 10 triples.
    # If all 10 triples are Blue (color 1), it's a violation.
    
    k4_class_sets = []
    for verts in combinations(range(n), 4):
        c0 = verts[0]; c1 = verts[1]; c2 = verts[2]; c3 = verts[3]
        triples = [
            (c0, c1, c2), (c0, c1, c3),
            (c0, c2, c3), (c1, c2, c3)
        ]
        class_set = frozenset([triple_to_class[t] for t in triples])
        k4_class_sets.append(class_set)
        
    k5_class_sets = []
    for verts in combinations(range(n), 5):
        class_set = set()
        for t in combinations(verts, 3):
            class_set.add(triple_to_class[t])
        k5_class_sets.append(frozenset(class_set))
        
    # 4. Local Search / Simulated Annealing on class colors
    # State: list of 0s and 1s of length num_classes
    
    def count_violations(colors):
        v_count = 0
        # Check Red K4
        for cs in k4_class_sets:
            # All classes in this K4 must be 0
            if all(colors[c] == 0 for c in cs):
                v_count += 1
        # Check Blue K5
        for cs in k5_class_sets:
            # All classes in this K5 must be 1
            if all(colors[c] == 1 for c in cs):
                v_count += 1
        return v_count

    def solve():
        best_colors = [random.randint(0, 1) for _ in range(num_classes)]
        best_score = count_violations(best_colors)
        
        if best_score == 0:
            return best_colors
            
        current_colors = best_colors[:]
        current_score = best_score
        
        # Simple Hill Climbing with random restarts
        # Since num_classes is ~100, this is very fast
        
        max_iterations = 5000
        restarts = 100
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
                
            if r > 0:
                current_colors = [random.randint(0, 1) for _ in range(num_classes)]
                current_score = count_violations(current_colors)
                if current_score < best_score:
                    best_colors = current_colors[:]
                    best_score = current_score
                    if best_score == 0:
                        return best_colors
            
            for i in range(max_iterations):
                if time.time() > deadline - 5:
                    return best_colors
                
                # Pick a random class to flip
                idx = random.randint(0, num_classes - 1)
                old_color = current_colors[idx]
                current_colors[idx] = 1 - old_color
                
                new_score = count_violations(current_colors)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = current_colors[:]
                        if best_score == 0:
                            return best_colors
                else:
                    # Revert
                    current_colors[idx] = old_color
                    
        return best_colors

    colors = solve()
    
    # Verify and construct output
    if count_violations(colors) == 0:
        # Construct the full color list for all triples in lexicographic order
        final_colors = []
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    final_colors.append(colors[triple_to_class[(a, b, c)]])
        return {'n': n, 'colors': final_colors}
    
    # Fallback if n=34 fails (should not happen)
    # Try n=33 or return seed
    # For safety, return a valid small n
    n_fb = 9
    colors_fb = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fb), 3)]
    return {'n': n_fb, 'colors': colors_fb}

# Keep the required function signature
# The task says "Keep the function names and signatures of the seed program"
# The seed program has `construct()`.

if __name__ == "__main__":
    res = construct()
    print(res['n'])
