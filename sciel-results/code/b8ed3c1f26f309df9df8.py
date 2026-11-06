import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def build_data(n):
        idx = {}
        i = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = i
                    i += 1
        n_triples = i
        
        quads = []
        for quad in combinations(range(n), 4):
            tri_indices = [idx[t] for t in combinations(quad, 3)]
            quads.append(tri_indices)
        
        quints = []
        for quint in combinations(range(n), 5):
            tri_indices = [idx[t] for t in combinations(quint, 3)]
            quints.append(tri_indices)
        
        triple_to_quads = [[] for _ in range(n_triples)]
        for qi, tri_indices in enumerate(quads):
            for ti in tri_indices:
                triple_to_quads[ti].append(qi)
        
        triple_to_quints = [[] for _ in range(n_triples)]
        for qi, tri_indices in enumerate(quints):
            for ti in tri_indices:
                triple_to_quints[ti].append(qi)
                
        return n_triples, quads, quints, triple_to_quads, triple_to_quints

    def solve_n(n):
        n_triples, quads, quints, triple_to_quads, triple_to_quints = build_data(n)
        
        # Simulated Annealing
        best_E = float('inf')
        best_colors = None
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            colors = np.random.randint(0, 2, size=n_triples)
            
            # Initialize counts
            red_count = np.zeros(len(quads), dtype=np.int32)
            blue_count = np.zeros(len(quints), dtype=np.int32)
            
            for qi, tri_indices in enumerate(quads):
                for ti in tri_indices:
                    if colors[ti] == 0:
                        red_count[qi] += 1
                        
            for qi, tri_indices in enumerate(quints):
                for ti in tri_indices:
                    if colors[ti] == 1:
                        blue_count[qi] += 1
            
            E = np.sum(red_count == 4) + np.sum(blue_count == 10)
            
            if E == 0:
                return colors.tolist()
                
            T = 1.0
            cooling = 0.995
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                if E == 0:
                    return colors.tolist()
                    
                # Pick a random triple
                t = random.randint(0, n_triples - 1)
                
                delta_E = 0
                if colors[t] == 0:
                    # Flip to 1 (Blue)
                    # Red violations removed
                    for qi in triple_to_quads[t]:
                        if red_count[qi] == 4:
                            delta_E -= 1
                    # Blue violations added
                    for qi in triple_to_quints[t]:
                        if blue_count[qi] == 9:
                            delta_E += 1
                else:
                    # Flip to 0 (Red)
                    # Blue violations removed
                    for qi in triple_to_quints[t]:
                        if blue_count[qi] == 10:
                            delta_E -= 1
                    # Red violations added
                    for qi in triple_to_quads[t]:
                        if red_count[qi] == 3:
                            delta_E += 1
                
                # Acceptance
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Apply flip
                    old_color = colors[t]
                    colors[t] = 1 - old_color
                    E += delta_E
                    
                    # Update counts
                    if old_color == 0:
                        for qi in triple_to_quads[t]:
                            red_count[qi] -= 1
                        for qi in triple_to_quints[t]:
                            blue_count[qi] += 1
                    else:
                        for qi in triple_to_quads[t]:
                            red_count[qi] += 1
                        for qi in triple_to_quints[t]:
                            blue_count[qi] -= 1
                    
                    if E < best_E:
                        best_E = E
                        best_colors = colors.copy()
                        
                T *= cooling
                
        if best_colors is not None and best_E == 0:
            return best_colors.tolist()
            
        return None

    best_n = 1
    best_colors = [0]
    
    # Try increasing n
    for n in range(2, 35):
        if time.time() > deadline - 10:
            break
            
        result = solve_n(n)
        if result is not None:
            best_n = n
            best_colors = result
        else:
            # If we fail for n, we might still try n+1 if we have time?
            # Usually if we can't find n, we can't find n+1.
            # But maybe a different restart works?
            # For now, break to save time for larger n?
            # No, if we can't find n=10, we can't find n=11.
            break
            
    return {'n': best_n, 'colors': best_colors}
