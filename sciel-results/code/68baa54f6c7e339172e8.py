import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for 150s limit
    
    best_edges = None
    best_n = None
    best_size = float('inf')
    
    # Try a few n values
    for n in [11, 10, 12, 9]:
        if time.time() - start_time > time_limit:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        # Precompute for each edge, the set of colorings it breaks.
        # Colorings are integers 0 to 2^n - 1.
        edge_to_colorings = []
        for e in all_edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
            
            broken = set()
            for c in range(1 << n):
                intersection = c & e_mask
                if intersection == 0 or intersection == e_mask:
                    broken.add(c)
            edge_to_colorings.append(broken)
        
        # Randomized Greedy Set Cover
        target_colorings = 1 << n
        all_colorings_set = set(range(target_colorings))
        
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            uncovered = all_colorings_set.copy()
            chosen = []
            
            # Greedy selection
            while uncovered:
                best_edge = -1
                best_covered = 0
                # Randomize order to break ties
                indices = list(range(len(all_edges)))
                random.shuffle(indices)
                
                for i in indices:
                    if i in chosen:
                        continue
                    covered = len(uncovered & edge_to_colorings[i])
                    if covered > best_covered:
                        best_covered = covered
                        best_edge = i
                        if best_covered == len(uncovered):
                            break
                
                if best_edge == -1:
                    break # Should not happen
                
                chosen.append(best_edge)
                uncovered -= edge_to_colorings[best_edge]
            
            if not uncovered:
                # We have a cover. Now try to minimize.
                # Remove edges if possible.
                changed = True
                while changed:
                    changed = False
                    for i in range(len(chosen)):
                        # Try removing chosen[i]
                        new_chosen = chosen[:i] + chosen[i+1:]
                        # Check if new_chosen still covers all
                        covered = set()
                        for j in new_chosen:
                            covered |= edge_to_colorings[j]
                        if covered == all_colorings_set:
                            chosen = new_chosen
                            changed = True
                            break
                
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_edges = [list(all_edges[i]) for i in chosen]
                    best_n = n
                    # If we hit the known optimum, we can stop early?
                    if best_size <= 51:
                        # We found a very good solution.
                        break
        
        if best_size <= 51:
            break # Found optimal or better
    
    if best_edges is None:
        # Fallback to seed solution
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    else:
        return {'n': best_n, 'edges': best_edges}
