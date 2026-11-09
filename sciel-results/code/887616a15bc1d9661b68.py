import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    deadline = time.time() + 590  # safety margin
    
    d = 11
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)  # C(11,4) = 330
    
    # Precompute conflict: two 4-sets conflict if they share >= 3 indices
    # conflict[i] = set of j that conflict with i
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
    
    # Precompute pairwise conflict using bit operations
    # Two 4-sets share >= 3 indices iff popcount(mask_i & mask_j) >= 3
    conflict = [0] * n_sets  # bitmask of conflicting sets
    for i in range(n_sets):
        mi = masks[i]
        for j in range(i + 1, n_sets):
            mj = masks[j]
            inter = mi & mj
            # Count bits
            c = 0
            tmp = inter
            while tmp:
                tmp &= tmp - 1
                c += 1
            if c >= 3:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Simulated annealing / local search for maximum independent set
    # Start with a good greedy solution, then do local search
    
    rng = random.Random(42)
    
    def greedy_init():
        """Greedy: repeatedly pick the set with fewest conflicts among remaining."""
        selected = set()
        remaining = set(range(n_sets))
        while remaining:
            # Pick set with minimum degree in remaining
            best = None
            best_deg = float('inf')
            for s in remaining:
                deg = bin(conflict[s] & ((1 << n_sets) - 1)).count('1')
                # Only count conflicts with remaining
                # Actually, let's just count conflicts with remaining
                rem_mask = 0
                for r in remaining:
                    rem_mask |= (1 << r)
                deg = bin(conflict[s] & rem_mask).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best = s
            selected.add(best)
            remaining.discard(best)
            # Remove all sets conflicting with best
            to_remove = conflict[best] & ((1 << n_sets) - 1)
            for r in list(remaining):
                if to_remove & (1 << r):
                    remaining.discard(r)
        return selected
    
    def local_search(initial, time_limit=50):
        """Simulated annealing on independent set."""
        current = set(initial)
        best = set(current)
        best_size = len(current)
        
        # Build adjacency for fast lookup
        adj = [set() for _ in range(n_sets)]
        for i in range(n_sets):
            for j in range(i+1, n_sets):
                if conflict[i] & (1 << j):
                    adj[i].add(j)
                    adj[j].add(i)
        
        start = time.time()
        temperature = 1.0
        cooling = 0.9995
        
        while time.time() - start < time_limit and temperature > 0.001:
            # Try to add a random unselected set
            unselected = [s for s in range(n_sets) if s not in current]
            if not unselected:
                break
            
            s = rng.choice(unselected)
            # Check if s conflicts with any selected
            conflicts_with_selected = adj[s] & current
            if not conflicts_with_selected:
                current.add(s)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
            else:
                # Try swap: remove one conflicting set, add s
                if len(conflicts_with_selected) == 1:
                    remove = next(iter(conflicts_with_selected))
                    # Check if removing remove and adding s is valid
                    # s must not conflict with any other selected
                    other_conflicts = adj[s] - {remove}
                    if not (other_conflicts & current):
                        current.discard(remove)
                        current.add(s)
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                elif len(conflicts_with_selected) >= 2:
                    # Try to remove all conflicts and add s (only if net gain)
                    if len(conflicts_with_selected) < 1:  # never happens here
                        pass
                    # Skip for now, too complex
            temperature *= cooling
        
        return best
    
    # Multiple restarts with different seeds
    best_solution = set()
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        # Greedy init with random ordering
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = set()
        for s in order:
            if not (adj[s] & selected) if 'adj' in dir() else True:
                # Check conflicts
                valid = True
                for t in selected:
                    if conflict[s] & (1 << t):
                        valid = False
                        break
                if valid:
                    selected.add(s)
        
        # Local search
        improved = local_search(selected, time_limit=15)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Generate points from best_solution
    points = []
    for s in best_solution:
        i, j, k, l = all_4sets[s]
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
