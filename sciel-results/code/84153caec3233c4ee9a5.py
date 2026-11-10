"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    def get_triple_list(n):
        """Return list of all triples (a,b,c) a<b<c in lex order."""
        return list(combinations(range(n), 3))
    
    def build_constraint_structures(n):
        """Precompute all 4-set and 5-set triple index arrays."""
        triples = get_triple_list(n)
        num_triples = len(triples)
        idx_map = {}
        for i, (a, b, c) in enumerate(triples):
            idx_map[(a, b, c)] = i
        
        # 4-sets: each has 4 triples
        four_sets = list(combinations(range(n), 4))
        if four_sets:
            tri_idx_4 = np.empty((len(four_sets), 4), dtype=np.int32)
            for i, (a, b, c, d) in enumerate(four_sets):
                tri_idx_4[i, 0] = idx_map[(a, b, c)]
                tri_idx_4[i, 1] = idx_map[(a, b, d)]
                tri_idx_4[i, 2] = idx_map[(a, c, d)]
                tri_idx_4[i, 3] = idx_map[(b, c, d)]
        else:
            tri_idx_4 = None
        
        # 5-sets: each has 10 triples
        five_sets = list(combinations(range(n), 5))
        if five_sets:
            tri_idx_5 = np.empty((len(five_sets), 10), dtype=np.int32)
            for i, (a, b, c, d, e) in enumerate(five_sets):
                tri_idx_5[i, 0] = idx_map[(a, b, c)]
                tri_idx_5[i, 1] = idx_map[(a, b, d)]
                tri_idx_5[i, 2] = idx_map[(a, b, e)]
                tri_idx_5[i, 3] = idx_map[(a, c, d)]
                tri_idx_5[i, 4] = idx_map[(a, c, e)]
                tri_idx_5[i, 5] = idx_map[(a, d, e)]
                tri_idx_5[i, 6] = idx_map[(b, c, d)]
                tri_idx_5[i, 7] = idx_map[(b, c, e)]
                tri_idx_5[i, 8] = idx_map[(b, d, e)]
                tri_idx_5[i, 9] = idx_map[(c, d, e)]
        else:
            tri_idx_5 = None
        
        # Build reverse mapping: for each triple, which 4-sets and 5-sets it belongs to
        triple_to_4sets = [[] for _ in range(num_triples)]
        if tri_idx_4 is not None:
            for s in range(len(four_sets)):
                for t in range(4):
                    triple_to_4sets[tri_idx_4[s, t]].append(s)
        
        triple_to_5sets = [[] for _ in range(num_triples)]
        if tri_idx_5 is not None:
            for s in range(len(five_sets)):
                for t in range(10):
                    triple_to_5sets[tri_idx_5[s, t]].append(s)
        
        # Convert to arrays for faster access
        triple_to_4sets_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in triple_to_4sets]
        triple_to_5sets_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in triple_to_5sets]
        
        return num_triples, tri_idx_4, tri_idx_5, triple_to_4sets_arr, triple_to_5sets_arr, idx_map, triples
    
    def count_violations_fast(colors, tri_idx_4, tri_idx_5):
        """Count total violations (red K4 + blue K5)."""
        nv = 0
        if tri_idx_4 is not None and len(tri_idx_4) > 0:
            all_red = np.all(colors[tri_idx_4] == 0, axis=1)
            nv += int(np.sum(all_red))
        if tri_idx_5 is not None and len(tri_idx_5) > 0:
            all_blue = np.all(colors[tri_idx_5] == 1, axis=1)
            nv += int(np.sum(all_blue))
        return nv
    
    def local_search(n, max_steps=200000, seed=None):
        """Local search with smart move selection to find valid coloring."""
        if seed is not None:
            rng = random.Random(seed)
            np_rng = np.random.RandomState(seed)
        else:
            rng = random.Random()
            np_rng = np.random.RandomState()
        
        num_triples, tri_idx_4, tri_idx_5, t2s4, t2s5, idx_map, triples = build_constraint_structures(n)
        
        if num_triples == 0:
            return np.array([], dtype=np.int8)
        
        # Random initialization
        colors = np_rng.randint(0, 2, size=num_triples).astype(np.int8)
        
        current_violations = count_violations_fast(colors, tri_idx_4, tri_idx_5)
        best_violations = current_violations
        best_colors = colors.copy()
        
        step = 0
        no_improve = 0
        patience = 50000
        
        while step < max_steps and time_left() > 5.0:
            if current_violations == 0:
                return colors
            
            # Pick a triple involved in a violation if possible
            if current_violations > 0:
                # Try to pick from violated constraints
                violated_triples = set()
                if tri_idx_4 is not None and len(tri_idx_4) > 0:
                    red_4sets = np.where(np.all(colors[tri_idx_4] == 0, axis=1))[0]
                    for s in red_4sets[:min(20, len(red_4sets))]:
                        for t in range(4):
                            violated_triples.add(int(tri_idx_4[s, t]))
                if tri_idx_5 is not None and len(tri_idx_5) > 0:
                    blue_5sets = np.where(np.all(colors[tri_idx_5] == 1, axis=1))[0]
                    for s in blue_5sets[:min(20, len(blue_5sets))]:
                        for t in range(10):
                            violated_triples.add(int(tri_idx_5[s, t]))
                
                if violated_triples:
                    t_idx = rng.choice(list(violated_triples))
                else:
                    t_idx = np_rng.randint(0, num_triples)
            else:
                t_idx = np_rng.randint(0, num_triples)
            
            old_color = colors[t_idx]
            colors[t_idx] = 1 - old_color
            
            new_violations = count_violations_fast(colors, tri_idx_4, tri_idx_5)
            delta = new_violations - current_violations
            
            if delta <= 0:
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Accept with some probability to escape local minima
                accept_prob = min(1.0, np.exp(-delta / max(0.1, 1.0 - step / max_steps)))
                if rng.random() < accept_prob:
                    current_violations = new_violations
                    no_improve += 1
                else:
                    colors[t_idx] = old_color
                    no_improve += 1
            
            step += 1
            
            if no_improve > patience:
                # Restart with new random initialization
                colors = np_rng.randint(0, 2, size=num_triples).astype(np.int8)
                current_violations = count_violations_fast(colors, tri_idx_4, tri_idx_5)
                no_improve = 0
        
        if best_violations == 0:
            return best_colors
        return None
    
    def hill_climbing_from(colors, n, max_steps=100000, seed=None):
        """Hill climbing from a given starting coloring."""
        if seed is not None:
            np_rng = np.random.RandomState(seed)
        else:
            np_rng = np.random.RandomState()
        
        num_triples, tri_idx_4, tri_idx_5, t2s4, t2s5, idx_map, triples = build_constraint_structures(n)
        colors = colors.copy()
        
        current_violations = count_violations_fast(colors, tri_idx_4, tri_idx_5)
        best_violations = current_violations
        best_colors = colors.copy()
        
        step = 0
        no_improve = 0
        patience = 30000
        
        while step < max_steps and time_left() > 5.0:
            if current_violations == 0:
                return colors
            
            # Pick random triple
            t_idx = np_rng.randint(0, num_triples)
            old_color = colors[t_idx]
            colors[t_idx] = 1 - old_color
            
            new_violations = count_violations_fast(colors, tri_idx_4, tri_idx_5)
            delta = new_violations - current_violations
            
            if delta < 0:
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            elif delta == 0:
                no_improve += 1
            else:
                colors[t_idx] = old_color
                no_improve += 1
            
            step += 1
            
            if no_improve > patience:
                break
        
        if best_violations == 0:
            return best_colors
        return None
    
    # Strategy: try to find valid colorings for increasing n
    # The published construction has n=34, so aim for that
    best_n = 0
    best_colors = None
    
    # Try n values from high to low, with multiple attempts each
    for n in range(34, 9, -1):
        if time_left() < 10.0:
            break
        
        found = False
        num_attempts = min(5, max(1, int(time_left() // 10)))
        
        for attempt in range(num_attempts):
            if time_left() < 5.0:
                break
            
            result = local_search(n, max_steps=50000, seed=attempt * 12345 + n)
            if result is not None:
                best_n = n
                best_colors = result
                found = True
                break
        
        if found:
            # Try to extend further
            continue
        else:
            # If we can't find for this n, try lower n
            continue
    
    # If we haven't found anything yet, try smaller n values
    if best_n == 0:
        for n in range(20, 8, -1):
            if time_left() < 5.0:
                break
            
            found = False
            for attempt in range(3):
                if time_left() < 3.0:
                    break
                result = local_search(n, max_steps=30000, seed=attempt * 999 + n)
                if result is not None:
                    best_n = n
                    best_colors = result
                    found = True
                    break
            
            if found:
                break
    
    # Fallback: use a simple valid coloring for small n
    if best_n == 0 or best_colors is None:
        n = 9
        triples_list = get_triple_list(n)
        # Simple coloring: color based on sum mod 2
        best_colors = np.array([(a + b + c) % 2 for a, b, c in triples_list], dtype=np.int8)
        # Verify it's valid
        num_triples, tri_idx_4, tri_idx_5, _, _, _, _ = build_constraint_structures(n)
        v = count_violations_fast(best_colors, tri_idx_4, tri_idx_5)
        if v == 0:
            best_n = n
        else:
            # Try all zeros except make sure no red K4
            # For n=9, just use a known valid pattern
            best_colors = np.zeros(num_triples, dtype=np.int8)
            # Flip some to avoid red K4
            for i in range(0, num_triples, 3):
                best_colors[i] = 1
            v = count_violations_fast(best_colors, tri_idx_4, tri_idx_5)
            if v == 0:
                best_n = n
            else:
                # Last resort: n=3 always valid
                best_n = 3
                best_colors = np.array([0, 0, 0], dtype=np.int8)
    
    # Convert to list format
    triples_list = get_triple_list(best_n)
    color_list = best_colors.tolist()
    
    return {'n': best_n, 'colors': color_list}

# EVOLVE-BLOCK-END
