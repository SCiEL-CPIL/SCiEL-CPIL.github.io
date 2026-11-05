import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def get_triples(n):
        return list(combinations(range(n), 3))

    def build_structure(n):
        """
        Builds the necessary data structures for fast violation counting.
        Returns a dictionary with keys:
        'triples', 'tri_index', 'set4_triples', 'set5_triples', 
        'triple_to_set4', 'triple_to_set5', 'num_triples', 'num_set4', 'num_set5'
        """
        triples = list(combinations(range(n), 3))
        tri_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # 4-sets
        sets_4 = list(combinations(range(n), 4))
        set4_triples = []
        for s in sets_4:
            a, b, c, d = s
            ts = [
                tri_index[(a, b, c)],
                tri_index[(a, b, d)],
                tri_index[(a, c, d)],
                tri_index[(b, c, d)]
            ]
            set4_triples.append(ts)
        num_set4 = len(sets_4)
        
        # 5-sets
        sets_5 = list(combinations(range(n), 5))
        set5_triples = []
        for s in sets_5:
            ts = []
            # 10 triples in a 5-set
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(tri_index[(s[i], s[j], s[k])])
            set5_triples.append(ts)
        num_set5 = len(sets_5)
        
        # Reverse mappings
        triple_to_set4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set4_triples):
            for t_idx in ts:
                triple_to_set4[t_idx].append(i)
                
        triple_to_set5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(set5_triples):
            for t_idx in ts:
                triple_to_set5[t_idx].append(i)
                
        return {
            'n': n,
            'triples': triples,
            'set4_triples': set4_triples,
            'set5_triples': set5_triples,
            'triple_to_set4': triple_to_set4,
            'triple_to_set5': triple_to_set5,
            'num_triples': num_triples,
            'num_set4': num_set4,
            'num_set5': num_set5
        }

    def init_violations(colors, struct):
        """Initialize violation counts for 4-sets and 5-sets."""
        num_set4 = struct['num_set4']
        num_set5 = struct['num_set5']
        
        # viol4[i] = number of red triples in 4-set i
        viol4 = np.zeros(num_set4, dtype=np.int32)
        for i, ts in enumerate(struct['set4_triples']):
            for t_idx in ts:
                if colors[t_idx] == 0:
                    viol4[i] += 1
                    
        # viol5[i] = number of blue triples in 5-set i
        viol5 = np.zeros(num_set5, dtype=np.int32)
        for i, ts in enumerate(struct['set5_triples']):
            for t_idx in ts:
                if colors[t_idx] == 1:
                    viol5[i] += 1
                    
        # Total violations
        total_v = np.sum(viol4 == 4) + np.sum(viol5 == 10)
        
        return viol4, viol5, int(total_v)

    def delta_violations(t_idx, new_color, colors, struct, viol4, viol5):
        """
        Calculate change in total violations if triple t_idx is flipped to new_color.
        Does not modify the state.
        """
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
            
        d_v = 0
        
        # Update 4-sets containing t_idx
        # Red K4 violation if count of red == 4
        for s4_idx in struct['triple_to_set4'][t_idx]:
            cnt = viol4[s4_idx]
            # If we flip from Red (0) to Blue (1), red count decreases by 1
            # If we flip from Blue (1) to Red (0), red count increases by 1
            if old_color == 0:
                new_cnt = cnt - 1
            else:
                new_cnt = cnt + 1
                
            viol_before = 1 if cnt == 4 else 0
            viol_after = 1 if new_cnt == 4 else 0
            d_v += (viol_after - viol_before)
            
        # Update 5-sets containing t_idx
        # Blue K5 violation if count of blue == 10
        for s5_idx in struct['triple_to_set5'][t_idx]:
            cnt = viol5[s5_idx]
            # If we flip from Blue (1) to Red (0), blue count decreases by 1
            # If we flip from Red (0) to Blue (1), blue count increases by 1
            if old_color == 1:
                new_cnt = cnt - 1
            else:
                new_cnt = cnt + 1
                
            viol_before = 1 if cnt == 10 else 0
            viol_after = 1 if new_cnt == 10 else 0
            d_v += (viol_after - viol_before)
            
        return d_v

    def apply_flip(t_idx, new_color, colors, struct, viol4, viol5):
        """
        Apply the flip to the state (colors, viol4, viol5).
        Returns the change in total violations.
        """
        old_color = colors[t_idx]
        if old_color == new_color:
            return 0
            
        d_v = 0
        
        # Update 4-sets
        for s4_idx in struct['triple_to_set4'][t_idx]:
            cnt = viol4[s4_idx]
            if old_color == 0:
                viol4[s4_idx] = cnt - 1
            else:
                viol4[s4_idx] = cnt + 1
                
            viol_before = 1 if cnt == 4 else 0
            viol_after = 1 if viol4[s4_idx] == 4 else 0
            d_v += (viol_after - viol_before)
            
        # Update 5-sets
        for s5_idx in struct['triple_to_set5'][t_idx]:
            cnt = viol5[s5_idx]
            if old_color == 1:
                viol5[s5_idx] = cnt - 1
            else:
                viol5[s5_idx] = cnt + 1
                
            viol_before = 1 if cnt == 10 else 0
            viol_after = 1 if viol5[s5_idx] == 10 else 0
            d_v += (viol_after - viol_before)
            
        colors[t_idx] = new_color
        return d_v

    def simulated_annealing(n, max_time, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        struct = build_structure(n)
        num_triples = struct['num_triples']
        
        # Initialize colors
        colors = np.random.randint(0, 2, size=num_triples)
        viol4, viol5, total_v = init_violations(colors, struct)
        
        best_colors = colors.copy()
        best_v = total_v
        
        T = 1.0
        T_min = 0.01
        alpha = 0.9995 # Cooling rate
        
        start_sa = time.time()
        
        while time.time() - start_sa < max_time:
            if best_v == 0:
                break
                
            # Pick a random triple to flip
            t_idx = np.random.randint(0, num_triples)
            new_color = 1 - colors[t_idx]
            
            d_v = delta_violations(t_idx, new_color, colors, struct, viol4, viol5)
            
            # Acceptance criterion
            if d_v < 0:
                # Accept if it improves
                apply_flip(t_idx, new_color, colors, struct, viol4, viol5)
                total_v += d_v
            else:
                # Accept with probability exp(-d_v / T)
                if T > T_min:
                    prob = np.exp(-d_v / T)
                    if np.random.random() < prob:
                        apply_flip(t_idx, new_color, colors, struct, viol4, viol5)
                        total_v += d_v
            
            if total_v < best_v:
                best_v = total_v
                best_colors = colors.copy()
                
            T *= alpha
            
        return best_colors, best_v

    best_n = 9
    best_colors = None
    
    # Try to find valid colorings for high n first, then fallback
    # We know n=34 is possible. Let's try 34, 35, 36.
    # If we fail on 34, try 33, etc.
    
    # Time budget allocation:
    # We have ~145s.
    # Let's allocate time to attempts.
    # Attempt 34: 60s
    # Attempt 35: 30s
    # Attempt 36: 10s
    # Fallback 33: 20s
    # Fallback 32: 15s
    # etc.
    
    attempts = [
        (34, 60.0),
        (35, 30.0),
        (36, 10.0),
        (33, 20.0),
        (32, 15.0),
        (31, 10.0)
    ]
    
    # If we have a known good seed for lower n, we could use it, but random restarts are robust.
    
    for target_n, time_budget in attempts:
        if time.time() > deadline - 5:
            break
            
        # Try multiple restarts within the time budget
        # For n=34, SA is fast, so we can do many restarts.
        # For n=36, it might be harder.
        
        found_valid = False
        start_attempt = time.time()
        
        while time.time() - start_attempt < time_budget:
            if time.time() > deadline - 5:
                break
                
            # Use a new random seed for each restart
            seed = int(time.time() * 1000) % (2**32)
            
            # Reduce time per SA run if we've done many restarts
            # Simple heuristic: just run SA for a short fixed time or until convergence
            # Let's run SA for 5 seconds per restart
            colors, v = simulated_annealing(target_n, 5.0, seed)
            
            if v == 0:
                best_n = target_n
                best_colors = colors.tolist()
                found_valid = True
                break
                
        if found_valid:
            # If we found a valid coloring for this n, we can stop trying higher n
            # because we want to maximize n, and we are trying in descending order of "ambition"
            # Wait, the list is 34, 35, 36... 36 is higher than 34.
            # I should sort by n descending.
            pass
        
    # Re-evaluate logic: I want to find the MAX n.
    # So I should try 36, then 35, then 34.
    # If I find 36, I'm done. If I fail 36, I try 35. If I find 35, I'm done (unless I have time for 36 again? No, 35 is valid).
    # Actually, if I fail 36, I should try 35. If I find 35, that's a score of 35/34 > 1.
    # If I find 34, that's 1.0.
    
    # Let's restructure the loop to go from high to low.
    best_n = 9
    best_colors = None
    
    targets = [36, 35, 34, 33, 32, 31, 30]
    
    for target_n in targets:
        if time.time() > deadline - 5:
            break
            
        # Allocate remaining time to this target, but cap it to avoid starving lower targets if high ones are hopeless?
        # Actually, if we find a valid coloring for target_n, we can return it immediately as it's the highest we've found so far in this descending loop?
        # No, we want the HIGHEST n.
        # So if we try 36 and fail, we try 35. If we find 35, we can return 35?
        # Yes, because we are iterating from high to low. The first success is the highest n.
        
        # How much time to give each?
        # If we give 20s to 36, 20s to 35, 20s to 34...
        # Total 140s.
        
        time_budget = 20.0
        start_attempt = time.time()
        found = False
        
        while time.time() - start_attempt < time_budget:
            if time.time() > deadline - 2:
                break
                
            seed = random.randint(0, 2**31)
            # Run SA for a short burst
            colors, v = simulated_annealing(target_n, 4.0, seed)
            
            if v == 0:
                best_n = target_n
                best_colors = colors.tolist()
                found = True
                break
        
        if found:
            # We found a valid coloring for this n. Since we are going high->low, this is our best.
            break
    
    # If we didn't find anything, we need a fallback.
    # The seed program had n=9.
    # We should probably try to find a valid one for a smaller n if we failed all above.
    if best_colors is None:
        # Fallback to a simple heuristic or random search for small n
        # Try n=15 with more time?
        if time.time() < deadline - 10:
             colors, v = simulated_annealing(15, 10.0, 42)
             if v == 0:
                 best_n = 15
                 best_colors = colors.tolist()
        
        if best_colors is None:
             # Last resort: n=9 seed
             triples9 = get_triples(9)
             best_n = 9
             best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples9]

    # Verify and return
    triples_final = get_triples(best_n)
    if len(best_colors) != len(triples_final):
        # Mismatch, regenerate
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples_final]
        best_n = 9 # Just in case

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"Found n={res['n']}")
