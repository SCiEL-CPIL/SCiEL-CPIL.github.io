import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_n = 0
    best_colors = None
    
    # We know n=34 is possible. Let's try to find it.
    # We can try n from 34 down to 10, or just try 34 and if it fails, try lower.
    # But the task says "Maximize n". So we should try the highest possible first.
    # The published construction has 34 points.
    
    # Let's try n=34 first. If we find a valid coloring, we return it.
    # If we don't find it in a reasonable time, we try n=33, etc.
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid coloring for this n
        found = False
        # Try a few different random seeds and initializations
        for seed in [42, 123, 999, 777, 111, 222, 333, 444]:
            if time.time() > deadline - 5:
                break
                
            colors = _find_coloring(n, deadline, seed)
            if colors is not None:
                # Verify
                if _check_validity(n, colors):
                    best_n = n
                    best_colors = colors
                    found = True
                    # Since we are iterating downwards, if we find one for n, 
                    # it's the max we've found so far. 
                    # We should return immediately if we want to maximize n?
                    # The loop goes 34, 33, ...
                    # So if we find 34, we are done.
                    # If we fail 34, we try 33.
                    # We can break the outer loop if we want to just return the first found?
                    # No, we want the MAX n. So if we find 34, we return.
                    # If we fail 34, we continue to 33.
                    # So we should break the inner loop, and the outer loop will continue to 33?
                    # No, if we find 34, we are done.
                    # If we fail 34, we try 33.
                    # So we break the inner loop, and the outer loop continues.
                    # But if we find 34, we can return immediately.
                    if n == 34:
                        return {'n': best_n, 'colors': best_colors}
                    break
        
        if found:
            # We found a valid coloring for n. 
            # Since we are going down from 34, this is the best we've found.
            # But maybe we can find 35? The problem says max 40, but published is 34.
            # R(4,5;3) is likely 35 or 36. 
            # Actually, R(4,5;3) = 35 is a known result? 
            # Let's check. 
            # If R(4,5;3) = 35, then there is no valid coloring for 35.
            # So 34 is the max.
            # We can return the best found.
            # But we should keep trying higher n? No, we are going down.
            # So if we find 34, we return.
            # If we fail 34, we try 33.
            # So we should return as soon as we find a valid coloring for the highest n we try.
            # Wait, if we try 34 and fail, we try 33. If we find 33, we return 33.
            # This is correct.
            return {'n': best_n, 'colors': best_colors}
            
    # If we didn't find anything (should not happen), return a trivial one
    if best_n == 0:
        best_n = 1
        best_colors = []
        
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline, seed):
    random.seed(seed)
    
    num_triples = n * (n - 1) * (n - 2) // 6
    triple_list = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    
    # Precompute 4-sets and 5-sets
    quads = list(combinations(range(n), 4))
    quads_triples_idx = []
    for quad in quads:
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quads_triples_idx.append([triple_to_idx[t] for t in ts])
        
    quints = list(combinations(range(n), 5))
    quints_triples_idx = []
    for quint in quints:
        ts = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quints_triples_idx.append([triple_to_idx[t] for t in ts])
        
    # Map triple index to list of quad indices and quint indices
    triple_to_quads = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quads_triples_idx):
        for t_idx in qt:
            triple_to_quads[t_idx].append(qi)
            
    triple_to_quints = [[] for _ in range(num_triples)]
    for qi, qt in enumerate(quints_triples_idx):
        for t_idx in qt:
            triple_to_quints[t_idx].append(qi)
            
    # Initial random coloring
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Initialize violation counts
    red_viol = [0] * len(quads)
    blue_viol = [0] * len(quints)
    total_viol = 0
    
    for qi, qt in enumerate(quads_triples_idx):
        if all(colors[t] == 0 for t in qt):
            red_viol[qi] = 1
            total_viol += 1
            
    for qi, qt in enumerate(quints_triples_idx):
        if all(colors[t] == 1 for t in qt):
            blue_viol[qi] = 1
            total_viol += 1
            
    # Local search
    T = 1.0
    T_min = 0.001
    cooling = 0.9999
    max_iters = 500000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if total_viol == 0:
            return colors
            
        # Pick a random triple to flip
        idx = random.randint(0, num_triples - 1)
        old_color = colors[idx]
        new_color = 1 - old_color
        
        # Calculate delta
        delta = 0
        
        # Check affected quads
        for qi in triple_to_quads[idx]:
            qt = quads_triples_idx[qi]
            # Check if other 3 are red
            others_red = all(colors[t] == 0 for t in qt if t != idx)
            
            if old_color == 0: # Was red, now blue
                if red_viol[qi] == 1:
                    # Was all-red, now not
                    red_viol[qi] = 0
                    delta -= 1
                # Flipping red to blue cannot create a new red violation
            else: # Was blue, now red
                if red_viol[qi] == 0 and others_red:
                    # Was not all-red, now is
                    red_viol[qi] = 1
                    delta += 1
                # Flipping blue to red cannot destroy a red violation (it wasn't all-red before)
                
        # Check affected quints
        for qi in triple_to_quints[idx]:
            qt = quints_triples_idx[qi]
            # Check if other 9 are blue
            others_blue = all(colors[t] == 1 for t in qt if t != idx)
            
            if old_color == 1: # Was blue, now red
                if blue_viol[qi] == 1:
                    # Was all-blue, now not
                    blue_viol[qi] = 0
                    delta -= 1
                # Flipping blue to red cannot create a new blue violation
            else: # Was red, now blue
                if blue_viol[qi] == 0 and others_blue:
                    # Was not all-blue, now is
                    blue_viol[qi] = 1
                    delta += 1
                # Flipping red to blue cannot destroy a blue violation (it wasn't all-blue before)
                
        total_viol += delta
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
            colors[idx] = new_color
        else:
            # Revert
            total_viol -= delta
            # We don't need to revert red_viol/blue_viol explicitly if we just use them for decision,
            # but we must keep them consistent with colors.
            # Since we didn't accept, colors[idx] is still old_color.
            # But we modified red_viol/blue_viol in the calculation.
            # We must revert them.
            for qi in triple_to_quads[idx]:
                qt = quads_triples_idx[qi]
                others_red = all(colors[t] == 0 for t in qt if t != idx)
                if old_color == 0:
                    if red_viol[qi] == 0 and others_red:
                        red_viol[qi] = 1
                    elif red_viol[qi] == 1 and not others_red:
                        red_viol[qi] = 0
                else:
                    if red_viol[qi] == 1 and not others_red:
                        red_viol[qi] = 0
                    elif red_viol[qi] == 0 and others_red:
                        red_viol[qi] = 1
            for qi in triple_to_quints[idx]:
                qt = quints_triples_idx[qi]
                others_blue = all(colors[t] == 1 for t in qt if t != idx)
                if old_color == 1:
                    if blue_viol[qi] == 0 and others_blue:
                        blue_viol[qi] = 1
                    elif blue_viol[qi] == 1 and not others_blue:
                        blue_viol[qi] = 0
                else:
                    if blue_viol[qi] == 1 and not others_blue:
                        blue_viol[qi] = 0
                    elif blue_viol[qi] == 0 and others_blue:
                        blue_viol[qi] = 1
                        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return None

def _check_validity(n, colors):
    triple_to_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_to_idx[(a, b, c)] = idx
                idx += 1
                
    for quad in combinations(range(n), 4):
        triples = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        if all(colors[triple_to_idx[t]] == 0 for t in triples):
            return False
            
    for quint in combinations(range(n), 5):
        triples_in_quint = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    triples_in_quint.append((quint[i], quint[j], quint[k]))
        if all(colors[triple_to_idx[t]] == 1 for t in triples_in_quint):
            return False
            
    return True
