"""Rewritten program: local search with incremental violation counting to minimize colors used."""
import time
import random

# EVOLVE-BLOCK-START

def _all_progressions(n=128):
    """Precompute all nonconstant 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 0<=a, a+3d<n."""
    progs = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    return progs

_PROGS = None

def _get_progs():
    global _PROGS
    if _PROGS is None:
        _PROGS = _all_progressions(128)
    return _PROGS

def _count_violations(colors, progs=None):
    """Count number of 4-APs that use fewer than 3 distinct colors."""
    if progs is None:
        progs = _get_progs()
    count = 0
    for a, b, c, d in progs:
        cols = {colors[a], colors[b], colors[c], colors[d]}
        if len(cols) < 3:
            count += 1
    return count

def _local_search(colors, progs, max_time=75.0, seed=42):
    """Local search to reduce violations and then reduce color count."""
    rng = random.Random(seed)
    n = len(colors)
    colors = list(colors)
    violations = _count_violations(colors, progs)
    
    # Precompute for each position, which progressions it participates in
    # For efficiency, we'll track violations per progression
    prog_violation = [False] * len(progs)
    for i, (a, b, c, d) in enumerate(progs):
        prog_violation[i] = len({colors[a], colors[b], colors[c], colors[d]}) < 3
    
    total_violations = sum(prog_violation)
    
    start_time = time.time()
    best_violations = total_violations
    best_colors = list(colors)
    best_color_count = len(set(colors))
    
    # Phase 1: Reduce violations to 0
    while total_violations > 0 and time.time() - start_time < max_time - 5:
        # Pick a random violated progression
        violated_indices = [i for i, v in enumerate(prog_violation) if v]
        if not violated_indices:
            break
        pi = rng.choice(violated_indices)
        a, b, c, d = progs[pi]
        pos = rng.choice([a, b, c, d])
        
        old_color = colors[pos]
        new_color = rng.randint(0, 127)
        if new_color == old_color:
            new_color = (new_color + 1) % 128
        
        # Compute delta in violations
        delta = 0
        # Check all progressions containing pos
        for j, (pa, pb, pc, pd) in enumerate(progs):
            if pos in (pa, pb, pc, pd):
                old_set = {colors[pa], colors[pb], colors[pc], colors[pd]}
                old_v = len(old_set) < 3
                # Apply new color
                if pos == pa:
                    new_set = {new_color, colors[pb], colors[pc], colors[pd]}
                elif pos == pb:
                    new_set = {colors[pa], new_color, colors[pc], colors[pd]}
                elif pos == pc:
                    new_set = {colors[pa], colors[pb], new_color, colors[pd]}
                else:
                    new_set = {colors[pa], colors[pb], colors[pc], new_color}
                new_v = len(new_set) < 3
                if old_v and not new_v:
                    delta -= 1
                elif not old_v and new_v:
                    delta += 1
        
        if delta <= 0:
            colors[pos] = new_color
            # Update violation status
            for j, (pa, pb, pc, pd) in enumerate(progs):
                if pos in (pa, pb, pc, pd):
                    old_set = {colors[pa], colors[pb], colors[pc], colors[pd]}
                    prog_violation[j] = len(old_set) < 3
            total_violations = sum(prog_violation)
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = list(colors)
                best_color_count = len(set(colors))
        else:
            # Accept with some probability (simulated annealing-like)
            if rng.random() < 0.1:
                colors[pos] = new_color
                for j, (pa, pb, pc, pd) in enumerate(progs):
                    if pos in (pa, pb, pc, pd):
                        old_set = {colors[pa], colors[pb], colors[pc], colors[pd]}
                        prog_violation[j] = len(old_set) < 3
                total_violations = sum(prog_violation)
    
    # Phase 2: Reduce color count (only if valid)
    if total_violations == 0:
        while time.time() - start_time < max_time - 2:
            used_colors = sorted(set(colors))
            if len(used_colors) <= 1:
                break
            # Try to eliminate one color
            target = rng.choice(used_colors)
            positions_with_target = [i for i in range(n) if colors[i] == target]
            
            # Try to recolor all positions with target to other colors
            success = True
            new_colors = list(colors)
            for pos in positions_with_target:
                # Find a color that doesn't create a violation
                found = False
                for new_c in used_colors:
                    if new_c == target:
                        continue
                    # Check if changing pos to new_c creates any violation
                    ok = True
                    for j, (pa, pb, pc, pd) in enumerate(progs):
                        if pos in (pa, pb, pc, pd):
                            old_set = {new_colors[pa], new_colors[pb], new_colors[pc], new_colors[pd]}
                            if pos == pa:
                                test_set = {new_c, new_colors[pb], new_colors[pc], new_colors[pd]}
                            elif pos == pb:
                                test_set = {new_colors[pa], new_c, new_colors[pc], new_colors[pd]}
                            elif pos == pc:
                                test_set = {new_colors[pa], new_colors[pb], new_c, new_colors[pd]}
                            else:
                                test_set = {new_colors[pa], new_colors[pb], new_colors[pc], new_c}
                            if len(test_set) < 3:
                                ok = False
                                break
                    if ok:
                        new_colors[pos] = new_c
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                colors = new_colors
                total_violations = 0
                if len(set(colors)) < best_color_count:
                    best_color_count = len(set(colors))
                    best_colors = list(colors)
    
    # Return best found
    if best_violations == 0:
        return best_colors
    return best_colors

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin
    
    progs = _get_progs()
    n = 128
    
    # Start with a simple 3-color pattern
    # colors[i] = i % 3
    colors = [i % 3 for i in range(n)]
    
    # Check initial violations
    v = _count_violations(colors, progs)
    
    # If already valid, try to reduce colors
    if v == 0:
        # Try 2-color patterns
        for c1 in range(128):
            for c2 in range(128):
                if c1 == c2:
                    continue
                test_colors = [c1 if i % 2 == 0 else c2 for i in range(n)]
                tv = _count_violations(test_colors, progs)
                if tv == 0:
                    return test_colors
    
    # Run local search with multiple seeds
    best_result = None
    best_score = float('inf')
    
    for seed in [42, 123, 777, 2024, 999, 555, 111, 888]:
        if time.time() - start_time > deadline:
            break
        
        # Start from different initial configurations
        if seed == 42:
            init = [i % 3 for i in range(n)]
        elif seed == 123:
            init = [i % 4 for i in range(n)]
        elif seed == 777:
            init = [random.Random(seed).randint(0, 2) for i in range(n)]
        elif seed == 2024:
            init = [random.Random(seed).randint(0, 3) for i in range(n)]
        else:
            init = [random.Random(seed).randint(0, 5) for i in range(n)]
        
        result = _local_search(init, progs, max_time=min(deadline - (time.time() - start_time), 10.0), seed=seed)
        
        if result is not None:
            v = _count_violations(result, progs)
            if v == 0:
                score = len(set(result))
                if score < best_score:
                    best_score = score
                    best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the initial 3-color pattern if it's valid
    colors = [i % 3 for i in range(n)]
    if _count_violations(colors, progs) == 0:
        return colors
    
    # Last resort: use the seed program's approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1) if x-1-3*d >= 0)
                and all(any(colors[x-1-j*d] != color for j in (1, 2)) 
                        for d in range(1, (x-1)//2 + 1) if x-1-2*d >= 0)):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
