"""Improved program: Multi-start randomized greedy + fast local search for B2[2] sets."""
import time
import random
import sys

def _fast_greedy():
    """Deterministic greedy: add smallest x that keeps B2[2] constraint."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        ok = True
        # Check sums with existing elements
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts.get(s2, 0) >= 2:
                ok = False
        if ok:
            chosen.append(x)
            # Update counts
            for y in chosen[:-1]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
    return chosen, counts

def _local_search_swap(chosen, counts, deadline, seed=42):
    """
    Fast local search using swaps.
    1. Try to ADD any valid element.
    2. If no ADD is possible, try SWAPs to change structure.
    """
    rng = random.Random(seed)
    chosen_set = set(chosen)
    current_size = len(chosen)
    best_size = current_size
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Phase 1: Try to ADD
        added = False
        available = [x for x in range(1, 1001) if x not in chosen_set]
        if available:
            # Check a random sample of available elements to speed up
            sample_size = min(50, len(available))
            samples = rng.sample(available, sample_size)
            
            for x in samples:
                ok = True
                for y in chosen:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        ok = False
                        break
                if ok:
                    s2 = 2 * x
                    if counts.get(s2, 0) >= 2:
                        ok = False
                
                if ok:
                    # Add x
                    chosen.append(x)
                    chosen_set.add(x)
                    for y in chosen[:-1]:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_chosen = list(chosen)
                        best_counts = dict(counts)
                    added = True
                    no_improve_count = 0
                    break
        
        if added:
            continue
        
        # Phase 2: Try SWAP
        if len(chosen) > 0:
            x = rng.choice(chosen)
            available = [x for x in range(1, 1001) if x not in chosen_set]
            if not available:
                break
            
            y = rng.choice(available)
            
            # Check validity of swap: (chosen - {x}) + {y}
            # We simulate the change in counts
            
            # 1. Remove x from counts
            for z in chosen:
                if z == x:
                    continue
                s = x + z
                counts[s] -= 1
            counts[2 * x] -= 1
            
            # 2. Check if y can be added
            ok = True
            for z in chosen:
                if z == x:
                    continue
                s = y + z
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s2 = 2 * y
                if counts.get(s2, 0) >= 2:
                    ok = False
            
            if ok:
                # Perform swap
                chosen.remove(x)
                chosen_set.remove(x)
                chosen.append(y)
                chosen_set.add(y)
                
                # Update counts for y
                for z in chosen:
                    if z == y:
                        continue
                    s = y + z
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * y] = counts.get(2 * y, 0) + 1
                
                no_improve_count = 0
            else:
                # Revert counts for x
                for z in chosen:
                    if z == x:
                        continue
                    s = x + z
                    counts[s] += 1
                counts[2 * x] += 1
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        else:
            break
            
    return best_chosen, best_counts, best_size

def construct():
    deadline = time.time() + 110  # 40s safety margin
    
    # Phase 1: Initial Greedy
    chosen, counts = _fast_greedy()
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_counts = dict(counts)
    
    print(f"Initial Greedy size: {best_size}", file=sys.stderr)
    
    # Phase 2: Local Search with multiple seeds
    for seed in range(50):
        if time.time() >= deadline:
            break
            
        c, cnt, sz = _local_search_swap(list(best_chosen), dict(best_counts), deadline, seed=seed)
        if sz > best_size:
            best_size = sz
            best_chosen = c
            best_counts = cnt
            print(f"Improved to {best_size} at seed {seed}", file=sys.stderr)
            
    # Phase 3: Final Polish (Greedy Add)
    if time.time() < deadline:
        chosen_set = set(best_chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                ok = True
                for y in best_chosen:
                    s = x + y
                    if best_counts.get(s, 0) >= 2:
                        ok = False
                        break
                if ok:
                    s2 = 2 * x
                    if best_counts.get(s2, 0) >= 2:
                        ok = False
                
                if ok:
                    best_chosen.append(x)
                    chosen_set.add(x)
                    for y in best_chosen[:-1]:
                        s = x + y
                        best_counts[s] = best_counts.get(s, 0) + 1
                    best_counts[2 * x] = best_counts.get(2 * x, 0) + 1
                    best_size += 1
                    improved = True
                    break

    print(f"Final size: {best_size}", file=sys.stderr)
    return best_chosen
