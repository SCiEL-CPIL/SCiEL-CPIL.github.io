"""B_3[1] construction: fast greedy with early rejection and swap-based improvement."""
import time
import random
import bisect

def construct():
    deadline = time.time() + 85.0
    N = 4096
    
    def can_add_fast(x, selected, pair_sums):
        """Check if x can be added to selected without violating B_3[1] property.
        Uses precomputed pair_sums set for O(k) checks instead of O(k^2)."""
        k = len(selected)
        # 3x
        if 3 * x in pair_sums or 3 * x in set(selected):
            # 3x might not be in pair_sums if pair_sums only stores s+t for s<=t
            pass
        # Check 3x against all existing triple sums - we need a separate set for triple sums
        # Actually, let's restructure: maintain triple_sums set
        return True  # Placeholder, will rewrite below

    # Better approach: maintain a set of all valid triple sums
    # triple_sum(a,b,c) = a+b+c where a<=b<=c (unordered, with repetition)
    
    def can_add(x, selected, triple_sums):
        """Check if x can be added to selected without violating B_3[1] property."""
        # 3x
        if 3 * x in triple_sums:
            return False
        # 2x + s for s in selected
        two_x = 2 * x
        for s in selected:
            if two_x + s in triple_sums:
                return False
        # x + 2s for s in selected
        for s in selected:
            if x + 2 * s in triple_sums:
                return False
        # x + s + t for s <= t in selected
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                if x + s + t in triple_sums:
                    return False
        return True

    def add_x(x, selected, triple_sums):
        """Add x to selected and update triple_sums."""
        n = len(selected)
        # 3x
        triple_sums.add(3 * x)
        # 2x + s
        for s in selected:
            triple_sums.add(2 * x + s)
            triple_sums.add(x + 2 * s)
        # x + s + t
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                triple_sums.add(x + s + t)
        selected.append(x)

    def remove_x(x, selected, triple_sums):
        """Remove x from selected and update triple_sums."""
        idx = selected.index(x)
        selected.pop(idx)
        # Remove all sums involving x
        to_remove = set()
        to_remove.add(3 * x)
        for s in selected:
            to_remove.add(2 * x + s)
            to_remove.add(x + 2 * s)
        n = len(selected)
        for i in range(n):
            s = selected[i]
            for j in range(i, n):
                t = selected[j]
                to_remove.add(x + s + t)
        triple_sums.difference_update(to_remove)

    # Phase 1: Standard greedy from 1 to 4096
    selected = []
    triple_sums = set()
    
    for x in range(1, N + 1):
        if time.time() > deadline - 12.0:
            break
        if can_add(x, selected, triple_sums):
            add_x(x, selected, triple_sums)
    
    best_selected = selected[:]
    best_triple_sums = triple_sums.copy()
    best_size = len(selected)
    
    # Phase 2: Try to improve via swaps
    if time.time() < deadline - 5.0:
        sel_set = set(selected)
        candidates = sorted([x for x in range(1, N + 1) if x not in sel_set], reverse=True)
        
        # Track which candidates we've tried
        tried = set()
        
        improved = True
        while improved and time.time() < deadline - 4.0:
            improved = False
            n_sel = len(selected)
            for i in range(n_sel):
                if time.time() > deadline - 4.0:
                    break
                orig = selected[i]
                for cand in candidates:
                    if time.time() > deadline - 4.0:
                        break
                    if cand in tried:
                        continue
                    if cand <= orig:
                        break  # Only try larger candidates (sorted descending)
                    # Try swap
                    remove_x(orig, selected, triple_sums)
                    if can_add(cand, selected, triple_sums):
                        add_x(cand, selected, triple_sums)
                        tried.add(cand)
                        # Remove from candidates list
                        candidates.remove(cand)
                        improved = True
                        break
                    else:
                        add_x(orig, selected, triple_sums)
                        tried.add(cand)
                selected.sort()
        
        if len(selected) > best_size:
            best_selected = selected[:]
            best_triple_sums = triple_sums.copy()
            best_size = len(selected)
    
    # Phase 3: Try adding more elements with multiple strategies
    if time.time() < deadline - 2.0:
        for attempt in range(10):
            if time.time() > deadline - 2.0:
                break
            
            sel_set = set(best_selected)
            remaining = [x for x in range(1, N + 1) if x not in sel_set]
            
            # Different ordering strategies
            if attempt == 0:
                remaining.sort(reverse=True)  # Descending
            elif attempt == 1:
                remaining.sort()  # Ascending
            elif attempt % 2 == 0:
                remaining.sort(key=lambda x: random.random())
            else:
                # Random with bias toward larger
                remaining.sort(key=lambda x: random.random() + x / N)
            
            # Try to add one element
            added = False
            for x in remaining:
                if time.time() > deadline - 2.0:
                    break
                temp_selected = best_selected[:]
                temp_sums = best_triple_sums.copy()
                if can_add(x, temp_selected, temp_sums):
                    add_x(x, temp_selected, temp_sums)
                    if len(temp_selected) > best_size:
                        best_selected = temp_selected
                        best_triple_sums = temp_sums
                        best_size = len(temp_selected)
                        added = True
                        break
            
            if not added:
                # No more single additions possible; break early if we've tried several
                if attempt >= 3:
                    break
    
    return sorted(best_selected)[:80]
