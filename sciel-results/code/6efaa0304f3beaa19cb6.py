"""Difference triangle set (4,7) - optimized search."""
import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(42)

    def diff_set(row):
        """Return set of all positive differences in a row."""
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i+1, n):
                s.add(row[j] - ri)
        return s

    def all_diffs(rows):
        s = set()
        for r in rows:
            s |= diff_set(r)
        return s

    def is_valid(rows):
        return len(all_diffs(rows)) == n_rows * k * (k + 1) // 2

    def max_entry(rows):
        return max(max(r) for r in rows)

    def row_diffs(row):
        """Return dict: diff -> (i,j) for each positive difference."""
        d = {}
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d[row[j] - row[i]] = (i, j)
        return d

    def compute_used(rows):
        return all_diffs(rows)

    # --- Construction: greedy with balancing ---
    def greedy_construct(balanced=False):
        """Build rows greedily. If balanced, try to spread differences."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if x is valid
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    row.append(x)
                    used |= new_diffs
                elif balanced:
                    # Try a few larger values to find one that works
                    for x2 in range(x, min(x + 50, 200)):
                        new_diffs = set()
                        ok2 = True
                        for a in row:
                            d = x2 - a
                            if d in used:
                                ok2 = False
                                break
                            new_diffs.add(d)
                        if ok2:
                            x = x2
                            row.append(x)
                            used |= new_diffs
                            break
                    else:
                        # Couldn't find one in window, try further
                        x = x + 1
                        while True:
                            new_diffs = set()
                            ok2 = True
                            for a in row:
                                d = x - a
                                if d in used:
                                    ok2 = False
                                    break
                                new_diffs.add(d)
                            if ok2:
                                row.append(x)
                                used |= new_diffs
                                break
                            x += 1
            rows.append(row)
        return rows

    # --- Local search: try moving a single mark in a row ---
    def local_search(rows, max_iters=50000):
        """Iteratively try to reduce max entry by moving individual marks."""
        best = [list(r) for r in rows]
        best_score = max_entry(best)
        used = compute_used(best)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try to reduce the largest mark in each row
            for ri in range(n_rows):
                row = best[ri]
                # Find the largest mark (last element)
                max_idx = len(row) - 1
                old_val = row[max_idx]
                
                # Try to reduce it
                for new_val in range(old_val - 1, row[max_idx - 1], -1):
                    # Check if new_val creates valid differences
                    new_diffs = set()
                    conflict = False
                    for i in range(max_idx):
                        d = new_val - row[i]
                        if d <= 0:
                            conflict = True
                            break
                        if d in used:
                            # Check if this diff is from the same row
                            # Remove old diffs involving max_idx
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    
                    # More careful check: remove old diffs from this row involving max_idx
                    old_diffs_removed = set()
                    for i in range(max_idx):
                        old_diffs_removed.add(row[max_idx] - row[i])
                    
                    # Check new diffs don't conflict with (used - old_diffs_removed)
                    remaining_used = used - old_diffs_removed
                    if new_diffs & remaining_used:
                        continue
                    
                    # Also check new_val > row[max_idx-1] (strictly increasing)
                    if new_val <= row[max_idx - 1]:
                        continue
                    
                    # Accept the move
                    used -= old_diffs_removed
                    used |= new_diffs
                    row[max_idx] = new_val
                    best_score = min(best_score, max_entry(best))
                    improved = True
                    break
            
            # Also try reducing other (non-last) marks
            for ri in range(n_rows):
                row = best[ri]
                for idx in range(len(row) - 2, 0, -1):  # from second-to-last down to index 1
                    old_val = row[idx]
                    # Try to reduce
                    lower_bound = row[idx - 1] + 1
                    for new_val in range(old_val - 1, lower_bound - 1, -1):
                        if new_val <= row[idx - 1]:
                            break
                        # Compute old diffs involving idx
                        old_diffs_removed = set()
                        for i in range(len(row)):
                            if i != idx:
                                if i < idx:
                                    old_diffs_removed.add(row[idx] - row[i])
                                else:
                                    old_diffs_removed.add(row[i] - row[idx])
                        
                        # Compute new diffs involving idx
                        new_diffs = set()
                        conflict = False
                        for i in range(len(row)):
                            if i == idx:
                                continue
                            if i < idx:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            if d <= 0:
                                conflict = True
                                break
                            new_diffs.add(d)
                        if conflict:
                            continue
                        
                        remaining_used = used - old_diffs_removed
                        if new_diffs & remaining_used:
                            continue
                        
                        # Check strictly increasing
                        if idx > 0 and new_val <= row[idx - 1]:
                            continue
                        if idx < len(row) - 1 and new_val >= row[idx + 1]:
                            continue
                        
                        # Accept
                        used -= old_diffs_removed
                        used |= new_diffs
                        row[idx] = new_val
                        best_score = min(best_score, max_entry(best))
                        improved = True
                        break
            
            if not improved:
                break
        
        return best

    # --- Polish pass: exhaustive reduction of every element ---
    def polish(rows):
        """Exhaustively try to reduce every non-first element in each row."""
        best = [list(r) for r in rows]
        used = compute_used(best)
        
        changed = True
        while changed:
            if time.time() > deadline - 3.0:
                break
            changed = False
            for ri in range(n_rows):
                row = best[ri]
                for idx in range(1, len(row)):
                    old_val = row[idx]
                    # Try to reduce to the minimum possible
                    lower = row[idx - 1] + 1
                    for new_val in range(old_val - 1, lower - 1, -1):
                        if new_val <= row[idx - 1]:
                            break
                        # Compute old diffs involving idx
                        old_diffs_removed = set()
                        for i in range(len(row)):
                            if i != idx:
                                if i < idx:
                                    old_diffs_removed.add(row[idx] - row[i])
                                else:
                                    old_diffs_removed.add(row[i] - row[idx])
                        
                        # Compute new diffs involving idx
                        new_diffs = set()
                        conflict = False
                        for i in range(len(row)):
                            if i == idx:
                                continue
                            if i < idx:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            if d <= 0:
                                conflict = True
                                break
                            new_diffs.add(d)
                        if conflict:
                            continue
                        
                        remaining_used = used - old_diffs_removed
                        if new_diffs & remaining_used:
                            continue
                        
                        # Check strictly increasing
                        if new_val <= row[idx - 1]:
                            continue
                        if idx < len(row) - 1 and new_val >= row[idx + 1]:
                            continue
                        
                        # Accept
                        used -= old_diffs_removed
                        used |= new_diffs
                        row[idx] = new_val
                        changed = True
                        break
        
        return best

    # --- Main search: ILS with perturbations ---
    best_rows = greedy_construct(balanced=False)
    best_score = max_entry(best_rows)
    
    # Polish the initial solution
    best_rows = polish(best_rows)
    best_score = max_entry(best_rows)
    
    # Local search
    best_rows = local_search(best_rows)
    best_rows = polish(best_rows)
    best_score = max_entry(best_rows)
    
    # ILS loop
    current_rows = [list(r) for r in best_rows]
    
    for it in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb: randomly move a few marks
        perturbed = [list(r) for r in current_rows]
        n_perturb = rng.randint(1, 3)
        for _ in range(n_perturb):
            ri = rng.randint(0, n_rows - 1)
            idx = rng.randint(1, k)  # not index 0
            old_val = perturbed[ri][idx]
            # Try to move to a random nearby value
            delta = rng.randint(-10, 10)
            new_val = old_val + delta
            if new_val <= perturbed[ri][idx - 1] or (idx < k and new_val >= perturbed[ri][idx + 1]):
                continue
            # Check validity
            old_diffs_removed = set()
            for i in range(k + 1):
                if i != idx:
                    if i < idx:
                        old_diffs_removed.add(old_val - perturbed[ri][i])
                    else:
                        old_diffs_removed.add(perturbed[ri][i] - old_val)
            
            new_diffs = set()
            conflict = False
            used = compute_used(perturbed) - old_diffs_removed
            for i in range(k + 1):
                if i == idx:
                    continue
                if i < idx:
                    d = new_val - perturbed[ri][i]
                else:
                    d = perturbed[ri][i] - new_val
                if d <= 0 or d in used:
                    conflict = True
                    break
                new_diffs.add(d)
            if not conflict:
                perturbed[ri][idx] = new_val
        
        # Local search on perturbed
        searched = local_search(perturbed, max_iters=10000)
        searched = polish(searched)
        score = max_entry(searched)
        
        if score < best_score:
            best_rows = searched
            best_score = score
            current_rows = [list(r) for r in searched]
    
    # Final polish
    best_rows = polish(best_rows)
    best_score = max_entry(best_rows)
    
    return best_rows
