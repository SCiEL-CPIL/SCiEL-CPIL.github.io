import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # 1. Generate 2,3,5-smooth numbers up to 128
    smooth = set()
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.add(x)
    
    smooth_list = sorted(smooth)
    smooth_set = smooth
    n_smooth = len(smooth_list)
    
    # Precompute antichain check using bitmask for efficiency
    # For each pair (i,j), check if smooth_list[i] divides smooth_list[j]
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth_list[j] % smooth_list[i] == 0:
                divides[i][j] = True
    
    # Precompute conflict masks: for each i, a bitmask of j where i and j conflict
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and (divides[i][j] or divides[j][i]):
                conflict_mask[i] |= (1 << j)
    
    def is_antichain_indices(indices):
        """Check if the set of indices forms an antichain using bitmask"""
        mask = 0
        for idx in indices:
            if mask & conflict_mask[idx]:
                return False
            mask |= (1 << idx)
        return True
    
    # Precompute valid antichains by size for quick lookup
    # Store as: antichains_by_size[size] = list of (sum, tuple_of_indices)
    antichains_by_size = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    # Size 1: trivial
    for i in range(n_smooth):
        antichains_by_size[1].append((smooth_list[i], (i,)))
    
    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not (divides[i][j] or divides[j][i]):
                antichains_by_size[2].append((smooth_list[i] + smooth_list[j], (i, j)))
    
    # Size 3
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if (divides[i][k] or divides[k][i]) or (divides[j][k] or divides[k][j]):
                    continue
                antichains_by_size[3].append((smooth_list[i] + smooth_list[j] + smooth_list[k], (i, j, k)))
    
    # Size 4
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if (divides[i][k] or divides[k][i]) or (divides[j][k] or divides[k][j]):
                    continue
                for l in range(k + 1, n_smooth):
                    if (divides[i][l] or divides[l][i]) or (divides[j][l] or divides[l][j]) or (divides[k][l] or divides[l][k]):
                        continue
                    antichains_by_size[4].append((smooth_list[i] + smooth_list[j] + smooth_list[k] + smooth_list[l], (i, j, k, l)))
    
    # Size 5
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divides[i][j] or divides[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                if (divides[i][k] or divides[k][i]) or (divides[j][k] or divides[k][j]):
                    continue
                for l in range(k + 1, n_smooth):
                    if (divides[i][l] or divides[l][i]) or (divides[j][l] or divides[l][j]) or (divides[k][l] or divides[l][k]):
                        continue
                    for m in range(l + 1, n_smooth):
                        if (divides[i][m] or divides[m][i]) or (divides[j][m] or divides[m][j]) or (divides[k][m] or divides[m][k]) or (divides[l][m] or divides[m][l]):
                            continue
                        antichains_by_size[5].append((smooth_list[i] + smooth_list[j] + smooth_list[k] + smooth_list[l] + smooth_list[m], (i, j, k, l, m)))
    
    # Build lookup: for each target n, find the minimum size antichain that sums to n
    best_for_target = [None] * 129
    
    for size in range(1, 6):
        for (total, indices) in antichains_by_size[size]:
            if total <= 128:
                if best_for_target[total] is None or size < len(best_for_target[total]):
                    best_for_target[total] = indices
    
    # For targets not covered, do a targeted search with pruning
    def search_antichain_for_target(n, max_size=8):
        """Search for an antichain summing to n with at most max_size terms"""
        candidates = [i for i in range(n_smooth) if smooth_list[i] < n]
        
        def backtrack(remaining, start, current, current_sum, depth):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return list(current)
            if depth == max_size:
                return None
            if not candidates:
                return None
            
            # Pruning: if the sum of the smallest remaining candidates can't reach remaining
            min_possible = sum(smooth_list[candidates[start:start + remaining]] if start + remaining <= len(candidates) else smooth_list[candidates[start:]])
            if min_possible > remaining:
                return None
            
            for idx in range(start, len(candidates)):
                i = candidates[idx]
                if smooth_list[i] > remaining:
                    break
                new_remaining = remaining - smooth_list[i]
                if new_remaining < 0:
                    continue
                # Check antichain constraint
                valid = True
                for j in current:
                    if divides[i][j] or divides[j][i]:
                        valid = False
                        break
                if not valid:
                    continue
                current.append(i)
                result = backtrack(new_remaining, idx + 1, current, current_sum + smooth_list[i], depth + 1)
                current.pop()
                if result is not None:
                    return result
            return None
        
        return backtrack(n, 0, [], 0, 0)
    
    answers = []
    
    for n in range(1, 129):
        if time.time() > deadline:
            # If we run out of time, fill remaining with fallback
            for m in range(n, 129):
                if best_for_target[m] is not None:
                    answers.append([smooth_list[i] for i in best_for_target[m]])
                else:
                    # Emergency fallback: use a single term if it's smooth
                    if m in smooth_set:
                        answers.append([m])
                    else:
                        # Last resort: try to find any valid representation
                        found = search_antichain_for_target(m, max_size=6)
                        if found:
                            answers.append([smooth_list[i] for i in found])
                        else:
                            # Very last resort: use two terms that sum to m
                            # Find any two smooth numbers that sum to m and don't divide each other
                            for a in smooth_list:
                                if a >= m:
                                    break
                                b = m - a
                                if b in smooth_set and not (a % b == 0 or b % a == 0):
                                    answers.append([a, b])
                                    break
                                else:
                                    answers.append([m])  # Should not happen
            break
        
        if best_for_target[n] is not None:
            indices = best_for_target[n]
            found = [smooth_list[i] for i in indices]
        else:
            # Fallback: search for size 6+ if needed
            found_indices = search_antichain_for_target(n, max_size=8)
            if found_indices is not None:
                found = [smooth_list[i] for i in found_indices]
            else:
                # Emergency: find any valid antichain
                found = [n] if n in smooth_set else None
                if found is None:
                    # Try two terms
                    for a in smooth_list:
                        if a >= n:
                            break
                        b = n - a
                        if b in smooth_set and not (a % b == 0 or b % a == 0):
                            found = [a, b]
                            break
                if found is None:
                    found = [n]  # Last resort
        
        answers.append(found)
    
    return answers
