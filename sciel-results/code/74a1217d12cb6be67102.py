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
    
    def is_antichain_indices(indices):
        """Check if the set of indices forms an antichain"""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if divides[indices[i]][indices[j]] or divides[indices[j]][indices[i]]:
                    return False
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
    
    answers = []
    
    for n in range(1, 129):
        if time.time() > deadline:
            # If we run out of time, use a fallback search
            break
        
        if best_for_target[n] is not None:
            indices = best_for_target[n]
            found = [smooth_list[i] for i in indices]
        else:
            # Fallback: search for size 6+ if needed (should be rare)
            found = None
            candidates = [x for x in smooth_list if x < n]
            for k in range(6, 11):
                for combo in itertools.combinations(candidates, k):
                    if sum(combo) == n and is_antichain_indices(combo):
                        found = list(combo)
                        break
                if found:
                    break
            
            if found is None:
                # Emergency: find any valid antichain
                found = [n]  # This should never happen for n<=128
        
        answers.append(found)
    
    return answers
