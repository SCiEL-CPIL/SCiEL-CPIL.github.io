import math
import time
import numpy as np

def construct():
    DEADLINE = 130.0
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_score(u):
        """Compute score using NumPy for speed."""
        u_arr = np.array(sorted(set(u)), dtype=np.int64)
        n = len(u_arr)
        if n < 2:
            return 1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        # Sumset: all a+b for a,b in u
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        # Diffset: all a-b for a,b in u
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        """Check if u is a Golomb ruler using a set of differences."""
        u = sorted(u)
        n = len(u)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def is_golomb_fast(u, max_diff):
        """Faster Golomb check using a boolean array."""
        u = sorted(u)
        n = len(u)
        seen = bytearray(max_diff + 1)
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d > max_diff:
                    return False
                if seen[d]:
                    return False
                seen[d] = 1
        return True
    
    # Known optimal Golomb rulers (length = max mark, marks sorted)
    # Source: standard Golomb ruler tables
    known_rulers = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76, 84],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76, 84, 92],
    }
    
    # Verify and compute scores for known rulers
    candidates = []
    for n, ruler in known_rulers.items():
        if not is_golomb(ruler):
            continue
        candidates.append(ruler)
    
    # Also try some non-Golomb sets that might have better ratio
    # For a set where |U-U|/|U+U| is large but max(U) is small
    # Actually for Golomb: ratio = (4n-2)/(n+1), which increases with n
    # But max(U) also grows, so we need compact rulers
    
    # Local search: try to find a more compact Golomb ruler
    def local_search_golomb(initial_u, time_budget=10.0):
        """Try to find a more compact Golomb ruler via local search."""
        start_ls = time.time()
        best_u = list(initial_u)
        best_score = compute_score(best_u)
        n = len(best_u)
        
        # Try: for each position, try to reduce the mark
        # Or: try adding a new mark and then optimizing
        # Strategy: hill-climb on max(U) while maintaining Golomb property
        
        improved = True
        while improved and (time.time() - start_ls) < time_budget and remaining() > 10:
            improved = False
            current = list(best_u)
            current_max = max(current)
            
            # Try removing the last mark and re-adding a smaller one
            if n > 3:
                base = current[:-1]
                base_max = max(base)
                # Try marks from base_max+1 upward, find smallest that gives Golomb
                for new_mark in range(base_max + 1, base_max + n * n + 1):
                    candidate = sorted(base + [new_mark])
                    if is_golomb_fast(candidate, 2 * new_mark):
                        s = compute_score(candidate)
                        if s > best_score:
                            best_u = candidate
                            best_score = s
                            improved = True
                        break  # First valid mark is the smallest, which minimizes max(U)
            
            # Try: swap a mark with a smaller one
            for i in range(1, len(best_u)):
                if (time.time() - start_ls) > time_budget or remaining() < 10:
                    break
                old_mark = best_u[i]
                # Try reducing mark i
                for new_mark in range(best_u[i-1] + 1, old_mark):
                    candidate = list(best_u)
                    candidate[i] = new_mark
                    candidate.sort()
                    if is_golomb_fast(candidate, 2 * max(candidate)):
                        s = compute_score(candidate)
                        if s > best_score:
                            best_u = candidate
                            best_score = s
                            improved = True
                            break
        
        return best_u, best_score
    
    # Run local search on the largest known rulers
    for n in [16, 15, 14, 13, 12, 11, 10]:
        if remaining() < 15:
            break
        ruler = known_rulers.get(n)
        if ruler is None:
            continue
        # Budget proportional to n
        budget = min(8.0, remaining() * 0.1)
        if budget < 1.0:
            break
        improved_u, _ = local_search_golomb(ruler, time_budget=budget)
        if improved_u != ruler:
            candidates.append(improved_u)
    
    # Also try greedy construction from scratch for various n
    def greedy_golomb(n, time_limit=5.0):
        """Greedy construction of a Golomb ruler with n marks."""
        start_g = time.time()
        u = [0]
        for i in range(1, n):
            if time.time() - start_g > time_limit or remaining() < 10:
                break
            pos = max(u) + 1
            while True:
                candidate = sorted(u + [pos])
                if is_golomb_fast(candidate, 2 * pos):
                    u = candidate
                    break
                pos += 1
                if pos > n * n * 50:
                    return u
        return u
    
    # Try greedy for n = 10..16
    for n in [16, 15, 14, 13, 12, 11, 10]:
        if remaining() < 10:
            break
        g_u = greedy_golomb(n, time_limit=min(5.0, remaining() * 0.08))
        if g_u and len(g_u) >= 5:
            candidates.append(g_u)
    
    # Find the best candidate
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
    
    # Final: try a quick refinement on the best set
    if remaining() > 5 and len(best_set) >= 4:
        refined, _ = local_search_golomb(best_set, time_budget=min(5.0, remaining() * 0.5))
        if compute_score(refined) > best_score:
            best_set = refined
    
    return best_set
