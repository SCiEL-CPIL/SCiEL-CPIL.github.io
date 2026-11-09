import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128

    # Precompute all 4-APs
    aps = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps_arr)

    def check_validity(colors):
        colors = np.array(colors, dtype=np.int32)
        # For each AP, check if it has >= 3 distinct colors
        # Get all 4 colors for each AP
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        
        # Count distinct: an AP is bad if it has <= 2 distinct colors
        # Bad if: all same, or exactly 2 distinct
        # all same: c0==c1==c2==c3
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        # exactly 2 distinct: not all same, but only 2 values
        # This means at least one pair is equal and not all are equal
        # Simpler: check if the set size is < 3
        # We can check: if (c0==c1 or c0==c2 or c0==c3) and (c1==c2 or c1==c3 or c2==c3)
        # Actually, let's just compute min and max and check
        # Or: bad if c0,c1,c2,c3 take at most 2 values
        # Equivalent: not (at least 3 distinct)
        # 3 distinct means: not (all same) and not (exactly 2)
        # Exactly 2: there exist i,j with ci!=cj, and for all k, ck in {ci,cj}
        
        # Simple approach: for each AP, check if number of distinct < 3
        # Use: (c0 != c1) + (c0 != c2) + (c0 != c3) + (c1 != c2) + (c1 != c3) + (c2 != c3)
        # If this sum is 0: all same (1 distinct) - bad
        # If this sum is 3: exactly 2 distinct (one color appears 3 times, other 1) - bad
        # If this sum is 6: all distinct (4 distinct) - good
        # If this sum is 2: 3 distinct (one color appears 2 times, others 1 each) - good
        # Wait, let me recount.
        
        # Number of distinct values:
        # 1 distinct: 0 unequal pairs
        # 2 distinct: 3 unequal pairs (one color appears 3 times: C(3,1)*C(1,1)=3, or 2+2: C(2,2)=1... wait)
        # Actually for 4 elements with 2 distinct colors:
        #   3+1 split: 3 of one, 1 of other. Unequal pairs: 3*1 = 3
        #   2+2 split: 2 of each. Unequal pairs: 2*2 = 4
        # 3 distinct: 2+1+1 split. Unequal pairs: 2*1 + 2*1 + 1*1 = 5
        # 4 distinct: 1+1+1+1. Unequal pairs: 6
        
        # So: sum of (ci != cj) over all pairs:
        # 1 distinct: 0
        # 2 distinct (3+1): 3
        # 2 distinct (2+2): 4
        # 3 distinct (2+1+1): 5
        # 4 distinct: 6
        
        # Bad if sum <= 4 (1 or 2 distinct)
        # Good if sum >= 5 (3 or 4 distinct)
        
        unequal = ((c0 != c1).astype(np.int32) + (c0 != c2).astype(np.int32) + 
                   (c0 != c3).astype(np.int32) + (c1 != c2).astype(np.int32) + 
                   (c1 != c3).astype(np.int32) + (c2 != c3).astype(np.int32))
        
        return np.all(unequal >= 5)

    def count_colors(colors):
        return len(set(colors))

    # Greedy construction with randomization
    def try_greedy(max_c, seed_val=0):
        rng = random.Random(seed_val)
        colors = [-1] * n

        for x in range(n):
            if time.time() > deadline - 5:
                return None

            valid_colors = []
            for c in range(max_c):
                ok = True
                for (a, b, c_idx, d_idx) in aps:
                    if x == a:
                        if colors[b] != -1 and colors[c_idx] != -1 and colors[d_idx] != -1:
                            if len({c, colors[b], colors[c_idx], colors[d_idx]}) < 3:
                                ok = False
                                break
                    elif x == b:
                        if colors[a] != -1 and colors[c_idx] != -1 and colors[d_idx] != -1:
                            if len({colors[a], c, colors[c_idx], colors[d_idx]}) < 3:
                                ok = False
                                break
                    elif x == c_idx:
                        if colors[a] != -1 and colors[b] != -1 and colors[d_idx] != -1:
                            if len({colors[a], colors[b], c, colors[d_idx]}) < 3:
                                ok = False
                                break
                    elif x == d_idx:
                        if colors[a] != -1 and colors[b] != -1 and colors[c_idx] != -1:
                            if len({colors[a], colors[b], colors[c_idx], c}) < 3:
                                ok = False
                                break
                if ok:
                    valid_colors.append(c)

            if not valid_colors:
                return None

            # Prefer used colors
            used = set(colors[:x])
            used_valid = [c for c in valid_colors if c in used]
            if used_valid:
                chosen = rng.choice(used_valid)
            else:
                chosen = rng.choice(valid_colors)

            colors[x] = chosen

        return colors

    # Local search to minimize colors
    def local_search(colors, max_c, seed_val=0, max_iters=10000):
        rng = random.Random(seed_val)
        current = list(colors)
        current_count = len(set(current))
        best = list(current)
        best_count = current_count

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            x = rng.randint(0, n - 1)
            old_color = current[x]
            new_color = rng.randint(0, max_c - 1)
            if new_color == old_color:
                continue

            current[x] = new_color
            if check_validity(current):
                new_count = len(set(current))
                if new_count < best_count:
                    best_count = new_count
                    best = list(current)
                if new_count <= current_count:
                    current_count = new_count
                else:
                    current[x] = old_color
            else:
                current[x] = old_color

        return best, best_count

    # Try 3 colors
    best_result = None
    best_count = n

    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        result = try_greedy(3, seed_val=attempt)
        if result is not None:
            cnt = count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_result = result
            if cnt == 3:
                break

    if best_result is not None and best_count == 3:
        return best_result

    # Try 4 colors
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        result = try_greedy(4, seed_val=1000 + attempt)
        if result is not None:
            cnt = count_colors(result)
            if cnt < best_count:
                best_count = cnt
                best_result = result
            if cnt == 3:
                break

    if best_result is not None:
        # Try local search to reduce
        if best_count > 3:
            improved, improved_count = local_search(best_result, best_count, seed_val=0, max_iters=5000)
            if improved_count < best_count:
                best_count = improved_count
                best_result = improved

        return best_result

    # Fallback
    colors = list(range(n))
    return colors
