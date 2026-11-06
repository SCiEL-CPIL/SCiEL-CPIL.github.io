import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    N = 128
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)

    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True

    def greedy_construct(k, seed):
        """
        Greedy left-to-right construction with palette size k.
        Returns None if stuck.
        """
        rng = random.Random(seed)
        colors = [-1] * N
        
        for pos in range(N):
            # Determine forbidden colors based on already assigned neighbors in taps
            forbidden_mask = 0
            stuck = False
            
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                pts = [a, b, c, d]
                others = [p for p in pts if p != pos]
                
                # Check if all others are assigned
                assigned_colors = []
                all_assigned = True
                for p in others:
                    if colors[p] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[p])
                
                if not all_assigned:
                    continue
                
                s = set(assigned_colors)
                if len(s) == 1:
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
                # len(s) == 3: no restriction
            
            if stuck:
                return None
            
            # Find a valid color in 0..k-1
            valid_colors = []
            for c in range(k):
                if not (forbidden_mask & (1 << c)):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Pick randomly to diversify
            colors[pos] = rng.choice(valid_colors)
        
        # Final validation
        if not is_valid_coloring(colors):
            return None
        
        return colors

    best_count = N
    best_colors = None
    
    # Try increasing number of colors
    # We can stop if we find a solution with 3 colors, as that is the theoretical minimum
    # (since we need at least 3 distinct colors in any 4-AP, we need at least 3 colors total?
    # No, we need at least 3 colors in EVERY 4-AP.
    # Can we use only 2 colors total?
    # If we use only 2 colors, any 4-AP can have at most 2 distinct colors.
    # So it's impossible.
    # So minimum is at least 3.
    
    for k in range(3, 15):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple seeds for each k
        # Increase seeds for smaller k as they are harder
        num_seeds = 500 if k <= 5 else 200
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
                
            colors = greedy_construct(k, seed)
            if colors is not None:
                # It is valid and uses at most k colors
                distinct = len(set(colors))
                if distinct < best_count:
                    best_count = distinct
                    best_colors = colors
                    found = True
                    # If we found a solution with k colors, we can stop trying this k
                    # because we want the minimum. But wait, distinct might be < k.
                    # If distinct < k, we have a solution with fewer colors.
                    # We should break out of the seed loop and the k loop if distinct == k?
                    # No, if distinct < k, we found a better solution.
                    # If distinct == k, we found a solution with k colors.
                    # We want the minimum. So if we find a valid coloring with k colors,
                    # we know the answer is <= k.
                    # We can stop searching for this k if we found one, but we should
                    # check if we can do better (i.e., is there a solution with k-1 colors?).
                    # Since we are iterating k upwards, if we found a solution with k,
                    # we don't need to try more seeds for this k, because any other solution
                    # with k colors is not better than the one we have (in terms of count).
                    # However, we might have found a solution with k colors that actually
                    # uses only k-1 distinct colors? No, if it uses k-1 distinct colors,
                    # it would have been found in the previous iteration (k-1) or we can
                    # just record the distinct count.
                    # Actually, if we run greedy with palette k, it might use only k-1 colors.
                    # So we should record the actual distinct count.
                    # But if we found a valid coloring with palette k, the minimum number
                    # of colors is <= k.
                    # We can break the seed loop.
                    break
        
        if found:
            # We found a valid coloring with palette size k.
            # The actual number of distinct colors is best_count.
            # If best_count == k, we might still find a solution with k+1 palette that uses k-1 colors?
            # Unlikely.
            # But we should continue to see if we can find a solution with fewer colors?
            # We are going upwards in k.
            # So if we found a solution with k=5 (using 4 colors), we know min <= 4.
            # We should have tried k=4. If we failed, we don't know.
            # But we can't go back to k=4.
            # So we just keep the best found.
            pass

    if best_colors is None:
        # Fallback: return a valid coloring with many colors (e.g., 128)
        # Just use identity coloring, which is valid?
        # Identity: c[i] = i.
        # 4-AP: i, i+d, i+2d, i+3d. Colors: i, i+d, i+2d, i+3d.
        # Distinct: 4. Valid.
        best_colors = list(range(N))
        best_count = N

    return best_colors
