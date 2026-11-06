import random

def construct():
    N = 128
    # Precompute 4-term arithmetic progressions ending at each index i
    # i is 0-indexed, representing position i+1
    # A 4-AP ending at i is (i-3d, i-2d, i-d, i) for d >= 1 such that i-3d >= 0
    aps_end = [[] for _ in range(N)]
    for i in range(N):
        # d can go up to i//3
        max_d = i // 3
        for d in range(1, max_d + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            aps_end[i].append((a, b, c))
    
    def is_valid(idx, c, colors):
        """Check if assigning color c to idx satisfies all 4-AP constraints ending at idx."""
        for (a, b, cc) in aps_end[idx]:
            # The colors involved are colors[a], colors[b], colors[cc], and c
            # We need at least 3 distinct colors
            # Optimization: check distinct count
            c1 = colors[a]
            c2 = colors[b]
            c3 = colors[cc]
            # If c == c1 == c2 == c3, count=1
            # If c is in {c1,c2,c3} and c1,c2,c3 have <3 distinct, etc.
            # Simple set creation is fast enough for small numbers
            if len({c1, c2, c3, c}) < 3:
                return False
        return True
    
    def try_random_greedy(K, num_restarts=5000):
        """Try to find a valid coloring with K colors using randomized greedy."""
        for r in range(num_restarts):
            random.seed(r) # Deterministic per restart index for reproducibility if needed, or just use random
            colors = [-1] * N
            success = True
            for idx in range(N):
                # Find all valid colors for this position
                valid_colors = []
                for c in range(K):
                    if is_valid(idx, c, colors):
                        valid_colors.append(c)
                
                if not valid_colors:
                    success = False
                    break
                
                # Pick a random valid color
                # Heuristic: Pick the one used least frequently so far? 
                # Or just random. Random is robust.
                colors[idx] = random.choice(valid_colors)
            
            if success:
                return colors
        return None
    
    # Try to minimize K
    # Start from 3, as 2 is impossible (any 4-AP would have <=2 colors)
    for K in range(3, 13):
        res = try_random_greedy(K)
        if res is not None:
            return res
    
    # Fallback: Deterministic greedy (similar to original but optimized)
    # This ensures we always return a valid result
    colors = []
    for x in range(1, 129):
        found = False
        for c in range(128):
            valid = True
            # Check all 4-APs ending at x-1 (0-indexed)
            idx = x - 1
            for (a, b, cc) in aps_end[idx]:
                if len({colors[a], colors[b], colors[cc], c}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            # Should not happen with 128 colors
            colors.append(0)
            
    return colors
