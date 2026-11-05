"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    def get_forbidden(idx, colors, K):
        """
        Returns a bitmask of forbidden colors for position idx.
        A color c is forbidden if placing it at idx creates a 4-term AP
        (idx, idx-d, idx-2d, idx-3d) with fewer than 3 distinct colors.
        
        Logic for a specific AP with existing colors c1, c2, c3:
        - If {c1, c2, c3} has 1 distinct color: c cannot be that color.
        - If {c1, c2, c3} has 2 distinct colors: c cannot be either of them.
        - If {c1, c2, c3} has 3 distinct colors: no restriction on c.
        """
        forbidden = 0
        max_d = idx // 3
        for d in range(1, max_d + 1):
            c1 = colors[idx - d]
            c2 = colors[idx - 2 * d]
            c3 = colors[idx - 3 * d]
            
            distinct = {c1, c2, c3}
            if len(distinct) == 1:
                val = next(iter(distinct))
                if val < K:
                    forbidden |= (1 << val)
            elif len(distinct) == 2:
                for val in distinct:
                    if val < K:
                        forbidden |= (1 << val)
            # len 3: no restriction
        
        return forbidden

    def solve_with_K(K):
        """
        Deterministic backtracking search with heuristic color ordering.
        Returns a valid coloring using at most K colors, or None.
        """
        colors = [-1] * n
        
        def dfs(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return colors[:]
            
            forbidden = get_forbidden(idx, colors, K)
            
            # Generate valid colors and order them by "least constraining value"
            # A color that is already used in the "future" (higher indices) is generally
            # more flexible because it's less likely to create new unique constraints
            # for subsequent steps compared to introducing a new color.
            # However, since we are filling left-to-right, "future" is unknown.
            # A better heuristic: prefer colors that appear frequently in the prefix.
            # This tends to keep the color count low and creates denser clusters,
            # which might be easier to satisfy for APs?
            # Actually, for "at least 3 colors", spreading colors might be better?
            # Let's stick to a simple heuristic: try colors 0..K-1, but perhaps
            # randomize slightly or use a fixed pattern?
            # The seed program used smallest valid.
            # Let's try: Sort valid colors by their frequency in the prefix (descending).
            # This encourages reusing existing colors, which is good for minimizing K.
            
            prefix = colors[:idx]
            freq = [0] * K
            for c in prefix:
                if c != -1:
                    freq[c] += 1
            
            valid_colors = []
            for c in range(K):
                if not (forbidden & (1 << c)):
                    valid_colors.append(c)
            
            # Sort by frequency descending, then by color index for stability
            valid_colors.sort(key=lambda c: (-freq[c], c))
            
            for c in valid_colors:
                colors[idx] = c
                res = dfs(idx + 1)
                if res is not None:
                    return res
                colors[idx] = -1
            
            return None
        
        return dfs(0)

    # Try K=3 first (optimal if possible)
    result = solve_with_K(3)
    if result is not None:
        return result

    # Try K=4
    if time.time() < deadline - 2:
        result = solve_with_K(4)
        if result is not None:
            return result

    # Fallback: Randomized Greedy with Backtracking (Limited)
    # If deterministic fails, try a few random restarts with greedy + limited backtracking
    for attempt in range(20):
        if time.time() > deadline - 1:
            break
        
        colors = [-1] * n
        success = True
        
        def greedy_dfs(idx, depth_limit=5):
            nonlocal success
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            forbidden = get_forbidden(idx, colors, 128)
            valid = [c for c in range(128) if not (forbidden & (1 << c))]
            
            if not valid:
                return False
            
            random.shuffle(valid)
            
            for c in valid:
                colors[idx] = c
                if greedy_dfs(idx + 1, depth_limit - 1):
                    return True
                colors[idx] = -1
                
                if depth_limit == 0:
                    break # No backtracking at this level
            
            return False
        
        # Simple greedy without backtracking first
        for idx in range(n):
            forbidden = get_forbidden(idx, colors, 128)
            valid = [c for c in range(128) if not (forbidden & (1 << c))]
            if not valid:
                success = False
                break
            # Pick color with max frequency in prefix to minimize colors used
            prefix = colors[:idx]
            freq = {}
            for c in prefix:
                if c != -1:
                    freq[c] = freq.get(c, 0) + 1
            valid.sort(key=lambda c: (-freq.get(c, 0), c))
            colors[idx] = valid[0]
            
        if success and all(c != -1 for c in colors):
            # Verify
            valid = True
            for i in range(n):
                for d in range(1, i//3 + 1):
                    if i - 3*d >= 0:
                        if len({colors[i], colors[i-d], colors[i-2*d], colors[i-3*d]}) < 3:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return colors

    # Last resort: Return a known valid pattern if we had one, or zeros (invalid)
    # Since we must return something, and invalid scores 0, we return zeros.
    return [0] * n
# EVOLVE-BLOCK-END
