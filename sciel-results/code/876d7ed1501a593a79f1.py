import random
import time

def construct():
    deadline = time.time() + 80  # 80 second safety margin
    
    # Try with increasing number of colors
    for num_colors in [3, 4, 5]:
        palette = list(range(num_colors))
        result = _try_construct(palette, deadline)
        if result is not None:
            return result
    
    # Fallback: return the seed program's result (shouldn't reach here)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

def _try_construct(palette, deadline):
    """Try to construct a valid coloring using the given palette."""
    n = 128
    colors = [-1] * n  # 0-indexed, colors[i] is color of position i+1
    
    def is_valid_placement(pos, color):
        """Check if placing 'color' at position pos (0-indexed) is valid.
        pos corresponds to position pos+1 in 1-indexed terms."""
        # Check all 4-term APs where pos+1 is the last term
        # AP: (a, a+d, a+2d, a+3d) where a+3d = pos+1
        # So a = pos+1-3d, and we need a >= 1, i.e., pos+1-3d >= 1
        for d in range(1, (pos+1)//3 + 1):
            a = pos + 1 - 3*d  # 1-indexed first term
            if a < 1:
                break
            # The four positions are a, a+d, a+2d, a+3d
            # In 0-indexed: a-1, a-1+d, a-1+2d, a-1+3d = pos
            idx0 = a - 1
            idx1 = a - 1 + d
            idx2 = a - 1 + 2*d
            idx3 = pos  # = a-1+3d
            c0 = colors[idx0]
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = color
            distinct = len({c0, c1, c2, c3})
            if distinct < 3:
                return False
        return True
    
    def backtrack(pos, depth=0):
        """Recursively assign colors from position pos onwards."""
        if time.time() > deadline:
            return False
        
        if pos == n:
            return True
        
        # Try colors in shuffled order
        shuffled = palette[:]
        random.shuffle(shuffled)
        
        for color in shuffled:
            if is_valid_placement(pos, color):
                colors[pos] = color
                if backtrack(pos + 1, depth + 1):
                    return True
                colors[pos] = -1
        
        return False
    
    # Use iterative approach with restarts to avoid deep recursion
    # Try multiple random restarts
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Reset
        colors[:] = [-1] * n
        
        # Greedy with occasional backtracking
        success = True
        for pos in range(n):
            if time.time() > deadline:
                success = False
                break
            
            shuffled = palette[:]
            random.shuffle(shuffled)
            
            placed = False
            for color in shuffled:
                if is_valid_placement(pos, color):
                    colors[pos] = color
                    placed = True
                    break
            
            if not placed:
                # Dead end - need to backtrack
                # Simple backtracking: try to find a position to reassign
                # For now, just fail this attempt
                success = False
                break
        
        if success:
            # Verify the solution
            if _verify(colors):
                return colors[:]
    
    return None

def _verify(colors):
    """Verify that the coloring satisfies all constraints."""
    n = len(colors)
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            idx0 = a - 1
            idx1 = a - 1 + d
            idx2 = a - 1 + 2*d
            idx3 = a - 1 + 3*d
            distinct = len({colors[idx0], colors[idx1], colors[idx2], colors[idx3]})
            if distinct < 3:
                return False
    return True
