"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 0..n-1
    # Note: positions are 0-indexed in our array, but the problem says 1..128.
    # A progression is (i, i+d, i+2d, i+3d) where 0 <= i, i+3d < n.
    progressions = []
    for i in range(n):
        max_d = (n - 1 - i) // 3
        for d in range(1, max_d + 1):
            progressions.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_progs = len(progressions)
    
    # For each position, store the progressions it is part of and the other indices
    pos_progs = [[] for _ in range(n)]
    for pi, (a, b, c, e) in enumerate(progressions):
        for pos in (a, b, c, e):
            others = [p for p in (a, b, c, e) if p != pos]
            pos_progs[pos].append(others)
    
    def try_coloring(num_colors):
        """
        Attempts to find a valid coloring using exactly 'num_colors' distinct colors (0 to num_colors-1).
        Returns the list of colors if found, else None.
        """
        colors = [-1] * n
        
        def is_valid_assignment(pos, color):
            """
            Check if assigning 'color' to 'pos' is consistent with all progressions 
            involving 'pos' where all other positions are already assigned.
            """
            for others in pos_progs[pos]:
                # Check if all others are assigned
                all_assigned = True
                other_colors = []
                for o in others:
                    if colors[o] == -1:
                        all_assigned = False
                        break
                    other_colors.append(colors[o])
                
                if not all_assigned:
                    continue
                
                # If all others are assigned, check the constraint
                # The set of colors in the progression must have size >= 3
                prog_colors = set(other_colors)
                prog_colors.add(color)
                if len(prog_colors) < 3:
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            # Try each color
            for color in range(num_colors):
                if is_valid_assignment(idx, color):
                    colors[idx] = color
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        
        # Symmetry breaking: fix first position to 0
        colors[0] = 0
        return backtrack(1)

    best_colors = None
    best_count = 128 # Initialize with worst case
    
    # Try 3 colors first
    if time.time() < deadline - 5:
        if try_coloring(3):
            # Reconstruct colors from the internal state? 
            # The try_coloring function modifies a local 'colors' list. 
            # We need to return it. Let's refactor try_coloring to return the list.
            pass # Placeholder, need to fix return logic

    # Refactored try_coloring to return the solution
    def try_coloring_return(num_colors):
        colors = [-1] * n
        
        def is_valid_assignment(pos, color):
            for others in pos_progs[pos]:
                all_assigned = True
                other_colors = []
                for o in others:
                    if colors[o] == -1:
                        all_assigned = False
                        break
                    other_colors.append(colors[o])
                
                if not all_assigned:
                    continue
                
                prog_colors = set(other_colors)
                prog_colors.add(color)
                if len(prog_colors) < 3:
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            for color in range(num_colors):
                if is_valid_assignment(idx, color):
                    colors[idx] = color
                    if backtrack(idx + 1):
                        return True
                    colors[idx] = -1
            return False
        
        colors[0] = 0
        if backtrack(1):
            return colors
        return None

    # Try 3 colors
    if time.time() < deadline - 5:
        sol = try_coloring_return(3)
        if sol is not None:
            return sol
            
    # Try 4 colors
    if time.time() < deadline - 5:
        sol = try_coloring_return(4)
        if sol is not None:
            return sol

    # Fallback to greedy construction if backtracking fails or times out
    # This ensures we always return a valid coloring, even if it uses more colors
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            # Check all 4-term progressions ending at position x-1
            # a, a+d, a+2d, a+3d = x-1
            # a = x-1-3d
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                a = x - 1 - 3 * d
                if a < 0:
                    continue
                c0 = colors[a]
                c1 = colors[a + d]
                c2 = colors[a + 2 * d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            # Also need to check progressions where x-1 is not the last term?
            # The greedy approach in the seed checked:
            # 1. all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            #    This checks progressions where x-1 is the LAST term (i+3d = x-1).
            #    Wait, x-1-j*d for j=1,2,3 gives x-1-d, x-1-2d, x-1-3d.
            #    So the progression is (x-1-3d, x-1-2d, x-1-d, x-1).
            #    This covers all progressions ending at x-1.
            
            # 2. all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            #    This checks progressions where x-1 is the THIRD term?
            #    If x-1 is the 3rd term (i+2d), then the terms are i, i+d, i+2d, i+3d.
            #    i+2d = x-1 => i = x-1-2d.
            #    Terms: x-1-2d, x-1-d, x-1, x-1+d.
            #    The constraint is that the 4 terms must have >= 3 colors.
            #    We are assigning color to x-1. The other three are x-1-2d, x-1-d, x-1+d.
            #    x-1+d is NOT assigned yet (it's in the future).
            #    So we can't fully check these progressions yet in a simple forward greedy pass 
            #    unless we ensure that the current assignment doesn't force a violation later.
            #    The seed's second condition: any(colors[x-1-j*d]!=color for j in (1,2))
            #    This means: color != colors[x-1-d] AND color != colors[x-1-2d].
            #    This ensures that the first three terms (x-1-2d, x-1-d, x-1) are not all the same color.
            #    If they were all the same, then even if the 4th term is different, we have only 2 colors.
            #    Wait, if the first three are same color C, and 4th is D, we have {C, D} size 2. Invalid.
            #    So we must ensure the first three are not all C.
            #    By forcing color != colors[x-1-d] and color != colors[x-1-2d], we ensure 
            #    that (colors[x-1-2d], colors[x-1-d], color) are not all equal.
            #    This is a necessary condition for validity, assuming the 4th term can be chosen later 
            #    to make it valid? No, if the first 3 are all C, the 4th MUST be not C to have >=3? 
            #    No, if first 3 are C, 4th is D, set is {C,D} size 2. Still invalid.
            #    So if first 3 are all C, it's impossible to make the progression valid with >=3 colors 
            #    unless... wait. 
            #    If C1=C2=C3=C, then no matter what C4 is, the set is {C, C4} which has size <= 2.
            #    So we MUST avoid having the first 3 terms of any 4-AP be the same color.
            #    The seed's condition enforces this for progressions where x-1 is the 3rd term.
            
            if valid:
                # Check the "3rd term" constraint
                max_d_3rd = (x - 1) // 2
                for d in range(1, max_d_3rd + 1):
                    # Progression: x-1-2d, x-1-d, x-1, x-1+d
                    # We are assigning x-1.
                    # We need to ensure (x-1-2d, x-1-d, x-1) are not all the same.
                    if colors[x-1-2*d] == color and colors[x-1-d] == color:
                        valid = False
                        break
            
            if valid:
                colors.append(color)
                found = True
                break
        
        if not found:
            # Should not happen if 128 colors are available
            colors.append(0)
            
    return colors

# EVOLVE-BLOCK-END
