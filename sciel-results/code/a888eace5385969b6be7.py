"""Bitwise-packing 3-color backtracking for 4-AP constraint on 128 positions."""

def construct():
    N = 128
    colors = [-1] * N
    colors[0] = 0

    # Precompute valid 4-APs (a, a+d, a+2d, a+3d) with 0-indexed positions
    # and 3-APs (a, a+d, a+2d) for the "no 3 same" invariant
    # We need: for any 4-AP (i, i+d, i+2d, i+3d), the 4 colors must have >= 3 distinct values.
    # Equivalent: not (all 4 same) and not (exactly 2 distinct).
    # The "no 3 same in any 3 consecutive terms of a 4-AP" is a necessary condition to avoid unsolvable states.
    
    # For efficiency, we'll precompute for each position i, the list of:
    # 1. 4-APs where i is the last term: (i-3d, i-2d, i-d, i)
    # 2. 3-APs where i is the last term: (i-2d, i-d, i) for the "no 3 same" check
    
    # Actually, let's just compute forbidden colors on the fly but with bitmasks.
    
    def compute_forbidden(i):
        """Compute bitmask of forbidden colors for position i, given current colors[0..i-1]."""
        forbidden = 0
        
        # 4-term progressions ending at i: (i-3d, i-2d, i-d, i)
        max_d_4 = i // 3
        for d in range(1, max_d_4 + 1):
            A = colors[i - 3 * d]
            B = colors[i - 2 * d]
            C = colors[i - d]
            
            # If any of A, B, C is -1 (unassigned), this constraint isn't fully active yet
            # But since we assign left-to-right, all positions < i are assigned.
            
            # Check the set {A, B, C}
            if A == B and B == C:
                # All three same: any color for i gives either {C} (size 1) or {C, new} (size 2). Both violations.
                # This state should be prevented by the 3-term check.
                return (1 << 128) - 1  # All forbidden
            
            # Determine distinct values in {A, B, C}
            if A == B:
                v1, v2 = A, C
            elif A == C:
                v1, v2 = A, B
            elif B == C:
                v1, v2 = B, A
            else:
                # All three distinct: any color for i is fine
                continue
            
            # Two distinct values v1, v2 (v1 != v2)
            # If i gets v1 or v2, the 4-set has size 2. Violation.
            # If i gets anything else, the 4-set has size 3. OK.
            forbidden |= (1 << v1)
            forbidden |= (1 << v2)
        
        # 3-term progressions ending at i: (i-2d, i-d, i)
        # If colors[i-2d] == colors[i-d] == K, then i cannot be K (else 3 same, leading to future violation)
        max_d_3 = i // 2
        for d in range(1, max_d_3 + 1):
            A = colors[i - 2 * d]
            B = colors[i - d]
            if A == B:
                forbidden |= (1 << A)
        
        return forbidden
    
    # Try backtracking with 3 colors (0, 1, 2)
    # If that fails, try with more colors via greedy
    
    def backtrack(i, max_color):
        """Try to assign colors[i..N-1] using colors 0..max_color-1. Returns True on success."""
        if i == N:
            return True
        
        forbidden = compute_forbidden(i)
        
        # Try each color from 0 to max_color-1
        for c in range(max_color):
            if not (forbidden & (1 << c)):
                colors[i] = c
                if backtrack(i + 1, max_color):
                    return True
                colors[i] = -1
        
        return False
    
    # First, try with 3 colors
    # We need to be careful about recursion depth. N=128 should be fine.
    import sys
    sys.setrecursionlimit(10000)
    
    # Reset colors
    colors = [-1] * N
    colors[0] = 0
    
    # Try 3 colors
    success = backtrack(1, 3)
    
    if success:
        # Verify the solution
        for i in range(N):
            max_d_4 = i // 3
            for d in range(1, max_d_4 + 1):
                vals = {colors[i-3*d], colors[i-2*d], colors[i-d], colors[i]}
                if len(vals) < 3:
                    success = False
                    break
            if not success:
                break
        
        if success:
            return colors
    
    # Fallback: greedy with more colors
    colors = [0] * N
    colors[0] = 0
    
    for i in range(1, N):
        forbidden = compute_forbidden(i)
        
        found = False
        for c in range(128):
            if not (forbidden & (1 << c)):
                colors[i] = c
                found = True
                break
        
        if not found:
            # Should not happen if logic is correct
            colors[i] = 0
    
    return colors
