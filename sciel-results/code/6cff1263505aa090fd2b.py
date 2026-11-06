import time
import sys

def construct():
    """
    Constructs a coloring of 1..128 with minimum colors such that
    every nonconstant 4-AP uses at least 3 distinct colors.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    
    # Precompute APs
    # aps[i] contains tuples (a, d, pos_in_ap) 
    # where i is the element at index pos_in_ap (0, 1, 2, 3) in a, a+d, a+2d, a+3d
    # We need to know which APs become "3 assigned" or "4 assigned" when we assign i.
    
    # For forward checking, we need to know, for each i, which APs it completes to 3 or 4 assigned variables.
    # Let's store for each i:
    # - ap_4th[i]: list of (a, d) such that i = a+3d. (Completes the AP)
    # - ap_3rd[i]: list of (a, d) such that i = a+2d. (Makes 3 assigned: a, a+d, i)
    # - ap_2nd[i]: list of (a, d) such that i = a+d. (Makes 2 assigned)
    # - ap_1st[i]: list of (a, d) such that i = a. (Makes 1 assigned)
    
    ap_4th = [[] for _ in range(N)]
    ap_3rd = [[] for _ in range(N)]
    ap_2nd = [[] for _ in range(N)]
    ap_1st = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            a0 = a
            a1 = a + d
            a2 = a + 2 * d
            a3 = a + 3 * d
            if a3 > N:
                break
            # 0-indexed for arrays
            ap_1st[a0 - 1].append((a1 - 1, a2 - 1, a3 - 1))
            ap_2nd[a1 - 1].append((a0 - 1, a2 - 1, a3 - 1))
            ap_3rd[a2 - 1].append((a0 - 1, a1 - 1, a3 - 1))
            ap_4th[a3 - 1].append((a0 - 1, a1 - 1, a2 - 1))
            
    def solve(K):
        colors = [-1] * N
        # domains[i] is a bitmask of allowed colors for position i
        # Initially all colors 0..K-1 are allowed
        full_mask = (1 << K) - 1
        domains = [full_mask] * N
        
        def is_valid_color(idx, color):
            """
            Checks if assigning 'color' to 'idx' is consistent with already assigned neighbors
            for APs where idx is the 4th element.
            """
            # Check APs where idx is the 4th element
            for a0, a1, a2 in ap_4th[idx]:
                c0 = colors[a0]
                c1 = colors[a1]
                c2 = colors[a2]
                # All must be assigned
                if c0 == -1 or c1 == -1 or c2 == -1:
                    continue
                # Check if {c0, c1, c2, color} has size >= 3
                s = {c0, c1, c2, color}
                if len(s) < 3:
                    return False
            return True

        def propagate(idx, color):
            """
            Updates domains of future variables based on the assignment colors[idx] = color.
            Returns False if any domain becomes empty.
            """
            # Case 1: idx is the 3rd element (a2). a0, a1 are assigned. a3 is unassigned.
            for a0, a1, a3 in ap_3rd[idx]:
                c0 = colors[a0]
                c1 = colors[a1]
                if c0 == -1 or c1 == -1:
                    continue
                # We have c0, c1, color assigned. a3 is unassigned.
                # Determine forbidden colors for a3.
                assigned_colors = {c0, c1, color}
                if len(assigned_colors) == 1:
                    # All same. a3 cannot be that color.
                    forbidden = assigned_colors
                elif len(assigned_colors) == 2:
                    # Two distinct. a3 cannot be any of these 2.
                    forbidden = assigned_colors
                else:
                    # 3 distinct. a3 can be anything (adding any color keeps size >= 3).
                    forbidden = set()
                
                if forbidden:
                    # Remove forbidden colors from domains[a3]
                    mask = domains[a3]
                    new_mask = mask
                    for c in forbidden:
                        new_mask &= ~(1 << c)
                    if new_mask == 0:
                        return False
                    domains[a3] = new_mask
            
            # Case 2: idx is the 2nd element (a1). a0 is assigned. a2, a3 unassigned.
            # This doesn't force immediate domain reduction on a single variable strongly,
            # unless we track pairs, which is complex. We skip for now.
            
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == N:
                return True
            
            # Get available colors for idx
            mask = domains[idx]
            if mask == 0:
                return False
            
            # Try colors in some order
            # Heuristic: Try colors that are most constrained? Or just 0..K-1
            # Let's try 0..K-1
            for color in range(K):
                if not (mask & (1 << color)):
                    continue
                
                if not is_valid_color(idx, color):
                    continue
                
                # Assign
                colors[idx] = color
                # Save domains to backtrack
                saved_domains = domains[:]
                
                if propagate(idx, color):
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                colors[idx] = -1
                domains = saved_domains
            
            return False

        return backtrack(0)

    # Try K=3
    if solve(3):
        return colors
    
    # If K=3 failed, try K=4
    if solve(4):
        return colors
        
    # If K=4 failed, try K=5
    if solve(5):
        return colors
        
    # Fallback: Greedy with many colors (should not happen if K=3/4/5 works)
    # But we need to return *something* valid.
    # A safe fallback is using x % 10 or similar, but we need to ensure validity.
    # Actually, if K=5 fails, we can just use a large K and greedy.
    # But 3 colors should definitely work for N=128.
    
    # Emergency fallback: return a coloring with 128 colors (all distinct)
    # This is always valid.
    return list(range(128))
