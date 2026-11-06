import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    # Precompute all 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0 and a+3d <= N
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map index to list of AP indices it belongs to
    # Indices are 0-based in arrays, 1-based in logic. Let's use 0-based arrays.
    # positions 0..127 correspond to 1..128
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos - 1].append(idx)

    def check_ap(ap_idx, colors):
        """
        Check if the AP at ap_idx is satisfiable given current colors.
        Returns True if satisfiable, False otherwise.
        Also returns a list of (pos, forbidden_colors) if domain reduction is possible?
        For now, just check satisfiability.
        """
        a, b, c, d = aps[ap_idx]
        ca = colors[a-1]
        cb = colors[b-1]
        cc = colors[c-1]
        cd = colors[d-1]
        
        assigned = []
        unassigned_count = 0
        for val in (ca, cb, cc, cd):
            if val != -1:
                assigned.append(val)
            else:
                unassigned_count += 1
        
        S = set(assigned)
        r = 3 - len(S)
        
        if r < 0:
            return True
        if r > unassigned_count:
            return False
        if r == 0:
            return True
        
        # If r == 1:
        # Need at least 1 new color from unassigned.
        # If unassigned_count == 1, that variable cannot be in S.
        # This is a constraint, but for simple check, it's satisfiable.
        # If r == 2:
        # Need 2 new distinct colors.
        # If unassigned_count == 2, both must be in D\S and distinct.
        
        return True

    def solve(k, colors, idx):
        """
        Backtracking solver.
        k: max colors allowed (0..k-1)
        colors: list of length N, -1 for unassigned
        idx: current index to assign (0..N-1)
        """
        if time.time() > deadline:
            return None
        
        if idx == N:
            return colors[:]
        
        # Symmetry breaking: c_i must be in {0, ..., min(k-1, i)}
        # Note: i is 0-based index. Position is i+1.
        # The number of distinct colors used so far in the prefix is at most idx+1.
        # We can restrict domain to 0..min(k-1, idx)
        max_val = min(k - 1, idx)
        
        for v in range(max_val + 1):
            colors[idx] = v
            
            # Check constraints for all APs containing idx
            valid = True
            for ap_idx in ap_indices[idx]:
                if not check_ap(ap_idx, colors):
                    valid = False
                    break
            
            if valid:
                # Recurse
                result = solve(k, colors, idx + 1)
                if result is not None:
                    return result
            
            colors[idx] = -1
        
        return None

    # Try k from 3 upwards
    for k in range(3, 129):
        colors = [-1] * N
        # Symmetry breaking for the first few variables to reduce search space
        # c_1 (idx 0) must be 0
        colors[0] = 0
        # c_2 (idx 1) can be 0 or 1 (if k>=2)
        # c_3 (idx 2) can be 0, 1, 2 (if k>=3)
        
        # We can just let the solver handle it with the domain restriction max_val = min(k-1, idx)
        # This effectively breaks symmetry.
        
        # Check time
        if time.time() > deadline:
            break
            
        result = solve(k, colors, 1) # Start from idx 1 since idx 0 is fixed
        
        if result is not None:
            # Ensure we return a list of integers
            # The colors are 0..k-1
            return result
        
        # If k=3 failed, try k=4, etc.
        # If time is running out, we might not finish.
        # We should probably return the best partial result or a random valid one?
        # No, the task requires a valid coloring.
        # If we run out of time, we should return a fallback?
        # The seed program returns a valid coloring (greedy).
        # We should keep the seed's result as a fallback if we don't find a better one in time.
        
    # Fallback: use the seed's greedy approach if we haven't found a solution
    # But the seed is slow. Let's just return a simple valid coloring if we fail.
    # A coloring where c_i = i % k is not necessarily valid.
    # The greedy algorithm in the seed is guaranteed to work but uses many colors.
    # We can replicate the seed's logic as a fallback.
    
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
