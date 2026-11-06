"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _get_all_progressions(n=128):
    """Precompute all nonconstant 4-term arithmetic progressions."""
    progs = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            e = a + 3 * d
            if e < n:
                progs.append((a, a + d, a + 2 * d, e))
    return progs

def _check_valid(colors, ap_list):
    """Check if all nonconstant 4-term progressions use at least 3 distinct colors."""
    for ap in ap_list:
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Fast check for < 3 distinct colors
        # If all same, or only 2 distinct values present
        if c0 == c1 == c2 == c3:
            return False
        # Check pairs for 2-color patterns
        # A 4-AP has < 3 colors if it uses 1 or 2 colors.
        # Using 2 colors means the set size is 2.
        # We can just use set for simplicity, it's fast enough for 2700 APs.
        if len({c0, c1, c2, c3}) < 3:
            return False
    return True

def _local_search(k, deadline, ap_list, n=128):
    """Local search to find a valid coloring with k colors."""
    # Initialize random coloring
    colors = [random.randint(0, k - 1) for _ in range(n)]
    
    # Try to find a valid solution
    while time.time() < deadline:
        # Find a violation
        violation_ap = None
        for ap in ap_list:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violation_ap = ap
                break
        
        if violation_ap is None:
            # No violations, we have a valid coloring
            return colors
        
        # Pick a random position in the violating AP to recolor
        idx = random.choice(violation_ap)
        old_color = colors[idx]
        
        # Try a new random color
        new_color = random.randint(0, k - 1)
        while new_color == old_color:
            new_color = random.randint(0, k - 1)
        
        colors[idx] = new_color
        
        # Occasionally restart if stuck? 
        # For now, simple random walk is robust for this size.
        # To improve convergence, we can bias towards colors that appear less in the AP
        # but random is often sufficient.
        
    return None

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50
    
    n = 128
    ap_list = _get_all_progressions(n)
    
    # 3 colors are impossible (W(3,4)=76 < 128 implies monochromatic 4-AP exists)
    # So minimum colors is at least 4.
    # Try k=4, then 5, then 6.
    for k in [4, 5, 6, 7]:
        if time.time() > deadline - 5:
            break
            
        # Run local search
        result = _local_search(k, deadline - 5, ap_list, n)
        
        if result is not None:
            # Verify before returning
            if _check_valid(result, ap_list):
                # Minimize labels actually used?
                # The problem says "Minimize the number of labels actually used".
                # If we used k=4 but only 3 labels appear in the array, that's better.
                # However, since 3 is impossible, we likely use all 4.
                # We can try to map the used colors to 0..m-1 to ensure compactness,
                # but the count is what matters.
                # The evaluator measures "raw value" which is the number of labels used.
                # So we just return the array. The count is derived from the array.
                return result
    
    # Fallback: If local search fails, use a deterministic pattern that is likely valid
    # or a high-color solution.
    # x % 7 is a good candidate for higher k.
    # Let's try to construct a valid one with k=8 using a simple pattern if search failed.
    # Or just return a random valid one found with higher k.
    
    # Try k=8 with a short time limit
    if time.time() < deadline - 2:
        result = _local_search(8, deadline - 2, ap_list, n)
        if result is not None and _check_valid(result, ap_list):
            return result
            
    # Emergency: Use x % 10 (likely valid)
    colors = [x % 10 for x in range(n)]
    if _check_valid(colors, ap_list):
        return colors
        
    # Absolute fallback: unique colors (always valid)
    return list(range(n))

# EVOLVE-BLOCK-END
