<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if k == 5 and g == 9:
        # Try Cayley graph on A6
        try:
            n, edges = find_cayley_a6()
            if n is not None:
                return n, edges
        except Exception:
            pass
        
        # Try Cayley graph on PSL(2, 8)
        try:
            n, edges = find_cayley_psl28()
            if n is not None:
                return n, edges
        except Exception:
            pass
            
    # Fallback to seed logic (L-U-W)
    # ... (copy seed code)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Use L-U-W construction with optimal d for girth
    # L-U-W graph has girth 2d+1
    # We need 2d+1 >= g => d >= (g-1)/2
    # So d = ceil((g-1)/2) = g // 2
    q = k
    d = max(3, g // 2)
>>>>>>> REPLACE
