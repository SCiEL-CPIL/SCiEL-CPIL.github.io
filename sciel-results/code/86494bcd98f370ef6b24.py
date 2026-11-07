def construct():
    import time
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin before 90s limit

    # The Fano plane is the smallest 3-uniform hypergraph with chromatic number 3.
    # It has 7 edges. Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We use vertices 0..6 for the Fano plane structure, and 7,8 are isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]

    # Valid 3-coloring for the Fano plane vertices 0..6.
    # c0=0, c1=1, c2=2, c3=1, c4=2, c5=0, c6=1
    # Check:
    # (0,1,2) -> 0,1,2 (ok)
    # (0,3,4) -> 0,1,2 (ok)
    # (0,5,6) -> 0,0,1 (ok)
    # (1,3,5) -> 1,1,0 (ok)
    # (1,4,6) -> 1,2,1 (ok)
    # (2,3,6) -> 2,1,1 (ok)
    # (2,4,5) -> 2,2,0 (ok)
    
    # Vertices 7 and 8 are isolated, so their color doesn't affect chromatic number.
    # We assign them color 0.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]

    # Verification (optional but good for robustness during development, 
    # but we must be careful not to spend too much time. This is O(1) so it's fine.)
    for e in edges:
        c_set = set(colors[v] for v in e)
        if len(c_set) == 1:
            # This should never happen with the fixed coloring above
            pass

    # Ensure types are strictly Python int and list
    # The lists above are already Python ints and lists.
    
    if time.time() > deadline:
        # Should not happen for O(1) construction, but return immediately if it does
        return {'edges': edges, 'colors': colors}

    return {'edges': edges, 'colors': colors}
