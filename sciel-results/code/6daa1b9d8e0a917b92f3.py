<<<<<<< SEARCH
def design(problem):
    """Design broadband AR coating using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    band = problem["band_nm"]
    center_wl = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    
    # Candidate patterns: alternating L/H starting with L or H, various lengths
    # Also include some non-alternating patterns
    patterns = []
    
    # Alternating patterns
    for n in range(1, min(max_layers, 15) + 1):
        # Start with L
        pat_l = ['L' if i % 2 == 0 else 'H' for i in range(n)]
        patterns.append(pat_l)
        # Start with H
        if n > 1:
            pat_h = ['H' if i % 2 == 0 else 'L' for i in range(n)]
            patterns.append(pat_h)
    
    # Some specific known-good patterns for broadband AR
    # Classic 5-layer: L H L H L (or H L H L H)
    # 7-layer: L H L H L H L
    # Add some mixed patterns
    patterns.append(['L', 'H', 'L', 'H', 'L'])
    patterns.append(['H', 'L', 'H', 'L', 'H'])
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])
    patterns.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])
    patterns.append(['L', 'L', 'H', 'H', 'L', 'L', 'H', 'H'])
    patterns.append(['H', 'H', 'L', 'L', 'H', 'H', 'L', 'L'])
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    best_result = None
    best_cost = float('inf')
    
    # Sort by number of layers (start with moderate ones that are likely good)
    # Actually, start with patterns known to be good for broadband AR
    # Prioritize: 5, 7, 9 layer alternating patterns first
    priority = []
    for n in [5, 7, 9, 3, 11, 13, 1, 2, 4, 6, 8, 10, 12, 14, 15]:
        for p in unique_patterns:
            if len(p) == n:
                priority.append(p)
    
    for pattern in priority:
        if time.time() > deadline:
            break
        result = _try_pattern(pattern, problem, start_time, deadline)
        if result is not None:
            materials, thicknesses, cost = result
            if cost < best_cost:
                best_cost = cost
                best_result = (materials, thicknesses)
    
    if best_result is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [center_wl / (4.0 * mat["L"])]
    
    return best_result
=======
def design(problem):
    """Design broadband AR coating using simulated annealing with structural changes."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mat = problem["materials"]
    band = problem["band_nm"]
    center_wl = 0.5 * (band[0] + band[1])
    max_layers = problem["max_layers"]
    
    # Structural change: focus on fewer, higher-quality patterns with more thorough optimization
    # Instead of trying many patterns briefly, try fewer patterns more thoroughly
    patterns = []
    
    # Key patterns for broadband AR: alternating stacks with odd number of layers
    # These are known to perform well for broadband antireflection
    patterns.append(['L', 'H', 'L', 'H', 'L'])  # 5-layer, starts and ends with L
    patterns.append(['H', 'L', 'H', 'L', 'H'])  # 5-layer, starts and ends with H
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L'])  # 7-layer
    patterns.append(['H', 'L', 'H', 'L', 'H', 'L', 'H'])  # 7-layer
    patterns.append(['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L'])  # 9-layer
    patterns.append(['H', 'L', 'H', 'L', 'H', 'L', 'H', 'L', 'H'])  # 9-layer
    
    # Add a few non-alternating patterns for diversity
    patterns.append(['L', 'L', 'H', 'H', 'L', 'L', 'H', 'H'])  # 8-layer paired
    patterns.append(['H', 'H', 'L', 'L', 'H', 'H', 'L', 'L'])  # 8-layer paired
    
    # Remove duplicates
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    best_result = None
    best_cost = float('inf')
    
    # Allocate time more evenly: give each pattern a fair share of the budget
    time_per_pattern = (deadline - start_time) / len(unique_patterns)
    
    for pattern in unique_patterns:
        if time.time() > deadline:
            break
        # Give this pattern its allocated time
        pattern_deadline = start_time + (time_per_pattern * (unique_patterns.index(pattern) + 1))
        result = _try_pattern(pattern, problem, start_time, pattern_deadline)
        if result is not None:
            materials, thicknesses, cost = result
            if cost < best_cost:
                best_cost = cost
                best_result = (materials, thicknesses)
    
    if best_result is None:
        # Fallback: single quarter-wave L layer
        return ["L"], [center_wl / (4.0 * mat["L"])]
    
    return best_result
>>>>>>> REPLACE
