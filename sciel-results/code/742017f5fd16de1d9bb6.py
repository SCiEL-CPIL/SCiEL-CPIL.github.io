def _greedy_construct_np(points, primes, fixed_choices):
    """
    Run greedy construction using NumPy for speed.
    points: numpy array of integers.
    """
    for p in primes:
        # We only need to satisfy the condition for primes p <= len(points)
        # For p > len(points), it is impossible to cover all p residue classes,
        # so the condition is automatically met.
        if p > len(points):
            break
            
        if p in fixed_choices:
            r = fixed_choices[p]
            points = points[points % p != r]
            continue
            
        if len(points) < p:
            continue
            
        # Count points in each residue class mod p
        residues = points % p
        class_counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(class_counts)
        if min_count > 0:
            # Find the first class with this minimum count
            min_indices = np.where(class_counts == min_count)[0]
            r = min_indices[0]
            points = points[points % p != r]
            
    return points
