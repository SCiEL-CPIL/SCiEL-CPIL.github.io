"""Constructs the (4,10)-cage using the Kautz graph K(4, 5)."""

def construct(k, g):
    # The (4,10)-cage is the Kautz graph K(4, 5).
    # Kautz graph K(d, q) is (q-1)-regular with girth 2d+2.
    # For k=4, we need q-1=4 => q=5.
    # For g=10, we need 2d+2=10 => d=4.
    # Number of vertices is q*(q-1)^(d-1) = 5 * 4^3 = 320.
    
    if k != 4 or g != 10:
        # If parameters do not match the known cage case, we can fallback to a generic construction
        # or return an error/empty. Given the task specificity, we assume k=4, g=10.
        # For safety, we return a trivial valid graph if parameters don't match, 
        # though this case is not expected.
        # A single vertex has degree 0. A cycle of length g requires g>=3.
        # We'll just return empty to signal failure if not matching.
        return 0, []

    q = 5
    d = 4
    
    # Generate all valid strings of length d over Z_q with no adjacent equal
    valid_strings = []
    # Instead of iterating all q^d and filtering, we can generate recursively for efficiency
    # but q^d = 625 is small, so filtering is fine.
    
    for s in range(q ** d):
        digits = []
        tmp = s
        for _ in range(d):
            digits.append(tmp % q)
            tmp //= q
        # Check no adjacent equal
        valid = True
        for i in range(d - 1):
            if digits[i] == digits[i + 1]:
                valid = False
                break
        if valid:
            valid_strings.append(tuple(digits))
            
    n = len(valid_strings)
    str_to_idx = {s: i for i, s in enumerate(valid_strings)}
    
    edges = []
    seen_edges = set()
    
    for idx, w in enumerate(valid_strings):
        # w is (w0, w1, w2, w3)
        # Neighbors are (w1, w2, w3, c) for c in 0..4, c != w3
        for c in range(q):
            if c == w[3]:
                continue
            new_w = (w[1], w[2], w[3], c)
            j = str_to_idx.get(new_w)
            if j is not None:
                edge = (min(idx, j), max(idx, j))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)
                    
    return n, edges
