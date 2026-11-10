def _get_sym_positions(i, j, size):
    """Return all symmetric positions for pixel (i,j) under C4v."""
    s = size
    positions = [
        (i, j),
        (s-1-i, s-1-j),
        (j, i),
        (s-1-j, s-1-i),
        (s-1-j, i),
        (j, s-1-i),
        (s-1-i, j),
        (i, s-1-j)
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique
