def _is_latin(sq, n):
    """Check if sq is a valid Latin square of order n."""
    for r in range(n):
        row_set = set(sq[r])
        if len(row_set) != n:
            return False
        col_set = set(sq[c][r] for c in range(n))
        if len(col_set) != n:
            return False
    return True
