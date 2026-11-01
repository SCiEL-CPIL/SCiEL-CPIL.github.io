def _round_to_half_integer(mat):
    """Round each entry to the nearest half-integer."""
    return [[round(2*x)/2 for x in row] for row in mat]
