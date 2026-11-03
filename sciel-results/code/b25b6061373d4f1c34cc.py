def _to_flat_list(arr):
    """Convert (n, m, R) to (n*m, R) list of lists."""
    n, m, R = arr.shape
    return arr.reshape(n*m, R).tolist()
