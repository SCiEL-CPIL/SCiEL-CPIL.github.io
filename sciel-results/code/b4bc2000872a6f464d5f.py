def _local_search(code, q, n, radius, deadline):
    """
    Local search: primarily remove redundant codewords using fast counting method.
    """
    code = list(code)
    # We can run multiple passes of redundancy removal
    # Each pass removes at least one if possible
    while time.time() < deadline:
        new_code = _remove_redundant_fast(code, q, n, radius, deadline)
        if len(new_code) == len(code):
            break
        code = new_code
    return code
