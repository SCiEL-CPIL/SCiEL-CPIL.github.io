def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 115.0
    
    result = _block_decomp(n, m, p, deadline)
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook_decomp(n, m, p)
    
    return result
