<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Primary: Kronecker product of Strassen (2x2) and Winograd (3x3) -> rank 161
    best = _block_decomposition_6x6x6()
    
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Secondary: Try union with reduction
    if time.time() < deadline - 10.0:
        union = _union_decomposition_with_reduction()
        if _verify_fast(union["U"], union["V"], union["W"], n, m, p):
            # Check if union is actually better (shouldn't be, but verify)
            if len(union["U"][0]) < 161:
                return union
    
    # Fallback: Strassen + schoolbook (rank 189)
    best = _block_decomposition_strassen_schoolbook()
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Final fallback: schoolbook (rank 216)
    best = _schoolbook(n, m, p)
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Primary: Kronecker product of Strassen (2x2) and Winograd (3x3) -> rank 161
    best = _block_decomposition_6x6x6()
    
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        # Try to reduce rank further via greedy elimination
        if time.time() < deadline - 20.0:
            U_red, V_red, W_red, new_rank = _reduce_rank_by_elimination(
                best["U"], best["V"], best["W"], n, m, p, max_iterations=200
            )
            if new_rank < 161:
                return {"U": U_red, "V": V_red, "W": W_red}
        return best
    
    # Fallback: Strassen + schoolbook (rank 189)
    best = _block_decomposition_strassen_schoolbook()
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Final fallback: schoolbook (rank 216)
    best = _schoolbook(n, m, p)
    return best
>>>>>>> REPLACE
