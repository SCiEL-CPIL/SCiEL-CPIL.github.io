    # Strategy:
    # 1. Try to find rank 14 for 2x3x5. If found, construct rank 28 for 2x3x10.
    # 2. Try rank 15, 16, 17, 18 for 2x3x5.
    # 3. If none, return schoolbook.
    
    block_n, block_m, block_p = 2, 3, 5
    
    best_full_result = None
    best_rank = n * m * p
    
    # Try ranks for the block
    # Known lower bound for 2x3x5 is 14.
    target_block_ranks = [14, 15, 16, 17, 18, 19, 20]
    
    for r_block in target_block_ranks:
        if time.time() > deadline - 10:
            break
            
        # Budget for search: more attempts for lower ranks (harder) or fewer?
        # Usually lower rank is harder. But 14 is the min.
        # Let's give 200 attempts for 14, 100 for others.
        attempts = 200 if r_block == 14 else 100
        
        block_decomp = find_block_decomposition(block_n, block_m, block_p, r_block, attempts)
        
        if block_decomp is not None:
            # ... construction code ...
            if verify_block(U_full, V_full, W_full, n, m, p, tol=1e-9):
                best_full_result = {"U": U_full, "V": V_full, "W": W_full}
                best_rank = 2 * r_block
                break # Found a valid decomposition, move to next lower rank? No, we started at lowest.
            else:
                # Construction error? Should not happen if block was valid.
                print("Construction verification failed")
                continue
        
        # If we found a rank, we break the loop because we iterate from lowest rank up.
        # Wait, the loop is `for r_block in target_block_ranks`.
        # If we find rank 14, we break.
        # If we don't find 14, we try 15, etc.
        # So if best_full_result is not None, we should break.
        if best_full_result is not None:
            break

    if best_full_result is not None:
        return best_full_result
    else:
        return make_schoolbook()
    