<<<<<<< SEARCH
    # Process 7 pairs
    for pair_idx in range(7):
        c0 = 2 * pair_idx  # column offset
=======
    # Process 7 pairs
    for pair_idx in range(7):
        if time.time() > deadline:
            # If we time out (unlikely), return what we have so far?
            # No, partial result is invalid. 
            # But we must return *something*. 
            # Given the speed, this won't happen.
            pass
        c0 = 2 * pair_idx  # column offset
>>>>>>> REPLACE
