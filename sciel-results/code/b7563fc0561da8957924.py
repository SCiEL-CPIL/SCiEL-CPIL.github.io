<<<<<<< SEARCH
    # Try the block construction
    result = build_4x4x7_from_blocks()
    err = verify(result["U"], result["V"], result["W"], 56)
    if err < 1e-9:
        best_result = result
        best_rank = 56
    else:
        # Fallback to schoolbook
        pass

    return best_result
=======
    # Try the block construction
    result = build_4x4x7_from_blocks()
    err = verify(result["U"], result["V"], result["W"], 56)
    if err < 1e-9:
        best_result = result
        best_rank = 56
    else:
        # Debug: print error
        pass

    return best_result
>>>>>>> REPLACE

Actually, I realize I need to be more careful. Let me just implement a correct version from scratch with proper index handling:

