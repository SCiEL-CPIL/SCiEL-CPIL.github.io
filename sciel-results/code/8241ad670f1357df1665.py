    improved = True
    while improved:
        improved = False
        for idx in range(len(best)):
            # ...
            if _is_sorted_fast(candidate): # EXPENSIVE
