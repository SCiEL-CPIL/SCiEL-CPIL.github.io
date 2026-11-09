<<<<<<< SEARCH
    # Final intensive local search on best
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=1000)
    
    return best_classes if best_classes is not None else []
=======
    # Final intensive local search on best
    if best_classes is not None:
        best_classes = local_search(best_classes, max_iter=1000)
        # Residue reassignment polish to free space for missing small moduli
        if time.time() < deadline - 5.0:
            best_classes = residue_polish(best_classes, max_iter=300)
        # One more local search pass after polish
        if time.time() < deadline - 3.0:
            best_classes = local_search(best_classes, max_iter=500)
    
    return best_classes if best_classes is not None else []
>>>>>>> REPLACE
