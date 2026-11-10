<<<<<<< SEARCH
    # Phase 2: Local search polish
    if time.time() < deadline - 60:
        ls_pts, ls_lns, ls_count = _local_search(pts, lines, start_time, deadline - 30)
        pts = ls_pts
        lines = ls_lns
    
    # Final validation
=======
    # Phase 2: Local search polish
    if time.time() < deadline - 90:
        ls_pts, ls_lns, ls_count = _local_search(pts, lines, start_time, deadline - 60)
        pts = ls_pts
        lines = ls_lns
    
    # Phase 3: Final exhaustive polish
    if time.time() < deadline - 15:
        fp_pts, fp_lns, fp_count = _final_polish(pts, lines, deadline - 5)
        pts = fp_pts
        lines = fp_lns
    
    # Final validation
>>>>>>> REPLACE
