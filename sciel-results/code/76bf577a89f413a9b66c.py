<<<<<<< SEARCH
    # Final polish
    if best_pts is not None and _deadline_ok(8):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=150, margin=8)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
=======
    # Final polish - extended
    if best_pts is not None and _deadline_ok(8):
        counter = _full_counter(best_pts)
        p, _, c = _local_search(best_pts, counter, max_iter=250, margin=8)
        ps = max(c.values(), default=0)
        if ps >= best_score:
            best_pts = list(p)
            best_score = ps
        # Second polish pass
        if _deadline_ok(5):
            counter2 = _full_counter(best_pts)
            p2, _, c2 = _local_search(best_pts, counter2, max_iter=100, margin=5)
            ps2 = max(c2.values(), default=0)
            if ps2 >= best_score:
                best_pts = list(p2)
                best_score = ps2
>>>>>>> REPLACE
