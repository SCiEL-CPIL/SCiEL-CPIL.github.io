    # Final intensive polish on the best points found so far
    if best_points is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 5.0:
            final_pts, final_score = _polish_points(best_points, time_limit=remaining)
            try_update(final_score, final_pts)
    