<<<<<<< SEARCH
    # ---------- Phase C: Local search ----------
    if time.time() < deadline - 50:
        np.random.seed(12345)
        refined_points, refined_score = local_search(best_points, max_iters=10000)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()

    # ---------- Phase D: High-precision polish ----------
    if time.time() < deadline - 5:
        polished_points, polished_score = polish_points(best_points, max_time=min(35.0, deadline - time.time() - 3))
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points.copy()
=======
    # ---------- Phase C: Vectorized local search ----------
    if time.time() < deadline - 60:
        np.random.seed(12345)
        refined_points, refined_score = vectorized_local_search(best_points, max_iters=50000)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()

    # ---------- Phase D: Standard local search for fine-tuning ----------
    if time.time() < deadline - 55:
        np.random.seed(99999)
        refined_points, refined_score = local_search(best_points, max_iters=5000)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()

    # ---------- Phase E: High-precision polish ----------
    if time.time() < deadline - 5:
        polished_points, polished_score = polish_points(best_points, max_time=min(45.0, deadline - time.time() - 3))
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_points.copy()
>>>>>>> REPLACE
