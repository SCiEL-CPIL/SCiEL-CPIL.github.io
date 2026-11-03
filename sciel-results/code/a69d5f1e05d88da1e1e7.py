    if best_pts is not None:
        best_pts = _project_to_triangle(best_pts)
        # Final safety check
        if not _in_triangle(best_pts):
             # Force all to centroid if something went wrong
             best_pts = np.full((N, 2), [0.5, S3H/3])
    
    return best_pts
