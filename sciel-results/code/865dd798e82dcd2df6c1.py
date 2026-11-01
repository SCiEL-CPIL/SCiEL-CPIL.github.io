<<<<<<< SEARCH
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        pts, f = _polish(pts, deadline, max_iter=300)
        best_x = pts.flatten()
        best_f = f
=======
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        pts = best_x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        
        # First pass: RCG for faster convergence
        pts, f = _rcg_polish(pts, deadline, max_iter=500)
        
        # Second pass: fine-tune with gradient descent if time remains
        if time.time() < deadline - 2.0:
            pts, f = _polish(pts, deadline, max_iter=200)
        
        best_x = pts.flatten()
        best_f = f
>>>>>>> REPLACE
