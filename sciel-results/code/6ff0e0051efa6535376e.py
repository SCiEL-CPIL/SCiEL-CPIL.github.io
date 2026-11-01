    def _backtrack_line_search(x, p, f, grad_f, alpha=0.01, beta=0.5, max_iter=50):
        # x: current point
        # p: direction (tangent or euclidean)
        # f: energy
        # grad_f: gradient (Euclidean or Riemannian? Must match p's space)
        # For Riemannian: p is tangent, grad_f is Riemannian grad.
        # Move: exp_x(t*p)
        # For Cartesian: p is euclidean, grad_f is Euclidean grad.
        # Move: proj(x + t*p)
        
        # We need to pass a 'step' function.
        pass
    