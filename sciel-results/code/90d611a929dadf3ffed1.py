import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        calls = 0
        deadline = time.time() + 290

        def evaluate(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Generate 32 deterministic Sobol-like points in [-5, 5]^dim
        # For dim=5, we can use a simple Halton or Sobol sequence.
        # Since dim is small, we can just hardcode a good set of points or use a simple generator.
        # Let's use a simple deterministic set of points that covers the space well.
        # Actually, for a linear function, any full-rank set of points works.
        # Let's use the vertices of the hypercube? No, that's 32 points for dim=5.
        # Wait, 2^5 = 32. We have budget 500.
        # Evaluating all 32 vertices is a very robust way to find the minimum of a linear function.
        # The minimum of a linear function on a convex polytope is at a vertex.
        # So if we evaluate all 32 vertices, we are guaranteed to find the global minimum.
        # This uses 32 calls.
        
        # Let's generate all 2^dim vertices.
        # For dim=5, that's 32 points.
        # For dim=10, that's 1024 points, which might exceed budget if budget is small, but budget is 500.
        # The task says dim=5. But let's be safe.
        # If 2^dim > budget, we fall back to gradient estimation.
        
        if (1 << dim) <= self.budget:
            # Evaluate all vertices
            vertices = np.array([
                np.array([v[i] for i in range(dim)])
                for v in range(1 << dim)
            ])
            # Map binary representation to [-5, 5]
            # 0 -> -5, 1 -> 5
            vertices = vertices * 10.0 - 5.0
            
            best_x = None
            best_f = np.inf
            
            for v in vertices:
                f_v = evaluate(v)
                if f_v < best_f:
                    best_f = f_v
                    best_x = v.copy()
                    
            # We have used 2^dim calls.
            # We can use remaining calls to verify or just return.
            # Since the minimum is guaranteed to be at a vertex, we are done.
            # However, to be extremely safe against any potential non-linearity (though task says linear),
            # we could do a few local checks, but for a linear function, the vertex is exact.
            
            return best_x, best_f
        else:
            # Fallback to gradient estimation if dim is too large for vertex enumeration
            # (Not expected for dim=5, budget=500)
            x0 = np.zeros(dim)
            f0 = evaluate(x0)
            best_x = x0.copy()
            best_f = f0
            
            step = 5.0
            signs = np.zeros(dim)
            for i in range(dim):
                xi = np.zeros(dim)
                xi[i] = step
                fi = evaluate(xi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()
                    
                if fi < f0 - 1e-12:
                    signs[i] = 1.0
                elif fi > f0 + 1e-12:
                    signs[i] = -1.0
                else:
                    xi_small = np.zeros(dim)
                    xi_small[i] = 0.1
                    fi_small = evaluate(xi_small)
                    if fi_small < f0:
                        signs[i] = 1.0
                    else:
                        signs[i] = -1.0
                        
            x_vertex = np.where(signs == 1.0, 5.0, -5.0)
            f_vertex = evaluate(x_vertex)
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
                
            return best_x, best_f
