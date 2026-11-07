"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: rapid vertex identification for AOCC-optimized scoring."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        evaluated_vertices = set()

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        def time_ok():
            return time.time() < deadline

        def vertex_key(v):
            return tuple(int(1 if xi > 0 else 0) for xi in v)

        # Step 1: Evaluate at origin (call 1)
        origin = np.zeros(dim)
        f_origin = safe_call(origin)
        if f_origin is None:
            rng = np.random.default_rng(42)
            for _ in range(budget - calls_made):
                if not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        best_f = f_origin
        best_x = origin.copy()

        # Step 2: Quick gradient estimate - probe one direction to get gradient info
        # For linear f(x) = g^T x + c, probing x = +5*d gives g^T(5d) + c
        # Compare with origin: difference = 5 * g^T d
        # Use d = (1,1,...,1) to get sum of gradient components
        # Then use d = (1,-1,1,-1,...) to get alternating sum
        # From two probes we can estimate all gradient signs
        
        # Probe 1: x = (5, 5, ..., 5)
        v1 = np.full(dim, 5.0)
        f1 = safe_call(v1)
        if f1 is not None:
            evaluated_vertices.add(vertex_key(v1))
            if f1 < best_f:
                best_f = f1
                best_x = v1.copy()
        
        # Probe 2: x = (-5, -5, ..., -5)
        v2 = np.full(dim, -5.0)
        f2 = safe_call(v2)
        if f2 is not None:
            evaluated_vertices.add(vertex_key(v2))
            if f2 < best_f:
                best_f = f2
                best_x = v2.copy()
        
        # Probe 3: x = (5, -5, 5, -5, ...) - alternating
        v3 = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(dim)])
        f3 = safe_call(v3)
        if f3 is not None:
            evaluated_vertices.add(vertex_key(v3))
            if f3 < best_f:
                best_f = f3
                best_x = v3.copy()
        
        # Probe 4: x = (-5, 5, -5, 5, ...) - alternating opposite
        v4 = np.array([-5.0 if i % 2 == 0 else 5.0 for i in range(dim)])
        f4 = safe_call(v4)
        if f4 is not None:
            evaluated_vertices.add(vertex_key(v4))
            if f4 < best_f:
                best_f = f4
                best_x = v4.copy()

        # Step 3: Use gradient estimation from probes to identify likely optimal vertex
        # For linear f, the minimum is at a vertex where each coordinate is
        # -5 if g_i > 0, +5 if g_i < 0
        # We can estimate gradient components from vertex evaluations
        
        # Let's try to estimate gradient from the 4 corner probes
        # f(5e) - f(0) = 5 * sum(g_i)
        # f(-5e) - f(0) = -5 * sum(g_i)
        # f(alt1) - f(0) = 5*(g1 - g2 + g3 - g4 + g5)
        # f(alt2) - f(0) = -5*(g1 - g2 + g3 - g4 + g5)
        
        # Better approach: probe individual axis directions
        # But that's 10 more calls. Instead, use the 4 corner values to
        # narrow down which vertex is best, then verify.
        
        # Step 4: Systematic vertex search - check all 32 vertices
        # Priority: vertices close to current best in Hamming distance
        if calls_made < budget and time_ok():
            # Generate all 32 vertices
            all_vertices = []
            for mask in range(1 << dim):
                v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
                all_vertices.append((v, mask))
            
            # Sort by Hamming distance to current best
            best_key = vertex_key(best_x)
            all_vertices.sort(key=lambda x: sum(1 for i in range(dim) 
                                                if ((x[1] >> i) & 1) != best_key[i]))
            
            for v, mask in all_vertices:
                if calls_made >= budget or not time_ok():
                    break
                vk = vertex_key(v)
                if vk in evaluated_vertices:
                    continue
                evaluated_vertices.add(vk)
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
