import time
import numpy as np
from itertools import product as iproduct

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    def get_edge_types():
        """Categorize edges into top cycle, bottom cycle, and matching"""
        top_cycle = []
        bottom_cycle = []
        matching = []
        
        for idx, (u, v) in enumerate(edges):
            if u < m and v < m:
                top_cycle.append(idx)
            elif u >= m and v >= m:
                bottom_cycle.append(idx)
            else:
                matching.append(idx)
        
        return top_cycle, bottom_cycle, matching
    
    top_cycle, bottom_cycle, matching = get_edge_types()
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        """
        Fast connectivity check using cycle products.
        The 2-lift is connected iff the product of signs around the
        fundamental cycles involving matching edges is -1.
        
        For a prism graph, the fundamental cycles that must have odd
        number of -1 signs are those going from i (top) around top cycle
        to j, then matching to j+m, then bottom cycle back to i+m, then
        matching back to i.
        
        Equivalent: the product of all signs along any such cycle must be -1.
        We check the cycle: top edge i->i+1, matching (i+1, i+1+m), 
        bottom edge (i+1+m)->(i+m), matching (i+m, i).
        Product = s_top[i] * s_match[i+1] * s_bottom[i] * s_match[i]
        This must be -1 for connectivity.
        
        Actually, let's use a simpler characterization:
        The lift is connected iff there exists at least one "twist" 
        in the matching pattern relative to the cycles.
        
        More precisely: define the "winding" around the top cycle as
        the product of top signs. The lift is connected iff this product
        combined with matching signs creates an odd cycle in the lift.
        
        Simplest correct check: BFS on the 2n-vertex lift.
        """
        # Build lift adjacency
        lift_adj = [set() for _ in range(2*n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u].add(v)
                lift_adj[v].add(u)
                lift_adj[u + n].add(v + n)
                lift_adj[v + n].add(u + n)
            else:
                lift_adj[u].add(v + n)
                lift_adj[v + n].add(u)
                lift_adj[u + n].add(v)
                lift_adj[v].add(u + n)
        
        visited = [False] * (2*n)
        queue = [0]
        visited[0] = True
        count = 1
        head = 0
        while head < len(queue):
            curr = queue[head]
            head += 1
            for neighbor in lift_adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == 2 * n
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    # Precompute edge index mapping for fast lookup
    edge_idx_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_idx_map[(u, v)] = idx
        edge_idx_map[(v, u)] = idx
    
    # Get top cycle edges in order: (0,1), (1,2), ..., (m-1, 0)
    top_cycle_edges = []
    for i in range(m):
        j = (i + 1) % m
        key = (i, j) if (i, j) in edge_idx_map else (j, i)
        top_cycle_edges.append(edge_idx_map[key])
    
    # Get bottom cycle edges in order: (m, m+1), (m+1, m+2), ..., (2m-1, m)
    bottom_cycle_edges = []
    for i in range(m):
        j = (i + 1) % m
        u = m + i
        v = m + j
        key = (u, v) if (u, v) in edge_idx_map else (v, u)
        bottom_cycle_edges.append(edge_idx_map[key])
    
    # Get matching edges: (0, m), (1, m+1), ..., (m-1, 2m-1)
    matching_edges = []
    for i in range(m):
        u = i
        v = m + i
        key = (u, v) if (u, v) in edge_idx_map else (v, u)
        matching_edges.append(edge_idx_map[key])
    
    def fast_is_connected(signs):
        """
        Fast connectivity check using cycle product.
        The 2-lift is connected iff the product of signs around the
        fundamental cycle (go around top, cross matching, go back around bottom,
        cross matching back) is -1.
        
        Cycle: 0 -> 1 -> ... -> m-1 -> m-1+m -> m-2+m -> ... -> 0+m -> 0
        Product = (prod of top signs) * s_match[m-1] * (prod of bottom signs) * s_match[0]
        
        But actually we need to check ALL fundamental cycles. For a prism,
        it suffices to check one "ladder" cycle. The lift is connected iff
        the product around any single fundamental cycle that uses both 
        matchings is -1.
        
        Let's verify: the product around cycle i:
        top: i -> i+1, sign = signs[top_cycle_edges[i]]
        matching: (i+1) -> (i+1+m), sign = signs[matching_edges[i+1]]  (or i if wrapping)
        bottom: (i+1+m) -> (i+m), sign = signs[bottom_cycle_edges[i]] (reversed, same sign)
        matching: (i+m) -> i, sign = signs[matching_edges[i]]
        
        Product = signs[top_cycle_edges[i]] * signs[matching_edges[(i+1)%m]] * 
                  signs[bottom_cycle_edges[i]] * signs[matching_edges[i]]
        
        The lift is connected iff this product is -1 for at least one i.
        Actually, it must be -1 for ALL i (since all fundamental cycles 
        must be odd to keep the lift connected... wait, no).
        
        Let me think again. The 2-lift has 2n vertices. It's connected 
        iff the "winding number" is odd, which means the product of signs
        around any cycle that goes from one copy back to the other via
        both matchings must be -1.
        
        For the prism, all such fundamental cycles give the same product
        up to sign flips that cancel. Let's just check one.
        """
        # Check the cycle: 0 -> 1 -> ... -> m-1 -> (m-1)+m -> (m-2)+m -> ... -> 0+m -> 0
        prod = 1
        # Top cycle forward: 0 -> 1 -> ... -> m-1
        for i in range(m - 1):
            prod *= signs[top_cycle_edges[i]]
        # Matching from m-1 to (m-1)+m
        prod *= signs[matching_edges[m-1]]
        # Bottom cycle backward: (m-1)+m -> (m-2)+m -> ... -> 0+m
        for i in range(m - 1, 0, -1):
            prod *= signs[bottom_cycle_edges[i]]
        # Matching from 0+m to 0
        prod *= signs[matching_edges[0]]
        
        return prod == -1
    
    best_radius = float('inf')
    best_signs = None
    
    rng = np.random.default_rng(42)
    
    def random_valid_signing():
        for _ in range(100000):
            signs = rng.choice([-1, 1], size=num_edges)
            if fast_is_connected(signs):
                return signs
        # Fallback: force a valid signing
        signs = np.ones(num_edges, dtype=int)
        # Flip one matching edge to create the twist
        signs[matching_edges[0]] = -1
        return signs
    
    # Try SDP relaxation approach
    # The SDP for minimizing spectral radius of signed adjacency:
    # Minimize t subject to -t*I <= A(x) <= t*I, x_i in {-1,1}
    # Relax x_i^2 = 1 to X_ii = 1, X positive semidefinite.
    # 
    # For small graphs, we can solve this via scipy.
    # For the prism graph, the structure is highly symmetric, so we can
    # exploit the circulant structure.
    
    # Actually, let's use a different approach: since the graph is small
    # (m up to 9, so n up to 18, edges up to 27), we can use a smarter
    # search strategy.
    
    # Key insight: the optimal signing for a prism graph has a specific
    # structure. Let's enumerate the "type" of signing:
    # - Pattern of top cycle signs (up to rotation and reflection)
    # - Pattern of bottom cycle signs  
    # - Pattern of matching signs
    # 
    # But 2^27 is too large for full enumeration. Instead, use the SDP
    # relaxation to get a good starting point, then local search.
    
    # SDP relaxation via scipy
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
    
    # For the SDP, we minimize the largest eigenvalue of A(x) where
    # x_i are continuous in [-1,1] with the constraint that the lift
    # is connected (which we'll handle by checking after rounding).
    
    # Actually, a better approach for small m: use the fact that the
    # prism graph has a well-known optimal signing structure.
    
    # For m=4: 12 edges, 2^12 = 4096 possible signings
    # For m=5: 15 edges, 2^15 = 32768
    # For m=7: 21 edges, 2^21 = 2M (too many for exhaustive)
    # For m=9: 27 edges, 2^27 = 134M (too many)
    
    if m <= 5:
        # Exhaustive search for small m
        for trial in range(1 << num_edges):
            if time.time() > deadline - 5:
                break
            signs = np.array([(1 if (trial >> i) & 1 else -1) for i in range(num_edges)])
            if not fast_is_connected(signs):
                continue
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    else:
        # For larger m, use SDP-inspired initialization + local search
        
        # SDP relaxation: minimize max eigenvalue of A(x) where x_i in [-1,1]
        # We use gradient descent on the continuous relaxation
        
        def build_continuous_matrix(x):
            A = np.zeros((n, n))
            for i, (u, v) in enumerate(edges):
                s = x[i]
                A[u, v] = s
                A[v, u] = s
            return A
        
        def continuous_spectral_radius(x):
            A = build_continuous_matrix(x)
            evals = np.linalg.eigvalsh(A)
            return np.max(np.abs(evals))
        
        def gradient_spectral_radius(x):
            """Compute gradient using finite differences"""
            eps = 1e-4
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                fp = continuous_spectral_radius(x_plus)
                fm = continuous_spectral_radius(x_minus)
                grad[i] = (fp - fm) / (2 * eps)
            return grad
        
        # Multi-start continuous optimization
        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            if restart == 0:
                # Start from a random valid signing
                init_signs = random_valid_signing()
                x = init_signs.astype(float)
            elif restart == 1:
                # Start from all +1 except one -1 for connectivity
                x = np.ones(num_edges)
                x[matching_edges[0]] = -1.0
            elif restart == 2:
                # Start from alternating pattern
                x = np.ones(num_edges)
                for i in range(0, len(top_cycle_edges), 2):
                    x[top_cycle_edges[i]] = -1.0
            else:
                x = rng.uniform(-1, 1, size=num_edges)
            
            # Gradient descent with momentum
            velocity = np.zeros_like(x)
            lr = 0.1
            momentum = 0.9
            max_iter = 200
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                grad = gradient_spectral_radius(x)
                velocity = momentum * velocity - lr * grad
                x = x + velocity
                x = np.clip(x, -1.0, 1.0)
            
            # Round to discrete signs
            final_signs = np.where(x >= 0, 1, -1)
            
            if fast_is_connected(final_signs):
                r = spectral_radius(final_signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = final_signs.copy()
        
        # If we haven't found a valid signing yet, get one
        if best_signs is None:
            best_signs = random_valid_signing()
            best_radius = spectral_radius(best_signs)
        
        # Local search: flip single edges
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Simulated annealing
        T = 0.5
        cooling = 0.999
        min_T = 1e-6
        steps = 0
        
        while time.time() < deadline - 5 and T > min_T:
            steps += 1
            if steps % 10000 == 0 and time.time() > deadline - 5:
                break
            
            idx = rng.integers(0, num_edges)
            new_signs = current_signs.copy()
            new_signs[idx] *= -1
            
            if not fast_is_connected(new_signs):
                T *= cooling
                continue
            
            new_radius = spectral_radius(new_signs)
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_signs = new_signs
                current_radius = new_radius
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
            
            T *= cooling
        
        # Also try 2-edge flips
        if time.time() < deadline - 5:
            for _ in range(50000):
                if time.time() > deadline - 5:
                    break
                i = rng.integers(0, num_edges)
                j = rng.integers(0, num_edges)
                if i == j:
                    continue
                new_signs = best_signs.copy()
                new_signs[i] *= -1
                new_signs[j] *= -1
                
                if not fast_is_connected(new_signs):
                    continue
                
                r = spectral_radius(new_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = new_signs.copy()
    
    # Final polish: try flipping each edge individually
    if best_signs is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(num_edges):
                if time.time() > deadline - 2:
                    break
                new_signs = best_signs.copy()
                new_signs[idx] *= -1
                if not fast_is_connected(new_signs):
                    continue
                r = spectral_radius(new_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = new_signs
                    improved = True
    
    if best_signs is None:
        best_signs = random_valid_signing()
    
    return best_signs
# EVOLVE-BLOCK-END
