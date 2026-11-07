"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    deadline = time.time() + 55.0
    
    # Identify edge types
    # Prism graph: two m-cycles (0..m-1, m..2m-1) + matching (i, i+m)
    cycle1_edges = []
    cycle2_edges = []
    matching_edges = []
    
    # We need to map edge indices to positions in the cycles to exploit symmetry/structure
    # Assuming standard prism graph edge ordering or we just classify by endpoints
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            cycle1_edges.append(idx)
        elif u >= m and v >= m:
            cycle2_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-compute adjacency structure for fast radius calculation
    # The signed adjacency matrix A_s has entries A_s[u,v] = sign(e) * A[u,v]
    # For a prism graph, the spectrum is related to the eigenvalues of the "twisted" cycles.
    # However, to be safe and general for m=4,5,7,9, we will build the matrix once 
    # and update it via vectorized operations or simple numpy slicing for evaluation.
    
    # Build the base adjacency matrix (unweighted, just 1/0)
    # But we need the signed version.
    # Let's build a list of (u, v) for each edge index.
    edge_map = edges
    
    def compute_radius_fast(signs):
        """
        Compute spectral radius of the signed adjacency matrix.
        signs: array of shape (num_edges,) with values -1, 1
        Returns: spectral radius (max abs eigenvalue)
        """
        # Construct signed adjacency matrix
        # Since n is small (max 18), constructing the matrix is fast.
        # But we can do better.
        # A_s is symmetric.
        # A_s[i,j] = sign_k if edge k connects i,j.
        
        # For n <= 18, np.zeros((n,n)) and filling is ~18*18 = 324 ops.
        # eigvalsh on 18x18 is very fast.
        
        A = np.zeros((n, n), dtype=np.float64)
        for k in range(num_edges):
            u, v = edge_map[k]
            s = signs[k]
            A[u, v] += s
            A[v, u] += s
        
        # Check connectivity implicitly? 
        # The problem states: "The lift must be connected... or the whole result is invalid."
        # check_lift raises on invalid. We should try to avoid invalid lifts if possible, 
        # or we can just let check_lift validate at the end for the best candidate.
        # For the search, we can compute radius for *any* sign vector (even if lift is disconnected, 
        # the spectral radius is still defined for the base signed graph? 
        # Wait, "These are the new eigenvalues of the lift."
        # If the lift is disconnected, is the radius defined? Yes, max abs eigenvalue of the lift adjacency.
        # But the task says "or the whole result is invalid". 
        # So we must return a valid sign vector.
        # We will compute radius for all candidates, keep track of the best VALID one.
        # How to check validity quickly?
        # A 2-lift of a graph is connected iff the "monodromy" (product of signs around cycles) 
        # allows reaching both sides. 
        # Actually, it's easier to just use check_lift for the final candidate.
        # During search, we can compute radius of the signed base graph? 
        # No, the spectral radius of the LIFT is what matters.
        # The eigenvalues of the lift are the eigenvalues of the signed adjacency matrix of the base graph?
        # Let's verify this theory.
        # For a 2-lift defined by signs, the adjacency matrix of the lift is a block matrix:
        # [[A, B], [B, A]] if signs are +1 for matching? No.
        # Standard 2-lift: vertices (v,0), (v,1).
        # Edge (u,v) with sign s.
        # If s=+1: (u,0)-(v,0) and (u,1)-(v,1).
        # If s=-1: (u,0)-(v,1) and (u,1)-(v,0).
        # The adjacency matrix of the lift is:
        # A_lift = [[A_pos, A_neg], [A_neg, A_pos]] ? No.
        # Let A+ be adjacency of edges with sign +1, A- be adjacency of edges with sign -1.
        # Then A_lift = [[A+, A-], [A-, A+]].
        # The eigenvalues of this block matrix are eigenvalues of A+ + A- and A+ - A-.
        # A+ + A- = A (unweighted adjacency).
        # A+ - A- = A_signed (signed adjacency where edge weight is +1 or -1).
        # So the spectrum of the lift is the union of spectra of A and A_signed.
        # Since A is the adjacency of the prism graph (unweighted), its spectrum is fixed!
        # The variable part is the spectrum of A_signed.
        # The spectral radius of the lift is max( radius(A), radius(A_signed) ).
        # radius(A) for prism graph P_m x K_2 is fixed.
        # For m=4, radius(A) = 3 (3-regular).
        # For any m, radius(A) = 3.
        # So we just need to minimize radius(A_signed) !
        # And we must ensure the lift is connected.
        
        # This is a HUGE insight.
        # We don't need to build the 2n x 2n lift matrix.
        # We just need the spectral radius of the n x n signed adjacency matrix.
        
        # Let's verify: "mean over the four graphs of the spectral radius of the signed adjacency matrix of the base graph"
        # The task description says: "Scoring ... mean ... spectral radius of the signed adjacency matrix of the base graph"
        # Wait, the text says: "These are the new eigenvalues of the lift."
        # And "Scoring ... spectral radius of the signed adjacency matrix of the base graph".
        # This confirms it. The score is the spectral radius of A_signed.
        # We don't even need to check connectivity for the SCORE, but we MUST return a valid sign vector (connected lift).
        
        # So, objective: Minimize spectral_radius(A_signed).
        # Constraint: The 2-lift defined by signs must be connected.
        
        # How to check connectivity of the 2-lift?
        # The lift is disconnected iff there is a subset of vertices S in the base graph such that 
        # the "twist" around any cycle in S is +1? 
        # Actually, the 2-lift is connected iff the subgroup of Z_2 generated by the monodromies of the cycles is {0, 1} (i.e., not trivial).
        # More simply: The lift is disconnected iff there exists a 2-coloring of the base graph vertices such that 
        # every edge connects vertices of the same color? No, that's bipartiteness.
        # The lift is disconnected iff the signs are "coboundary", i.e., there exists a function f: V -> {0,1} such that 
        # sign(e) = (-1)^(f(u)+f(v)) for all edges (u,v).
        # In that case, the lift splits into two copies of the base graph.
        # So, Invalid Lift <=> Signs are a coboundary.
        # Valid Lift <=> Signs are NOT a coboundary.
        
        # This allows us to check validity in O(n+m) without graph traversal on the lift!
        
        return np.max(np.abs(np.linalg.eigvalsh(A)))

    def is_valid_lift(signs):
        """
        Returns True if the lift is connected.
        Lift is disconnected iff signs are a coboundary.
        i.e., exists f: V -> {0,1} s.t. sign(u,v) = (-1)^(f(u)+f(v)).
        This is equivalent to: The system of linear equations over GF(2) has a solution.
        Let f_v be 0 or 1.
        sign(u,v) = +1 => f_u + f_v = 0 (mod 2) => f_u = f_v
        sign(u,v) = -1 => f_u + f_v = 1 (mod 2) => f_u != f_v
        
        This is a 2-SAT / Bipartite Coloring problem on the graph where edges are constraints.
        If the graph is bipartite with respect to the signs, it's a coboundary.
        Wait, "bipartite with respect to signs" means we can assign 0/1 to vertices such that 
        adjacent vertices have same parity if sign=+1, different if sign=-1.
        This is always solvable if there are no odd cycles with product -1?
        No, it's just checking if the system is consistent.
        We can do BFS/DFS on the base graph.
        """
        # Build adjacency list with constraints
        # constraint(u, v, s): f_u + f_v = 0 if s=1, 1 if s=-1.
        # Let val[v] be the assigned value (0 or 1) or -1 if unassigned.
        val = np.full(n, -1, dtype=int)
        
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            # target_diff = 0 if s==1 else 1
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        for i in range(n):
            if val[i] != -1:
                continue
            val[i] = 0
            stack = [i]
            while stack:
                u = stack.pop()
                for v, s in adj[u]:
                    # f_u + f_v = 0 if s=1, 1 if s=-1
                    # f_v = f_u + (0 if s==1 else 1) mod 2
                    expected_v = (val[u] + (0 if s == 1 else 1)) % 2
                    if val[v] == -1:
                        val[v] = expected_v
                        stack.append(v)
                    elif val[v] != expected_v:
                        return False # Inconsistent, so NOT a coboundary, so VALID lift
        return True # Consistent, so IS a coboundary, so INVALID lift (disconnected)

    # Wait, if it returns True, it's a coboundary -> Disconnected -> Invalid.
    # So we want is_coboundary(signs) == False.
    
    def is_valid(signs):
        return not is_coboundary(signs)

    # Let's implement is_coboundary properly
    def is_coboundary(signs):
        val = np.full(n, -1, dtype=int)
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            adj[u].append((v, s))
            adj[v].append((u, s))
            
        for i in range(n):
            if val[i] != -1:
                continue
            val[i] = 0
            stack = [i]
            while stack:
                u = stack.pop()
                for v, s in adj[u]:
                    expected_v = (val[u] + (0 if s == 1 else 1)) % 2
                    if val[v] == -1:
                        val[v] = expected_v
                        stack.append(v)
                    elif val[v] != expected_v:
                        return False
        return True

    def try_signs(signs):
        nonlocal best_signs, best_radius
        if time.time() > deadline:
            return
        # Check validity first (cheap O(n+m))
        if is_coboundary(signs):
            return
        
        # Compute radius (expensive O(n^3) but n is small)
        try:
            r = compute_radius_fast(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        except Exception:
            pass

    # Strategy:
    # 1. Generate structured candidates (Fourier modes)
    # 2. Local search with fast radius evaluation
    
    candidates = []
    
    # All +1 (Coboundary? f=0 works. Yes, disconnected. Invalid.)
    # All -1 (Coboundary? f_v = v%2? If m is even, cycle is even, product of -1 is 1. 
    # If m is odd, product is -1. 
    # For m=4,5,7,9.
    # m=4: All -1. Cycle 1-2-3-4-1. Product (-1)^4 = 1. Coboundary? 
    # f0=0, f1=1, f2=0, f3=1. f0!=f3? 0!=1. Contradiction? 
    # Edge (3,0) sign -1. f3=1, f0=0. 1+0=1. OK.
    # So All -1 is a coboundary for even m? 
    # Let's just test with is_coboundary.
    
    candidates.append(np.ones(num_edges, dtype=int))
    candidates.append(-np.ones(num_edges, dtype=int))
    
    # Fourier modes
    for k in range(1, m):
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            signs = np.ones(num_edges, dtype=int)
            for j, idx in enumerate(cycle1_edges):
                val = np.cos(2 * np.pi * k * j / m + phase)
                signs[idx] = 1 if val >= 0 else -1
            for j, idx in enumerate(cycle2_edges):
                val = np.cos(2 * np.pi * k * j / m + phase)
                signs[idx] = 1 if val >= 0 else -1
            # Matching edges: try both
            candidates.append(signs.copy())
            signs_match_neg = signs.copy()
            for idx in matching_edges:
                signs_match_neg[idx] = -1
            candidates.append(signs_match_neg)
            
            # Also try alternating matching signs
            signs_match_alt = signs.copy()
            for j, idx in enumerate(matching_edges):
                signs_match_alt[idx] = 1 if j % 2 == 0 else -1
            candidates.append(signs_match_alt)

    # Try all candidates
    for signs in candidates:
        if time.time() > deadline:
            break
        try_signs(signs)
    
    # If no valid candidate found, fallback to random search
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            try_signs(signs)
            
    if best_signs is None:
        # Last resort: try to fix a random one
        rng = np.random.default_rng(42)
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            if not is_coboundary(signs):
                try_signs(signs)
                break

    if best_signs is None:
        # Emergency fallback: All +1 is invalid. 
        # We must return something. 
        # Let's return a known valid one if we find one, else all +1 (which might fail validation in evaluator).
        # But we must return a valid one.
        # Let's force a valid one by flipping one edge of a coboundary.
        base = np.ones(num_edges, dtype=int)
        # Flip edge 0
        base[0] = -1
        # Check if valid
        if not is_coboundary(base):
            best_signs = base
        else:
            # Flip edge 1
            base[1] = -1
            if not is_coboundary(base):
                best_signs = base
            else:
                # Just return all +1 and hope for the best? No, task says invalid is 0 score.
                # We should have found something.
                best_signs = base

    # Local Search from best
    if time.time() < deadline - 5:
        current = best_signs.copy()
        current_radius = best_radius
        
        # Iterate flips
        for _ in range(50): # 50 iterations of full pass
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 2:
                    break
                trial = current.copy()
                trial[i] = -trial[i]
                if is_coboundary(trial):
                    continue
                try:
                    r = compute_radius_fast(trial)
                    if r < current_radius - 1e-12:
                        current = trial
                        current_radius = r
                        improved = True
                        break
                except Exception:
                    pass
            if not improved:
                break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
