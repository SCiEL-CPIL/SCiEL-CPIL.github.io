"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def _build_structure(graph):
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Identify edge types: 0=bottom cycle, 1=top cycle, 2=matching
    edge_types = []
    for (u, v) in edges:
        if u < m and v < m:
            edge_types.append(0)  # bottom
        elif u >= m and v >= m:
            edge_types.append(1)  # top
        else:
            edge_types.append(2)  # matching
    
    edge_types = np.array(edge_types)
    
    # Build incidence for gradient computation
    # For each edge, store (u, v, type)
    # We need to compute dR/dphase for each vertex
    
    # Adjacency structure for efficient matrix building
    # For each vertex, list of (edge_index, other_vertex, sign_in_matrix)
    # sign_in_matrix is +1 or -1 depending on how the edge contributes
    
    # Actually, let's build the signed adjacency matrix directly
    # A[u,v] = sign if edge exists
    
    return m, edges, edge_types

def _build_signed_adj(m, edges, signs):
    """Build signed adjacency matrix."""
    n = 2 * m
    A = np.zeros((n, n))
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(m, edges, signs):
    """Compute spectral radius of signed adjacency matrix."""
    A = _build_signed_adj(m, edges, signs)
    # Use eigvalsh since A is symmetric
    try:
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    except:
        return 100.0

def _check_validity(m, edges, signs):
    """Check if the lift is connected and 3-regular."""
    # The lift is always 3-regular by construction (each vertex in lift has degree 3)
    # We just need to check connectivity
    # Build the lift graph
    n_lift = 4 * m
    # Vertex (v, s) in lift, where v in 0..2m-1, s in {0,1}
    # Edge (u, v) with sign s connects:
    #   if s == +1: (u,0)-(v,0) and (u,1)-(v,1)
    #   if s == -1: (u,0)-(v,1) and (u,1)-(v,0)
    
    # Use union-find for connectivity
    parent = list(range(n_lift))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        if s == 1:  # +1
            union(2*u, 2*v)      # (u,0)-(v,0)
            union(2*u+1, 2*v+1)  # (u,1)-(v,1)
        else:  # -1
            union(2*u, 2*v+1)    # (u,0)-(v,1)
            union(2*u+1, 2*v)    # (u,1)-(v,0)
    
    # Check all vertices connected to vertex 0
    root = find(0)
    for i in range(1, n_lift):
        if find(i) != root:
            return False
    return True

def _phase_to_signs(phases, m, edges):
    """Convert continuous phases to discrete signs."""
    signs = np.zeros(len(edges), dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        # sign = sign(Re(z_u * conj(z_v))) = sign(cos(theta_u - theta_v))
        diff = phases[u] - phases[v]
        c = np.cos(diff)
        if c >= 0:
            signs[idx] = 1
        else:
            signs[idx] = -1
    return signs

def _gradient_spectral_radius(phases, m, edges, signs):
    """
    Compute gradient of spectral radius w.r.t. phases.
    Uses power iteration to get the dominant eigenvector, then computes gradient.
    """
    A = _build_signed_adj(m, edges, signs)
    n = 2 * m
    
    # Power iteration to get dominant eigenvector
    x = np.ones(n) / np.sqrt(n)
    for _ in range(100):
        Ax = A @ x
        norm = np.linalg.norm(Ax)
        if norm < 1e-15:
            break
        x = Ax / norm
    
    # Spectral radius
    rho = abs(x @ (A @ x))
    
    # Gradient: d(rho)/d(A_ij) = (x_i * x_j + x_j * x_i) / 2 * sign(rho)
    # Actually for symmetric matrix, d(rho)/d(A_ij) = x_i * x_j for the top eigenvalue
    # More precisely: if lambda is the largest eigenvalue with eigenvector x (normalized),
    # then d(lambda)/d(A_ij) = x_i * x_j
    
    # For signed adjacency, A[u,v] = sign_{uv}
    # d(sign_{uv})/d(theta_u) = -sign_{uv} * sin(theta_u - theta_v) * (1/cos(theta_u - theta_v))
    #                          = -sign_{uv} * tan(theta_u - theta_v)
    # d(sign_{uv})/d(theta_v) = +sign_{uv} * tan(theta_u - theta_v)
    
    grad = np.zeros(2 * m)
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        diff = phases[u] - phases[v]
        c = np.cos(diff)
        if abs(c) < 1e-6:
            # Near zero crossing, gradient is large; use subgradient
            grad[u] += -s * x[u] * x[v] * np.sign(np.sin(diff)) * 100
            grad[v] += s * x[u] * x[v] * np.sign(np.sin(diff)) * 100
        else:
            tan_val = np.sin(diff) / c
            d_s_d_u = -s * tan_val
            d_s_d_v = s * tan_val
            grad[u] += d_s_d_u * x[u] * x[v]
            grad[v] += d_s_d_v * x[u] * x[v]
    
    return rho, grad

def _local_search(m, edges, signs, max_iters=5000):
    """Local search by flipping individual signs."""
    current_signs = signs.copy()
    current_radius = _spectral_radius(m, edges, current_signs)
    
    if not _check_validity(m, edges, current_signs):
        return None, float('inf')
    
    for _ in range(max_iters):
        improved = False
        for idx in range(len(edges)):
            trial = current_signs.copy()
            trial[idx] = -trial[idx]
            if not _check_validity(m, edges, trial):
                continue
            r = _spectral_radius(m, edges, trial)
            if r < current_radius - 1e-10:
                current_signs = trial
                current_radius = r
                improved = True
                break
        if not improved:
            break
    
    return current_signs, current_radius

def sign_edges(graph, budget):
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    deadline = time.time() + 85  # 5 second safety margin
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase-based continuous optimization
    # Initialize phases randomly
    rng = np.random.default_rng(3)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        phases = rng.uniform(0, 2*np.pi, size=2*m)
        
        # Gradient descent on continuous phases
        lr = 0.1
        for step in range(500):
            if time.time() > deadline:
                break
            
            signs = _phase_to_signs(phases, m, edges)
            rho, grad = _gradient_spectral_radius(phases, m, edges, signs)
            
            # Gradient step
            phases -= lr * grad
            phases = phases % (2 * np.pi)
            
            # Reduce learning rate
            lr *= 0.995
        
        # Final sign assignment from phases
        signs = _phase_to_signs(phases, m, edges)
        
        # Check validity and evaluate
        if _check_validity(m, edges, signs):
            r = _spectral_radius(m, edges, signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        
        # Local search refinement
        if best_signs is not None and time.time() < deadline:
            refined, ref_radius = _local_search(m, edges, best_signs, max_iters=2000)
            if refined is not None and ref_radius < best_radius:
                best_radius = ref_radius
                best_signs = refined
    
    # If no valid solution found, try random search
    if best_signs is None:
        for _ in range(1000):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            if _check_validity(m, edges, signs):
                r = _spectral_radius(m, edges, signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs
    
    # Final local search if we have a solution
    if best_signs is not None and time.time() < deadline - 2:
        refined, ref_radius = _local_search(m, edges, best_signs, max_iters=3000)
        if refined is not None and ref_radius < best_radius:
            best_radius = ref_radius
            best_signs = refined
    
    if best_signs is None:
        # Fallback: all +1 signs (should be valid for prism graphs)
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
# EVOLVE-BLOCK-END
