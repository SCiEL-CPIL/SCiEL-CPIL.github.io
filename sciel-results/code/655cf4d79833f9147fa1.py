import numpy as np
import time
from collections import deque

def _lift_connected_and_radius(signs, n, edges, m):
    """Return spectral radius if lift is connected 3-regular, else None."""
    N = 2 * n
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            pairs = [(2*u, 2*v), (2*u+1, 2*v+1)]
        else:
            pairs = [(2*u, 2*v+1), (2*u+1, 2*v)]
        for a, b in pairs:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * N
    seen[0] = True
    stack = [0]
    count = 1
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if not seen[y]:
                seen[y] = True
                count += 1
                stack.append(y)
    if count != N:
        return None
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def _spectral_radius_from_phases(phases, edges, n):
    """Compute spectral radius of signed adjacency given vertex phases."""
    A = np.zeros((n, n), dtype=float)
    for (u, v) in edges:
        s = 1.0 if np.cos(phases[u] - phases[v]) >= 0 else -1.0
        A[u, v] = s
        A[v, u] = s
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))

def _gradient_descent_on_phases(phases, edges, n, max_iter=500, lr=0.05):
    """Gradient descent on smooth surrogate: sum of squared eigenvalues."""
    phases = phases.copy()
    for it in range(max_iter):
        # Compute current signed adjacency
        A = np.zeros((n, n), dtype=float)
        for (u, v) in edges:
            cos_val = np.cos(phases[u] - phases[v])
            s = 1.0 if cos_val >= 0 else -1.0
            A[u, v] = s
            A[v, u] = s
        
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(A)
        # Objective: sum of squared eigenvalues = trace(A^2) = 2 * sum of edges
        # Gradient of trace(A^2) w.r.t. phases is more complex
        # Use finite differences for stability
        grad = np.zeros(n)
        eps = 1e-5
        for i in range(n):
            phases_plus = phases.copy()
            phases_plus[i] += eps
            r_plus = _spectral_radius_from_phases(phases_plus, edges, n)
            phases_minus = phases.copy()
            phases_minus[i] -= eps
            r_minus = _spectral_radius_from_phases(phases_minus, edges, n)
            grad[i] = (r_plus - r_minus) / (2 * eps)
        
        # Update phases
        phases -= lr * grad
        # Keep phases in [0, 2*pi)
        phases = phases % (2 * np.pi)
    
    return phases

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # safety margin under 60s (kill at 90s)

    n = graph['n']
    edges = graph['edges']
    m = n // 2

    # Build adjacency for spanning tree (BFS)
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    visited = [False] * n
    tree_edge_set = set()
    visited[0] = True
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_set.add(idx)
                dq.append(v)

    chords = [i for i in range(len(edges)) if i not in tree_edge_set]
    num_chords = len(chords)

    best_radius = float('inf')
    best_signs = None

    # Fallback valid assignment: mask with single -1 (guarantees connectivity
    # because product over the unique fundamental cycle containing that chord
    # is -1). Compute it first as a guaranteed-valid baseline.
    base_signs = np.ones(len(edges), dtype=int)
    base_signs[chords[0]] = -1
    r = _lift_connected_and_radius(base_signs, n, edges, m)
    if r is not None:
        best_radius = r
        best_signs = base_signs.copy()

    # Enumerate all 2^num_chords assignments
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
        signs = np.ones(len(edges), dtype=int)
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
        r = _lift_connected_and_radius(signs, n, edges, m)
        if r is None:
            continue
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Phase-based local search to escape local minima
    if time.time() < deadline - 5:
        # Try multiple random initializations of phases
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            
            # Initialize phases randomly
            phases = rng.uniform(0, 2 * np.pi, size=n)
            
            # Run gradient descent
            phases_opt = _gradient_descent_on_phases(phases, edges, n, max_iter=200, lr=0.05)
            
            # Convert phases to signs
            signs = np.ones(len(edges), dtype=int)
            for i, (u, v) in enumerate(edges):
                if np.cos(phases_opt[u] - phases_opt[v]) < 0:
                    signs[i] = -1
            
            r = _lift_connected_and_radius(signs, n, edges, m)
            if r is not None and r < best_radius:
                best_radius = r
                best_signs = signs.copy()

    # Safety: ensure we always return something valid
    if best_signs is None:
        best_signs = base_signs.copy()

    return best_signs
