import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q): points [x:y:z], adjacency if x1*x2+y1*y2+z1*z2 == 0
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    assert len(points) == q*q + q + 1

    # Build adjacency matrix for polarity graph (C4-free, 90 edges)
    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                A[i, j] = 1
                A[j, i] = 1

    m = A.sum() // 2  # should be 90
    assert m == 90, f"Expected 90 edges, got {m}"

    # Find the edge that adds fewest C4s to the polarity graph
    # C4s added by edge (u,v) = number of common neighbors of u and v
    best_edge = None
    best_common = n + 1
    for u in range(n):
        for v in range(u+1, n):
            if A[u, v] == 0:
                common = np.sum(A[u] & A[v])
                if common < best_common:
                    best_common = common
                    best_edge = (u, v)

    A[best_edge[0], best_edge[1]] = 1
    A[best_edge[1], best_edge[0]] = 1
    m += 1
    assert m == 91

    # Count total C4s efficiently using matrix operations
    def count_total_c4(A):
        """Count total number of 4-cycles in graph with adjacency matrix A."""
        A2 = A @ A
        # Number of C4s = (1/8) * (sum(A2^2) - sum(A^4) - 4*sum(A^2) - trace(A2^2) + 4*trace(A2))
        # Simpler: for each pair (i,j) with i<j, C4s through them as opposite vertices
        # C4s = (1/2) * sum over non-adjacent pairs of C(common_neighbors, 2)
        # But even faster: total C4s = (1/8) * (tr(A^4) - 4*sum(A^2) - 2*sum(A^2) + ...) 
        # Let's use the direct formula:
        # Number of C4s = (1/8) * (sum_{i,j} (A^2)_{ij}^2 - sum_{i,j} (A^2)_{ij} - 4*sum_{i} (A^2)_{ii})
        # Actually: C4s = (1/8) * (||A^2||_F^2 - 2*sum(A^2) - 4*tr(A^2))
        # where sum(A^2) = sum of all entries of A^2 = sum of degrees squared - sum of degrees
        # Let me just use the pairwise method which is clear and fast enough for n=31
        
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 0:
                    common = np.sum(A[u] & A[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2

    total_c4 = count_total_c4(A)

    # Spectral-guided local search
    # Key insight: edges that contribute to large eigenvalues of A contribute more to C4s
    # We'll use eigenvalue structure to guide edge swaps

    def compute_eigenvalues(A):
        """Compute eigenvalues of adjacency matrix."""
        try:
            eigvals = np.linalg.eigvalsh(A.astype(np.float64))
            return eigvals
        except:
            return np.zeros(n)

    def spectral_score(A):
        """Compute spectral norm and eigenvalue spread as a proxy for C4 contribution."""
        eigvals = np.linalg.eigvalsh(A.astype(np.float64))
        return np.sum(eigvals**2)  # This equals sum of all entries of A^2

    # Precompute for efficiency
    def get_edges(A):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 1:
                    edges.append((u, v))
        return edges

    def get_nonedges(A):
        nonedges = []
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 0:
                    nonedges.append((u, v))
        return nonedges

    # Fast delta C4 computation
    def fast_delta_c4(A, ru, rv, au, av):
        """
        Compute delta in C4 count when removing edge (ru,rv) and adding edge (au,av).
        """
        # C4s destroyed by removing (ru, rv)
        destroyed = 0
        nu = np.where(A[ru] == 1)[0]
        nv = np.where(A[rv] == 1)[0]
        for a in nu:
            if a == rv:
                continue
            na = np.where(A[a] == 1)[0]
            for b in nv:
                if b == ru or b == a:
                    continue
                if A[a, b] == 1:
                    destroyed += 1
        
        # C4s created by adding (au, av) in graph without (ru, rv)
        A[ru, rv] = 0
        A[rv, ru] = 0
        
        created = 0
        nu = np.where(A[au] == 1)[0]
        nv = np.where(A[av] == 1)[0]
        for a in nu:
            if a == av:
                continue
            na = np.where(A[a] == 1)[0]
            for b in nv:
                if b == au or b == a:
                    continue
                if A[a, b] == 1:
                    created += 1
        
        # Restore (ru, rv)
        A[ru, rv] = 1
        A[rv, ru] = 1
        
        return created - destroyed

    # Spectral-guided edge scoring
    def score_edges_spectral(A, edges, nonedges):
        """Score edges and nonedges using spectral information."""
        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(A.astype(np.float64))
        spectral_norm = np.sum(eigvals**2)
        
        # For each edge, compute how much it contributes to spectral norm
        edge_scores = []
        for (u, v) in edges:
            # Temporarily remove edge
            A[u, v] = 0
            A[v, u] = 0
            eigvals_new = np.linalg.eigvalsh(A.astype(np.float64))
            spectral_new = np.sum(eigvals_new**2)
            # Restore
            A[u, v] = 1
            A[v, u] = 1
            # Higher reduction in spectral norm = edge is more "central" to C4 structure
            edge_scores.append((spectral_norm - spectral_new, (u, v)))
        
        nonedge_scores = []
        for (u, v) in nonedges:
            # Temporarily add edge
            A[u, v] = 1
            A[v, u] = 1
            eigvals_new = np.linalg.eigvalsh(A.astype(np.float64))
            spectral_new = np.sum(eigvals_new**2)
            # Restore
            A[u, v] = 0
            A[v, u] = 0
            # Lower increase in spectral norm = edge is less disruptive
            nonedge_scores.append((spectral_new - spectral_norm, (u, v)))
        
        return edge_scores, nonedge_scores

    # Local search with spectral guidance
    best_A = A.copy()
    best_total = total_c4
    no_improve_count = 0
    rng = np.random.default_rng(42)

    while time.time() < deadline - 5.0:
        improved = False
        
        edges = get_edges(A)
        nonedges = get_nonedges(A)
        
        if not edges or not nonedges:
            break
        
        # Compute spectral scores (this is expensive, so we do it every few iterations)
        if no_improve_count % 3 == 0 or no_improve_count == 0:
            edge_scores, nonedge_scores = score_edges_spectral(A, edges, nonedges)
            edge_scores.sort(key=lambda x: -x[0])  # descending by spectral contribution
            nonedge_scores.sort(key=lambda x: x[0])  # ascending by spectral disruption
        else:
            # Use C4-based scoring for speed
            edge_scores = []
            for (u, v) in edges:
                nu = np.where(A[u] == 1)[0]
                nv = np.where(A[v] == 1)[0]
                count = 0
                for a in nu:
                    if a == v:
                        continue
                    na = np.where(A[a] == 1)[0]
                    for b in nv:
                        if b == u or b == a:
                            continue
                        if A[a, b] == 1:
                            count += 1
                edge_scores.append((count, (u, v)))
            
            nonedge_scores = []
            for (u, v) in nonedges:
                nu = np.where(A[u] == 1)[0]
                nv = np.where(A[v] == 1)[0]
                count = 0
                for a in nu:
                    if a == v:
                        continue
                    na = np.where(A[a] == 1)[0]
                    for b in nv:
                        if b == u or b == a:
                            continue
                        if A[a, b] == 1:
                            count += 1
                nonedge_scores.append((count, (u, v)))
            
            edge_scores.sort(key=lambda x: -x[0])
            nonedge_scores.sort(key=lambda x: x[0])
        
        # Try top swaps
        for (ec, (ru, rv)) in edge_scores[:20]:
            if improved:
                break
            for (ac, (au, av)) in nonedge_scores[:20]:
                delta = fast_delta_c4(A, ru, rv, au, av)
                if delta < 0:
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                    total_c4 += delta
                    improved = True
                    break
        
        if not improved:
            # Simulated annealing step
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                re = edges[rng.integers(len(edges))]
                ae = nonedges[rng.integers(len(nonedges))]
                ru, rv = re
                au, av = ae
                delta = fast_delta_c4(A, ru, rv, au, av)
                T = max(1.0, 8.0 - (no_improve_count / 50.0))
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                    total_c4 += delta
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            if total_c4 < best_total:
                best_total = total_c4
                best_A = A.copy()
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                # Restart with perturbation
                for _ in range(3):
                    re = edges[rng.integers(len(edges))]
                    ae = nonedges[rng.integers(len(nonedges))]
                    ru, rv = re
                    au, av = ae
                    A[ru, rv] = 0
                    A[rv, ru] = 0
                    A[au, av] = 1
                    A[av, au] = 1
                total_c4 = count_total_c4(A)
                no_improve_count = 0

    # Return best solution
    final_edges = get_edges(best_A)
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    return [[e[0], e[1]] for e in final_edges]
