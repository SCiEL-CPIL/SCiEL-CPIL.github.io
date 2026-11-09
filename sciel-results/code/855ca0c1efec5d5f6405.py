"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def fast_1flip_local_search(labels, max_passes=50):
        """
        1-flip local search: try flipping single vertices.
        Returns (best_labels, best_cut).
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)

            for v in order:
                if time.time() > deadline - 2.0:
                    break
                # Compute delta for flipping v
                same_count = 0
                opp_count = 0
                for u in neighbors[v]:
                    if labels[v] == labels[u]:
                        same_count += 1
                    else:
                        opp_count += 1
                delta = same_count - opp_count
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()

            if not improved:
                break

        return best_local_labels, best_local_cut

    def fast_2swap_local_search(labels, max_passes=30):
        """
        2-swap local search: try flipping pairs of vertices simultaneously.
        Returns (best_labels, best_cut).
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            random.shuffle(pairs)

            for (i, j) in pairs:
                if time.time() > deadline - 2.0:
                    break

                li = labels[i]
                lj = labels[j]
                nli = 1 - li
                nlj = 1 - lj

                delta = 0
                for u in neighbors[i]:
                    if u == j:
                        continue
                    if labels[i] == labels[u]:
                        delta += 1
                    else:
                        delta -= 1

                for u in neighbors[j]:
                    if u == i:
                        continue
                    if labels[j] == labels[u]:
                        delta += 1
                    else:
                        delta -= 1

                if delta > 0:
                    labels[i] = nli
                    labels[j] = nlj
                    current_cut += delta
                    improved = True
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()

            if not improved:
                break

        return best_local_labels, best_local_cut

    def admm_max_cut(rho=1.0, max_iters=200, tol=1e-6):
        """
        ADMM for Maximum Cut.
        
        Formulation: maximize sum_{i<j} x_i * x_j * A_ij
        where x_i in {-1, +1}
        
        ADMM formulation:
        minimize -x^T A x / 2 + indicator_{x in {-1,+1}^n}
        
        Split: minimize f(x) + g(z) subject to x = z
        f(x) = -x^T A x / 2
        g(z) = 0 if z in {-1,+1}^n, inf otherwise
        
        Augmented Lagrangian:
        L = -x^T A x / 2 + rho/2 * ||x - z + u||^2 - rho/2 * ||u||^2
        
        x-update: minimize -x^T A x / 2 + rho/2 * ||x - z + u||^2
        => (rho * I - A) x = rho * (z - u)
        => x = (rho * I - A)^{-1} * rho * (z - u)
        
        z-update: z = sign(x + u)
        
        u-update: u = u + x - z
        """
        # Precompute matrix for x-update: (rho*I - A)
        I = np.eye(n)
        A_float = adj.astype(np.float64)
        M = rho * I - A_float
        
        # Try to factorize once if possible (M should be SPD for large enough rho)
        try:
            from scipy.linalg import cho_factor, cho_solve
            L_factor, lower = cho_factor(M)
            use_cholesky = True
        except:
            use_cholesky = False
            try:
                M_inv = np.linalg.inv(M)
            except:
                M_inv = None
        
        # Initialize
        z = np.random.choice([-1.0, 1.0], size=n)
        u = np.zeros(n)
        
        best_z = z.copy()
        best_cut_val = -1
        
        prev_obj = None
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # x-update: solve (rho*I - A) x = rho * (z - u)
            rhs = rho * (z - u)
            if use_cholesky:
                x = cho_solve((L_factor, lower), rhs)
            elif M_inv is not None:
                x = M_inv @ rhs
            else:
                # Fallback: use numpy solve
                try:
                    x = np.linalg.solve(M, rhs)
                except:
                    x = z.copy()  # fallback
            
            # z-update: z = sign(x + u)
            z_new = np.sign(x + u)
            # Handle zeros (shouldn't happen often)
            z_new[z_new == 0] = 1.0
            
            # Compute cut value for current z
            z_int = z_new.astype(np.int8)
            cut_val = compute_cut(z_int)
            if cut_val > best_cut_val:
                best_cut_val = cut_val
                best_z = z_new.copy()
            
            # Check convergence
            if prev_obj is not None:
                primal_res = np.linalg.norm(x - z_new)
                dual_res = np.linalg.norm(z - z_new) * rho
                if primal_res < tol and dual_res < tol:
                    break
            prev_obj = cut_val
            
            # u-update: u = u + x - z
            u = u + x - z_new
            z = z_new
        
        return best_z.astype(np.int8), best_cut_val

    def spectral_init():
        """
        Spectral initialization using Fiedler vector of Laplacian.
        Returns 0/1 label array.
        """
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            labels = (fiedler > 0).astype(np.int8)
            if random.random() < 0.5:
                labels = 1 - labels
            return labels
        except:
            return np.random.randint(0, 2, size=n).astype(np.int8)

    def random_init():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    # Track best
    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Phase 1: ADMM with different rho values
    rho_values = [0.5, 1.0, 2.0, 5.0]
    for rho in rho_values:
        if time.time() > deadline - 30.0:
            break
        
        # Multiple restarts for each rho
        for restart in range(3):
            if time.time() > deadline - 25.0:
                break
            
            admm_labels, admm_cut = admm_max_cut(rho=rho, max_iters=150, tol=1e-5)
            
            # Polish with local search
            if time.time() < deadline - 10.0:
                polished, cut = fast_1flip_local_search(admm_labels, max_passes=30)
                try_labels(polished)
                polished2, cut2 = fast_2swap_local_search(polished, max_passes=15)
                try_labels(polished2)

    # Phase 2: Spectral init + local search (keep as backup)
    for _ in range(3):
        if time.time() > deadline - 15.0:
            break
        labels = spectral_init()
        polished, cut = fast_1flip_local_search(labels, max_passes=20)
        try_labels(polished)
        polished2, cut2 = fast_2swap_local_search(polished, max_passes=10)
        try_labels(polished2)

    # Phase 3: Final intensive polish on best found
    if best_labels is not None and time.time() < deadline - 1.0:
        final = best_labels.copy()
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            final, c1 = fast_1flip_local_search(final, max_passes=50)
            final, c2 = fast_2swap_local_search(final, max_passes=30)
        try_labels(final)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
