import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    J = np.eye(N) - np.ones((N, N)) / N
    
    def edm_to_gram(d_vec):
        D_mat = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                D_mat[i, j] = d_vec[idx]
                D_mat[j, i] = d_vec[idx]
                idx += 1
        G = -0.5 * (J @ D_mat @ J)
        return D_mat, G

    def edm_objective(d_vec):
        D_mat, G = edm_to_gram(d_vec)
        
        off_diag = D_mat[~np.eye(N, dtype=bool)]
        if np.any(off_diag <= 1e-6):
            return 1e10
        
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]
        
        neg_eigs = eigvals[eigvals < 0]
        pen_psd = np.sum(neg_eigs**2) * 1000.0
        
        rank_pen = np.sum(eigvals[3:]**2) * 100.0
        
        min_d = np.min(off_diag)
        max_d = np.max(off_diag)
        ratio = max_d / min_d
        
        return ratio + pen_psd + rank_pen

    def extract_coords_from_edm(d_vec):
        D_mat, G = edm_to_gram(d_vec)
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        if len(eigvals) < 3:
            return None
        V = eigvecs[:, :3]
        L = np.sqrt(np.maximum(eigvals[:3], 0))
        X_centered = V @ np.diag(L)
        return X_centered

    def coord_ratio(X):
        dists_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists_sq = dists_sq[mask]
        min_d_sq = np.min(dists_sq)
        max_d_sq = np.max(dists_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    inits_coords = []
    inits_edms = []
    
    # 1. Fibonacci Sphere
    golden_angle = np.pi * (3 - np.sqrt(5))
    X_fib = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        X_fib[i, 0] = r * np.cos(theta)
        X_fib[i, 1] = r * np.sin(theta)
        X_fib[i, 2] = z
    X_fib = X_fib - np.mean(X_fib, axis=0)
    inits_coords.append(X_fib)
    D_fib = np.sum((X_fib[:, np.newaxis, :] - X_fib[np.newaxis, :, :])**2, axis=2)
    inits_edms.append(D_fib[~np.eye(N, dtype=bool)])

    # 2. Icosahedron + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rand_pts = np.random.randn(2, D)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    X_ico = np.vstack([ico_verts, rand_pts])
    X_ico = X_ico - np.mean(X_ico, axis=0)
    inits_coords.append(X_ico)
    D_ico = np.sum((X_ico[:, np.newaxis, :] - X_ico[np.newaxis, :, :])**2, axis=2)
    inits_edms.append(D_ico[~np.eye(N, dtype=bool)])

    # 3. Random
    for _ in range(5):
        X_rand = np.random.randn(N, D)
        X_rand = X_rand - np.mean(X_rand, axis=0)
        inits_coords.append(X_rand)
        D_rand = np.sum((X_rand[:, np.newaxis, :] - X_rand[np.newaxis, :, :])**2, axis=2)
        inits_edms.append(D_rand[~np.eye(N, dtype=bool)])

    # Find best initial coordinate ratio
    best_coord_ratio = float('inf')
    best_coord_X = inits_coords[0]
    for X in inits_coords:
        r = coord_ratio(X)
        if r < best_coord_ratio:
            best_coord_ratio = r
            best_coord_X = X

    # Optimize EDMs
    best_edm_ratio = float('inf')
    best_edm_X = None
    
    for d0 in inits_edms:
        try:
            res = minimize(edm_objective, d0, method='Powell', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Check if the result is a valid EDM (low penalties)
            # We need to recompute penalties to check validity
            D_mat, G = edm_to_gram(res.x)
            eigvals = np.linalg.eigvalsh(G)
            eigvals = np.sort(eigvals)[::-1]
            neg_eigs = eigvals[eigvals < 0]
            pen_psd = np.sum(neg_eigs**2)
            rank_pen = np.sum(eigvals[3:]**2)
            
            # If penalties are small, it's a valid 3D configuration
            if pen_psd < 1e-4 and rank_pen < 1e-4:
                X_candidate = extract_coords_from_edm(res.x)
                if X_candidate is not None:
                    r_cand = coord_ratio(X_candidate)
                    if r_cand < best_edm_ratio:
                        best_edm_ratio = r_cand
                        best_edm_X = X_candidate
        except Exception:
            continue

    # Decide which result to return
    if best_edm_X is not None and best_edm_ratio < best_coord_ratio:
        return best_edm_X
    else:
        return best_coord_X

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def construct():
    J = np.eye(N) - np.ones((N, N)) / N
    
    def edm_to_gram(d_vec):
        D_mat = np.zeros((N, N))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                D_mat[i, j] = d_vec[idx]
                D_mat[j, i] = d_vec[idx]
                idx += 1
        G = -0.5 * (J @ D_mat @ J)
        return D_mat, G

    def edm_objective(d_vec):
        D_mat, G = edm_to_gram(d_vec)
        
        off_diag = D_mat[~np.eye(N, dtype=bool)]
        if np.any(off_diag <= 1e-6):
            return 1e10
        
        eigvals = np.linalg.eigvalsh(G)
        eigvals = np.sort(eigvals)[::-1]
        
        neg_eigs = eigvals[eigvals < 0]
        pen_psd = np.sum(neg_eigs**2) * 1000.0
        
        rank_pen = np.sum(eigvals[3:]**2) * 100.0
        
        min_d = np.min(off_diag)
        max_d = np.max(off_diag)
        ratio = max_d / min_d
        
        return ratio + pen_psd + rank_pen

    def extract_coords_from_edm(d_vec):
        D_mat, G = edm_to_gram(d_vec)
        eigvals, eigvecs = np.linalg.eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        if len(eigvals) < 3:
            return None
        V = eigvecs[:, :3]
        L = np.sqrt(np.maximum(eigvals[:3], 0))
        X_centered = V @ np.diag(L)
        return X_centered

    def coord_ratio(X):
        dists_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists_sq = dists_sq[mask]
        min_d_sq = np.min(dists_sq)
        max_d_sq = np.max(dists_sq)
        if min_d_sq < 1e-12:
            return 1e10
        return max_d_sq / min_d_sq

    inits_coords = []
    inits_edms = []
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    X_fib = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        X_fib[i, 0] = r * np.cos(theta)
        X_fib[i, 1] = r * np.sin(theta)
        X_fib[i, 2] = z
    X_fib = X_fib - np.mean(X_fib, axis=0)
    inits_coords.append(X_fib)
    D_fib = np.sum((X_fib[:, np.newaxis, :] - X_fib[np.newaxis, :, :])**2, axis=2)
    inits_edms.append(D_fib[~np.eye(N, dtype=bool)])

    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    rand_pts = np.random.randn(2, D)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    X_ico = np.vstack([ico_verts, rand_pts])
    X_ico = X_ico - np.mean(X_ico, axis=0)
    inits_coords.append(X_ico)
    D_ico = np.sum((X_ico[:, np.newaxis, :] - X_ico[np.newaxis, :, :])**2, axis=2)
    inits_edms.append(D_ico[~np.eye(N, dtype=bool)])

    for _ in range(5):
        X_rand = np.random.randn(N, D)
        X_rand = X_rand - np.mean(X_rand, axis=0)
        inits_coords.append(X_rand)
        D_rand = np.sum((X_rand[:, np.newaxis, :] - X_rand[np.newaxis, :, :])**2, axis=2)
        inits_edms.append(D_rand[~np.eye(N, dtype=bool)])

    best_coord_ratio = float('inf')
    best_coord_X = inits_coords[0]
    for X in inits_coords:
        r = coord_ratio(X)
        if r < best_coord_ratio:
            best_coord_ratio = r
            best_coord_X = X

    best_edm_ratio = float('inf')
    best_edm_X = None
    
    for d0 in inits_edms:
        try:
            res = minimize(edm_objective, d0, method='Powell', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            D_mat, G = edm_to_gram(res.x)
            eigvals = np.linalg.eigvalsh(G)
            eigvals = np.sort(eigvals)[::-1]
            neg_eigs = eigvals[eigvals < 0]
            pen_psd = np.sum(neg_eigs**2)
            rank_pen = np.sum(eigvals[3:]**2)
            
            if pen_psd < 1e-4 and rank_pen < 1e-4:
                X_candidate = extract_coords_from_edm(res.x)
                if X_candidate is not None:
                    r_cand = coord_ratio(X_candidate)
                    if r_cand < best_edm_ratio:
                        best_edm_ratio = r_cand
                        best_edm_X = X_candidate
        except Exception:
            continue

    if best_edm_X is not None and best_edm_ratio < best_coord_ratio:
        return best_edm_X
    else:
        return best_coord_X

# EVOLVE-BLOCK-END
