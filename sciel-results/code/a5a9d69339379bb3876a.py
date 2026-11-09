"""Sutton-Chen cluster global minima via coarse-to-fine annealing basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _truncate_by_radius(pts, n, center=None):
    if len(pts) <= n:
        while len(pts) < n:
            p = pts[np.random.default_rng(len(pts)).integers(0, len(pts))]
            p = p + np.random.default_rng(len(pts) * 7).normal(0, 0.1, 3)
            pts.append(p)
        return np.array(pts[:n])
    if center is None:
        center = np.mean(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_shells(n, r1=1.0, r2=1.27, r3=1.51):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * r1)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * r2
                pts.append(mid)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * r3
                    pts.append(fc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            v = verts[rng.integers(0, len(verts))]
            v2 = verts[rng.integers(0, len(verts))]
            p = (v + v2) / 2
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _decahedral_cluster(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    r = 1.0
    pts = []
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] - 0.6]) * r)
    pts.append(np.array([0, 0, -0.6]))
    for v in verts:
        pts.append(np.array([v[0], v[1], v[2] + 0.6]) * r)
    pts.append(np.array([0, 0, 0.6]))
    rng = np.random.default_rng(11)
    while len(pts) < n:
        p = rng.normal(0, 1.5, 3)
        pts.append(p)
    return _truncate_by_radius(np.array(pts), n)


def _random_compact(n, seed, min_dist=0.75):
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)


def construct(instance):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # Target parameters
    if kind == '12-6':
        target_n_exp, target_m_exp, target_c = 12, 6, 144.41
    elif kind == '9-6':
        target_n_exp, target_m_exp, target_c = 9, 6, 39.432
    else: # 10-8
        target_n_exp, target_m_exp, target_c = 10, 8, 34.408

    # --- Initial Seeds ---
    seeds = [
        _fcc_cluster(n, lattice=0.72),
        _hcp_cluster(n, lattice=0.72),
        _icosahedral_shells(n),
        _decahedral_cluster(n),
        _random_compact(n, seed=42, min_dist=0.75),
    ]

    # --- Coarse-to-Fine Annealing Strategy ---
    # We will run a basin-hopping-like process where we relax using a "softened" potential
    # and then gradually restore the true potential parameters.
    
    # Stage 1: High "Temperature" (Soft Potential)
    # Reduce exponents to flatten the landscape and allow large rearrangements.
    # Scale c to reduce the depth of the well, making it easier to escape.
    
    # Define annealing stages: (n_exp_scale, m_exp_scale, c_scale, temp_scale, n_steps)
    # We will use a simple linear ramp from soft to hard.
    
    # Initial state: very soft
    # Final state: target parameters
    
    # We will perform a sequence of relaxations.
    # To simulate "basin hopping" with the available `relax` (which is just L-BFGS-B),
    # we will:
    # 1. Relax with soft parameters.
    # 2. Perturb the result.
    # 3. Relax with slightly harder parameters.
    # 4. Repeat until target parameters are reached.
    
    # However, `relax` uses the *true* potential internally? 
    # The task says: `relax(positions, 'sutton_chen', kind)` runs L-BFGS-B.
    # It likely uses the standard parameters for `kind`. 
    # We CANNOT easily change the parameters inside `relax` unless we implement our own relaxation loop.
    
    # Let's check if we can implement a simple gradient descent or L-BFGS step ourselves 
    # using `sutton_chen_energy_gradient` which returns energy and gradient for the *true* potential.
    # If we can't change the potential inside the helper, we can't do "soft" relaxation with the helper.
    
    # Alternative Strategy: 
    # Since we can't change the potential in `relax`, we must rely on the fact that 
    # L-BFGS-B is a local minimizer. The "coarse-to-fine" idea might be implemented 
    # by perturbing the structure significantly between relaxations, effectively 
    # performing a random walk in configuration space, but guided by the fact that 
    # we always return to a local minimum of the *true* potential.
    
    # Wait, the inspiration says: "Use a coarse-to-fine annealing strategy where the potential is dynamically softened...".
    # This implies we need to implement the energy/gradient calculation ourselves with modified parameters 
    # to perform the "soft" relaxation, or use a different method.
    
    # Let's implement a custom relaxation function that allows us to specify the potential parameters.
    # We can use `scipy.optimize.minimize` with L-BFGS-B, providing our own energy and gradient functions 
    # that use the scaled parameters.
    
    from scipy.optimize import minimize
    
    def sc_energy_gradient(pos, n_exp, m_exp, c):
        """
        Compute energy and gradient for Sutton-Chen with specified parameters.
        E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)]
        """
        n = pos.shape[0]
        # Compute pairwise distances
        # r_ij = ||p_i - p_j||
        # For efficiency, we can compute the full distance matrix or use broadcasting.
        # For n up to ~80, full matrix is fine.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(r, np.inf)
        
        # Pairwise terms
        r_inv_n = r ** (-n_exp)
        r_inv_m = r ** (-m_exp)
        
        # Sum over j for each i
        # S_i = sum_{j!=i} r_ij^-m_exp
        S = np.sum(r_inv_m, axis=1)
        
        # Energy
        # E = sum_i [ 0.5 * sum_{j!=i} r_ij^-n_exp - c * sqrt(S_i) ]
        repulsive_sum = 0.5 * np.sum(r_inv_n, axis=1)
        attractive_sum = c * np.sqrt(S)
        energy = np.sum(repulsive_sum - attractive_sum)
        
        # Gradient
        # dE/dr_ij = -n_exp * r_ij^(-n_exp-1) - c * (1/(2*sqrt(S_i))) * (-m_exp * r_ij^(-m_exp-1))
        # Note: S_i depends on r_ij. dS_i/dr_ij = -m_exp * r_ij^(-m_exp-1)
        # d/d r_ij [ -c * sqrt(S_i) ] = -c * (1/(2*sqrt(S_i))) * dS_i/dr_ij
        # = -c * (1/(2*sqrt(S_i))) * (-m_exp * r_ij^(-m_exp-1))
        # = c * m_exp * r_ij^(-m_exp-1) / (2*sqrt(S_i))
        
        # So dE/dr_ij = -n_exp * r_ij^(-n_exp-1) + c * m_exp * r_ij^(-m_exp-1) / (2*sqrt(S_i))
        
        # Avoid division by zero in S
        S_safe = np.where(S > 1e-12, S, 1.0)
        
        dE_dr = -n_exp * r ** (-n_exp - 1) + c * m_exp * r ** (-m_exp - 1) / (2.0 * np.sqrt(S_safe))
        
        # Gradient with respect to position p_i
        # grad_i = sum_{j!=i} dE/dr_ij * (p_i - p_j) / r_ij
        # dE/dp_i = sum_j dE/dr_ij * (p_i - p_j) / r_ij
        
        # dE_dr is (n, n) matrix. (p_i - p_j) is (n, n, 3).
        # We need to multiply element-wise and sum over j.
        
        # Normalize the difference vector
        # diff / r
        # r has inf on diagonal, so diff/r is 0 on diagonal (0/inf=0)
        # But we must be careful with 0/0 if r is 0. We filled diagonal with inf, so r is never 0 off-diagonal?
        # If two atoms are at the same position, r=0. This causes issues.
        # We assume valid inputs (no overlaps < 0.35).
        
        unit_vec = diff / r[:, :, np.newaxis]
        
        # grad_i = sum_j dE_dr[i,j] * unit_vec[i,j]
        grad = np.sum(dE_dr[:, :, np.newaxis] * unit_vec, axis=1)
        
        return energy, grad

    def custom_relax(pos, n_exp, m_exp, c, max_iter=1000, tol=1e-8):
        """
        Relax using L-BFGS-B with custom potential parameters.
        """
        def fun(x):
            p = x.reshape(-1, 3)
            e, g = sc_energy_gradient(p, n_exp, m_exp, c)
            return e, g.flatten()
        
        res = minimize(fun, pos.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        
        p_final = res.x.reshape(-1, 3)
        e_final = res.fun
        return e_final, p_final

    # --- Annealing Schedule ---
    # We start with a very soft potential and ramp up to the target.
    # Stages:
    # 1. Very soft: n_exp*0.5, m_exp*0.5, c*0.5
    # 2. Medium soft: n_exp*0.75, m_exp*0.75, c*0.75
    # 3. Target: n_exp, m_exp, c
    
    # We will perform basin hopping within each stage by perturbing and re-relaxing.
    
    stages = [
        (0.5, 0.5, 0.5, 5, 0.5),  # scale_n, scale_m, scale_c, n_restarts, perturbation_size
        (0.75, 0.75, 0.75, 5, 0.3),
        (1.0, 1.0, 1.0, 10, 0.2),
    ]

    best_e = np.inf
    best_pos = None
    rng = np.random.default_rng(12345)

    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        current_pos = seed.copy()
        current_e = np.inf
        
        # Evaluate initial energy with true potential to track
        try:
            current_e, _ = sutton_chen_energy_gradient(current_pos, kind)
        except:
            current_e = np.inf

        for stage_idx, (sn, sm, sc, n_restarts, pert_size) in enumerate(stages):
            # Calculate scaled parameters
            cur_n_exp = target_n_exp * sn
            cur_m_exp = target_m_exp * sm
            cur_c = target_c * sc
            
            # For the final stage, we want the true energy.
            # For previous stages, we use the custom relaxer.
            
            # Perform n_restarts of relaxation + perturbation
            for r in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Relax with current stage parameters
                try:
                    if stage_idx == len(stages) - 1:
                        # Use the provided relax for the final stage (true potential)
                        e, p = relax(current_pos, potential, kind)
                    else:
                        # Use custom relax with soft parameters
                        e, p = custom_relax(current_pos, cur_n_exp, cur_m_exp, cur_c, max_iter=200)
                    
                    if np.isfinite(e) and _validate_positions(p):
                        # Check true energy to see if this is actually good
                        true_e, _ = sutton_chen_energy_gradient(p, kind)
                        
                        if true_e < best_e:
                            best_e = true_e
                            best_pos = p.copy()
                        
                        # Update current best for this lineage
                        if true_e < current_e:
                            current_e = true_e
                            current_pos = p.copy()
                        
                        # Perturb for next restart
                        if r < n_restarts - 1 and stage_idx < len(stages) - 1:
                            # Perturb more aggressively in softer stages
                            current_pos = current_pos + rng.normal(0, pert_size, size=(n, 3))
                            if not _validate_positions(current_pos):
                                # If perturbation causes overlap, revert to previous
                                current_pos = current_pos - rng.normal(0, pert_size, size=(n, 3))
                except Exception:
                    pass
            
            # End of stage, move to next (harder) stage
            # We keep the best position found in this stage as the starting point for the next

    # --- Final Polish with True Potential ---
    if best_pos is not None and time.time() < deadline - 5:
        # Use the provided relax for a final high-precision optimization
        try:
            e_final, p_final = relax(best_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                best_e = e_final
                best_pos = p_final
        except Exception:
            pass

    if best_pos is None:
        # Fallback
        best_pos = _random_compact(n, seed=999, min_dist=0.8)
        try:
            best_e, best_pos = relax(best_pos, potential, kind)
        except Exception:
            pass
        if not _validate_positions(best_pos):
            return _random_compact(n, seed=12345, min_dist=0.9)

    return best_pos
# EVOLVE-BLOCK-END
