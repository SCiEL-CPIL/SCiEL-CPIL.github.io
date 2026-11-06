"""Morse cluster global minima: Parallel Tempering (Replica Exchange)."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import time

def _generate_seeds(n, rho, num_seeds=6):
    """Generate diverse initial configurations."""
    seeds = []
    rng = np.random.default_rng(42)
    
    # 1. Random compact
    radius = n ** (1/3) * 0.8
    pts = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    seeds.append(np.array(pts))
    
    # 2. Icosahedral shell
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    ico_pts = [np.zeros(3)]
    if n > 1:
        ico_pts.extend(verts.tolist())
    # Add second shell if needed
    if n > 13:
        r2 = 2.0
        verts2 = verts * r2
        ico_pts.extend(verts2.tolist())
    # Add third shell if needed
    if n > 25:
        r3 = 2.5
        verts3 = verts * r3
        ico_pts.extend(verts3.tolist())
    # Fill rest randomly
    while len(ico_pts) < n:
        p = rng.uniform(-3, 3, 3)
        ico_pts.append(p)
    seeds.append(np.array(ico_pts[:n]))

    # 3. FCC-like lattice
    max_r = int(np.ceil(n ** (1.0/3.0) * 1.5)) + 2
    candidates = []
    for ix in range(-max_r, max_r + 1):
        for iy in range(-max_r, max_r + 1):
            for iz in range(-max_r, max_r + 1):
                d = np.sqrt(ix*ix + iy*iy + iz*iz)
                candidates.append((d, [ix, iy, iz]))
    candidates.sort(key=lambda x: x[0])
    fcc_pts = []
    for d, p in candidates:
        if len(fcc_pts) >= n:
            break
        fcc_pts.append(p)
    seeds.append(np.array(fcc_pts[:n]))

    # 4. BCC-like lattice (simple cubic with body centers approximated by denser SC)
    # Just use SC with slightly different spacing
    sc_pts = []
    for d, p in candidates:
        if len(sc_pts) >= n:
            break
        sc_pts.append([p[0]*0.95, p[1]*0.95, p[2]*0.95])
    seeds.append(np.array(sc_pts[:n]))

    # 5. Random on sphere shell
    shell_pts = []
    for _ in range(n):
        z = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2*np.pi)
        x = np.sqrt(1-z*z) * np.cos(theta)
        y = np.sqrt(1-z*z) * np.sin(theta)
        r = n ** (1/3) * 0.6 * (0.5 + 0.5*rng.random())
        shell_pts.append([x*r, y*r, z*r])
    seeds.append(np.array(shell_pts))

    # 6. Another random with different seed
    rng2 = np.random.default_rng(123)
    pts2 = []
    for _ in range(n):
        p = rng2.uniform(-radius, radius, 3)
        pts2.append(p)
    seeds.append(np.array(pts2))

    return seeds

def _morse_energy(pos, rho):
    """Calculate Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    e = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-6:
                return 1e10
            x = rho * (1.0 - r)
            ex = np.exp(x)
            e += ex * (ex - 2.0)
    return e

def _parallel_tempering(instance, time_budget):
    n = instance['n']
    rho = instance['rho']
    potential = instance['potential']
    
    t_start = time.time()
    
    # Generate initial configurations
    seeds = _generate_seeds(n, rho, 6)
    
    # Relax each seed initially to get a good starting point for the lowest T replica
    # and to ensure they are valid
    relaxed_seeds = []
    for s in seeds:
        try:
            e, p = relax(s.copy(), potential, rho)
            relaxed_seeds.append(p)
        except:
            relaxed_seeds.append(s)
            
    # Setup Parallel Tempering
    # We will use 4 replicas
    n_replicas = 4
    temps = [0.1, 0.5, 1.0, 2.0] # k_B * T
    # Actually, for Morse, energy scale is ~n. Let's scale T with n.
    # E_min ~ -n. So T should be comparable to energy differences.
    # Let's use T = [0.1, 0.3, 0.8, 1.5] * n ** 0.5 or similar.
    # Let's stick to absolute values for now and tune.
    # A good heuristic: T_k = T_min * (T_max/T_min)^(k/(N-1))
    T_min = 0.05
    T_max = 1.0
    temps = [T_min * (T_max/T_min)**(i/(n_replicas-1)) for i in range(n_replicas)]
    
    # Initialize replicas
    # Replica 0 gets the best relaxed seed
    # Others get shuffled seeds
    best_seed_idx = 0
    best_e = np.inf
    for i, s in enumerate(relaxed_seeds):
        e = _morse_energy(s, rho)
        if e < best_e:
            best_e = e
            best_seed_idx = i
            
    replicas = []
    energies = []
    for i in range(n_replicas):
        if i == 0:
            pos = relaxed_seeds[best_seed_idx].copy()
        else:
            pos = relaxed_seeds[(best_seed_idx + i) % len(relaxed_seeds)].copy()
        e = _morse_energy(pos, rho)
        replicas.append(pos)
        energies.append(e)
        
    step_sizes = [0.2] * n_replicas
    
    best_energy = np.inf
    best_pos = replicas[0].copy()
    
    max_steps = 2000
    swap_interval = 10
    
    for step in range(max_steps):
        if time.time() - t_start > time_budget - 2.0:
            break
            
        for k in range(n_replicas):
            # Random move
            step_size = step_sizes[k]
            idx = np.random.randint(n)
            disp = np.random.normal(0, step_size, 3)
            new_pos = replicas[k].copy()
            new_pos[idx] += disp
            
            new_e = _morse_energy(new_pos, rho)
            
            # Metropolis acceptance
            if new_e < energies[k] or np.random.random() < np.exp(-(new_e - energies[k]) / temps[k]):
                replicas[k] = new_pos
                energies[k] = new_e
                step_sizes[k] *= 1.01
                if step_sizes[k] > 1.0:
                    step_sizes[k] = 1.0
            else:
                step_sizes[k] *= 0.98
                if step_sizes[k] < 0.01:
                    step_sizes[k] = 0.01
            
            # Update global best
            if energies[k] < best_energy:
                best_energy = energies[k]
                best_pos = replicas[k].copy()
        
        # Swap replicas
        if step % swap_interval == 0:
            for k in range(n_replicas - 1):
                # Try to swap k and k+1
                dE = energies[k] / temps[k+1] - energies[k+1] / temps[k]
                if dE < 0 or np.random.random() < np.exp(-dE):
                    # Swap
                    replicas[k], replicas[k+1] = replicas[k+1], replicas[k]
                    energies[k], energies[k+1] = energies[k+1], energies[k]
                    step_sizes[k], step_sizes[k+1] = step_sizes[k+1], step_sizes[k]
        
        # Relax the best replica occasionally to help it settle
        if step % 100 == 0 and step > 0:
            try:
                e_rel, p_rel = relax(replicas[0].copy(), potential, rho)
                if e_rel < energies[0]:
                    replicas[0] = p_rel
                    energies[0] = e_rel
                    if e_rel < best_energy:
                        best_energy = e_rel
                        best_pos = p_rel.copy()
            except:
                pass

    # Final relaxation of the best found
    try:
        e_final, p_final = relax(best_pos.copy(), potential, rho)
        return p_final
    except:
        return best_pos

def construct(instance):
    """Find low-energy Morse cluster via Parallel Tempering."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    # Time budget per instance: ~45s (300s / 6 instances with margin)
    time_per_instance = 45.0
    
    return _parallel_tempering(instance, time_per_instance)
# EVOLVE-BLOCK-END
