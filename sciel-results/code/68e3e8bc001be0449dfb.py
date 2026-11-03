<<<<<<< SEARCH
def _simulated_annealing(pts, time_limit=15.0):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_steps = 5000

    rng = np.random.RandomState(42)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        noise_scale = T * 0.1
        perturbation = rng.randn(n, 3) * noise_scale
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        if new_energy < current_energy or rng.rand() < math.exp(-(new_energy - current_energy) / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy
=======
def _pso_optimize(pts, time_limit=15.0, n_particles=8):
    """Particle Swarm Optimization on the sphere to escape local minima."""
    start_time = time.time()
    n = len(pts)
    rng = np.random.RandomState(123)
    
    # Initialize particles around the given points
    particles = []
    for i in range(n_particles):
        if i == 0:
            p = _project_to_sphere(pts.copy())
        else:
            # Random perturbation around initial points
            noise = rng.randn(n, 3) * 0.1
            p = _project_to_sphere(pts + noise)
        e, _ = _energy_and_grad(p)
        particles.append({'pos': p, 'best_pos': p.copy(), 'best_energy': e})
    
    # Global best
    global_best_idx = min(range(n_particles), key=lambda i: particles[i]['best_energy'])
    global_best_pos = particles[global_best_idx]['best_pos'].copy()
    global_best_energy = particles[global_best_idx]['best_energy']
    
    # Velocity (in tangent space)
    velocities = [np.zeros_like(p['pos']) for p in particles]
    
    # PSO parameters
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    w = 0.9   # Inertia weight
    max_vel = 0.05  # Max velocity magnitude
    
    step = 0
    max_steps = 2000
    
    while time.time() - start_time < time_limit and step < max_steps:
        step += 1
        
        for i in range(n_particles):
            if time.time() - start_time > time_limit:
                break
                
            # Compute gradient for this particle
            pos = particles[i]['pos']
            _, grad = _energy_and_grad(pos)
            grad_tangent = grad - np.sum(grad * pos, axis=1, keepdims=True) * pos
            
            # Update velocity (in tangent space)
            r1 = rng.rand(n, 3)
            r2 = rng.rand(n, 3)
            
            # Cognitive: towards particle's own best
            cognitive = (particles[i]['best_pos'] - pos)
            cognitive_tangent = cognitive - np.sum(cognitive * pos, axis=1, keepdims=True) * pos
            
            # Social: towards global best
            social = (global_best_pos - pos)
            social_tangent = social - np.sum(social * pos, axis=1, keepdims=True) * pos
            
            # New velocity = w * v + c1 * r1 * cognitive + c2 * r2 * social
            new_vel = (w * velocities[i] + 
                      c1 * r1 * cognitive_tangent + 
                      c2 * r2 * social_tangent)
            
            # Clamp velocity magnitude
            vel_norm = np.linalg.norm(new_vel, axis=1, keepdims=True)
            mask = vel_norm > max_vel
            if np.any(mask):
                new_vel[mask] = new_vel[mask] * (max_vel / vel_norm[mask])
            
            velocities[i] = new_vel
            
            # Update position
            new_pos = pos + velocities[i]
            new_pos = _project_to_sphere(new_pos)
            
            # Evaluate
            new_energy, _ = _energy_and_grad(new_pos)
            
            # Update particle best
            if new_energy < particles[i]['best_energy']:
                particles[i]['best_energy'] = new_energy
                particles[i]['best_pos'] = new_pos.copy()
                
                # Update global best
                if new_energy < global_best_energy:
                    global_best_energy = new_energy
                    global_best_pos = new_pos.copy()
            
            particles[i]['pos'] = new_pos
        
        # Decay inertia
        w = max(0.4, w * 0.995)
    
    return global_best_pos, global_best_energy
>>>>>>> REPLACE
