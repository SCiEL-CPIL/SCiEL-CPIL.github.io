"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 210.0  # 210s of 230s wall-clock

    half = size // 2  # 16

    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant."""
        N = size
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    def compute_eps_inv_fft(cell):
        """Compute FFT of 1/epsilon for the pixel map."""
        N = size
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        return np.fft.fft2(eps_inv) / (N * N)

    def fast_tm_bands(fft_coeff, max_g=2, num_kpts=12):
        """Fast TM band evaluation with small plane-wave basis.
        Returns gap-midgap ratio estimate.
        """
        N = size
        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        
        def get_coeff(mx, my):
            mx_w = mx % N
            my_w = my % N
            return fft_coeff[my_w, mx_w]
        
        G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
        
        # Precompute eps_inv matrix
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_inv_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)
        
        # K-points in irreducible BZ: 0 <= kx <= ky, kx + ky <= pi
        k_points = []
        n_per = 4
        for t in np.linspace(0, math.pi, n_per):
            k_points.append((0.0, t))
        for t in np.linspace(0, math.pi/2, n_per):
            k_points.append((t, t))
        for t in np.linspace(0, math.pi/2, n_per):
            k_points.append((math.pi/2 - t, math.pi/2 + t))
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = np.zeros(num_g)
            for q in range(num_g):
                kG_sq[q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2
            
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    def medium_tm_bands(fft_coeff, max_g=4, num_kpts=15):
        """Medium-accuracy TM band evaluation for refinement."""
        N = size
        g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
        num_g = len(g_indices)
        
        def get_coeff(mx, my):
            mx_w = mx % N
            my_w = my % N
            return fft_coeff[my_w, mx_w]
        
        G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
        
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                eps_inv_mat[p, q] = get_coeff(mx_p - mx_q, my_p - my_q)
        
        k_points = []
        n_per = 5
        for t in np.linspace(0, math.pi, n_per):
            k_points.append((0.0, t))
        for t in np.linspace(0, math.pi/2, n_per):
            k_points.append((t, t))
        for t in np.linspace(0, math.pi/2, n_per):
            k_points.append((math.pi/2 - t, math.pi/2 + t))
        k_points.append((math.pi/6, math.pi/3))
        k_points.append((math.pi/4, math.pi/2))
        k_points.append((math.pi/8, 3*math.pi/8))
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = np.zeros(num_g)
            for q in range(num_g):
                kG_sq[q] = (kx + G_vecs[q, 0])**2 + (ky + G_vecs[q, 1])**2
            
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # --- Initial designs ---
    def make_ring_quadrant(r_inner, r_outer):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_disk_quadrant(radius):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                r = math.sqrt(x*x + y*y)
                if r <= radius:
                    q[i][j] = 1
        return q

    def make_cross_quadrant(width):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / size
                y = (i + 0.5) / size
                if x < width or y < width:
                    q[i][j] = 1
        return q

    def random_quadrant(fill_prob=0.4):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                if np.random.random() < fill_prob:
                    q[i][j] = 1
        return q

    # Generate diverse initial population
    np.random.seed(42)
    population = []
    
    # Structured initial designs
    for ri, ro in [(0.15, 0.35), (0.2, 0.4), (0.1, 0.3), (0.25, 0.45), (0.3, 0.5)]:
        population.append(make_ring_quadrant(ri, ro))
    for r in [0.15, 0.2, 0.25, 0.3, 0.35]:
        population.append(make_disk_quadrant(r))
    for w in [0.08, 0.1, 0.12, 0.15]:
        population.append(make_cross_quadrant(w))
    
    # Random designs with various fill fractions
    for fp in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for _ in range(3):
            population.append(random_quadrant(fp))
    
    # Evaluate population with fast solver
    def evaluate_quadrant(q):
        cell = build_cell_from_quadrant(q)
        fft_coeff = compute_eps_inv_fft(cell)
        return fast_tm_bands(fft_coeff)
    
    scored_population = []
    for q in population:
        score = evaluate_quadrant(q)
        scored_population.append((score, q))
    
    scored_population.sort(key=lambda x: -x[0])
    
    # --- Evolutionary algorithm ---
    best_score = scored_population[0][0]
    best_quadrant = [row[:] for row in scored_population[0][1]]
    
    # Keep top K for breeding
    top_k = 15
    elites = [q[:] for _, q in scored_population[:top_k]]
    
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    
    generation = 0
    max_generations = 200
    pop_size = 30
    
    while generation < max_generations:
        if time.time() > deadline - 40:
            break
        
        generation += 1
        
        # Create new population
        new_population = []
        
        # Keep some elites
        for q in elites[:5]:
            new_population.append([row[:] for row in q])
        
        # Mutate and crossover
        while len(new_population) < pop_size:
            # Pick parent
            if np.random.random() < 0.7:
                # Mutate a random elite
                parent = np.random.choice(elites[:top_k])
                child = [row[:] for row in parent]
                # Flip 1-3 random pixels
                n_flips = np.random.randint(1, 4)
                for _ in range(n_flips):
                    i, j = positions[np.random.randint(len(positions))]
                    child[i][j] = 1 - child[i][j]
            else:
                # Crossover two elites
                p1 = np.random.choice(elites[:top_k])
                p2 = np.random.choice(elites[:top_k])
                child = [row[:] for row in p1]
                for i in range(half):
                    for j in range(i, half):
                        if np.random.random() < 0.5:
                            child[i][j] = p2[i][j]
                # Add small mutation
                if np.random.random() < 0.3:
                    i, j = positions[np.random.randint(len(positions))]
                    child[i][j] = 1 - child[i][j]
            
            new_population.append(child)
        
        # Evaluate new population
        for q in new_population:
            score = evaluate_quadrant(q)
            if score > best_score:
                best_score = score
                best_quadrant = [row[:] for row in q]
            scored_population.append((score, q))
        
        # Update elites
        scored_population.sort(key=lambda x: -x[0])
        elites = [q[:] for _, q in scored_population[:top_k]]
        
        # Keep only recent scored entries to prevent memory growth
        scored_population = scored_population[:pop_size * 3]
    
    # --- Final refinement with medium-accuracy solver ---
    if time.time() < deadline - 30:
        # Refine best design with local search using medium solver
        current_q = [row[:] for row in best_quadrant]
        current_score = medium_tm_bands(compute_eps_inv_fft(build_cell_from_quadrant(current_q)))
        
        if current_score > best_score:
            best_score = current_score
        
        no_improve = 0
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            if no_improve > 20:
                break
            
            improved = False
            # Try flipping each pixel position
            for (i, j) in positions:
                if time.time() > deadline - 15:
                    break
                
                old_val = current_q[i][j]
                current_q[i][j] = 1 - old_val
                
                cell = build_cell_from_quadrant(current_q)
                fft_coeff = compute_eps_inv_fft(cell)
                score = medium_tm_bands(fft_coeff)
                
                if score > current_score:
                    current_score = score
                    best_score = max(best_score, current_score)
                    best_quadrant = [row[:] for row in current_q]
                    improved = True
                    no_improve = 0
                else:
                    current_q[i][j] = old_val
                    no_improve += 1
            
            if not improved:
                break
    
    # Build final cell
    final_cell = build_cell_from_quadrant(best_quadrant)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert final_cell[i][j] == final_cell[size-1-i][j], f"Vertical symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[i][size-1-j], f"Horizontal symmetry broken at ({i},{j})"
            assert final_cell[i][j] == final_cell[j][i], f"Transpose symmetry broken at ({i},{j})"
    
    return final_cell
# EVOLVE-BLOCK-END
