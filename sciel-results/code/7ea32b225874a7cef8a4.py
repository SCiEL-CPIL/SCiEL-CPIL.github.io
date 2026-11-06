"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing TM band-4/5 gap
    using a pixel-level genetic algorithm with FFT-accelerated PWE surrogate.
    """
    start_time = time.time()
    deadline = start_time + 210  # 210s safety margin (evaluator kills at 230s)
    
    size = problem["pixels"]
    half = size // 2
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Precompute k-points for the irreducible BZ (square lattice)
    # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 1 (units of pi/a)
    # Use a grid of points covering Gamma-M-X edge and interior
    def make_k_points(n_per_edge=6):
        """Generate k-points in the irreducible BZ wedge (0<=kx<=ky, kx^2+ky^2<=1)."""
        pts = []
        # Include high-symmetry points
        pts.append((0.0, 0.0))       # Gamma
        pts.append((0.5, 0.5))       # M
        pts.append((0.5, 0.0))       # X (M/2)
        # Grid points in the wedge
        for i in range(n_per_edge + 1):
            for j in range(n_per_edge + 1):
                kx = i / n_per_edge * 0.5
                ky = j / n_per_edge * 0.5
                if kx <= ky + 1e-9 and kx*kx + ky*ky <= 1.0 + 1e-9:
                    # Avoid duplicates
                    dup = False
                    for pkx, pky in pts:
                        if abs(pkx - kx) < 1e-9 and abs(pky - ky) < 1e-9:
                            dup = True
                            break
                    if not dup:
                        pts.append((kx, ky))
        return pts
    
    K_POINTS = make_k_points(n_per_edge=5)
    
    # FFT-accelerated PWE solver for TM mode
    def fast_tm_gap(cell_array, n_pw=15):
        """
        Compute TM band gap ratio using FFT-accelerated PWE.
        cell_array: 32x32 numpy array of 0/1
        Returns: gap ratio 2*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
        """
        N = size
        # Compute 1/epsilon field
        eps = np.where(cell_array == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Full 2D FFT of 1/epsilon gives all Fourier coefficients
        # F[m,n] = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-2pi i (m*i + n*j)/N)
        # np.fft.fft2 gives sum without the 1/N^2 factor
        F_full = np.fft.fft2(inv_eps) / (N * N)
        
        # We need Fourier coefficients f_{m,n} for |m|,|n| <= n_pw
        # F_full is indexed 0..N-1, where index k corresponds to frequency k if k<=N/2, k-N if k>N/2
        # Extract the needed coefficients
        f = np.zeros((2*n_pw+1, 2*n_pw+1), dtype=complex)
        for m in range(-n_pw, n_pw+1):
            for n in range(-n_pw, n_pw+1):
                # Map (m,n) to FFT index
                mi = m % N
                ni = n % N
                f[m+n_pw, n+n_pw] = F_full[mi, ni]
        
        # Precompute |k+p|^2 for all p in [-n_pw, n_pw]^2 for each k-point
        # This is shared across all k-points for a given p
        p_range = np.arange(-n_pw, n_pw+1)
        p_grid = np.meshgrid(p_range, p_range, indexing='ij')
        P2 = (p_grid[0]**2 + p_grid[1]**2)  # shape (2*n_pw+1, 2*n_pw+1)
        
        # For each k-point, build and solve the eigenvalue problem
        # TM PWE: eigenvalue problem is  A @ v = omega^2 * v
        # where A_{mn} = sum_p f_{m-p, n-q} * |k+p, k+q|^2
        # This is a 2D convolution: A = ifft2(fft2(f) * fft2(|k+p|^2)) restricted to window
        # But more directly: A is the matrix representation of the operator
        # (1/eps) * (-d^2/dx^2 - d^2/dy^2)
        
        # Efficient approach: for each k-point, build A using the convolution structure
        # A_{mn} = sum_p f_{m-p, n-q} * (kx+px)^2 + (ky+py)^2
        # This is a 2D convolution of f with the kernel K(p,q) = (kx+px)^2 + (ky+py)^2
        
        min_omega5 = float('inf')
        max_omega4 = 0.0
        n_modes = 2*n_pw + 1
        
        for kx, ky in K_POINTS:
            # Build kernel: K[px,py] = (kx+px)^2 + (ky+py)^2
            kx_arr = kx + p_range
            ky_arr = ky + p_range
            K = kx_arr[:, None]**2 + ky_arr[None, :]**2  # shape (n_modes, n_modes)
            
            # A = 2D circular convolution of f with K
            # A[m,n] = sum_{p,q} f[m-p, n-q] * K[p,q]
            # Using FFT: A = ifft2(fft2(f) * fft2(K))
            F_f = np.fft.fft2(f)
            F_K = np.fft.fft2(K)
            A = np.real(np.fft.ifft2(F_f * F_K))
            
            # A should be real and symmetric for TM (since f is Hermitian and K is real)
            # Symmetrize to remove numerical noise
            A = (A + A.T) / 2.0
            
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(A)
                eigvals = np.sort(eigvals)
                # Bands 4 and 5 (1-indexed) = indices 3 and 4 (0-indexed)
                if len(eigvals) >= 5:
                    omega4 = eigvals[3]
                    omega5 = eigvals[4]
                    if omega4 > 1e-10 and omega5 > 1e-10:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == float('inf') or max_omega4 < 1e-10:
            return 0.0
        
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap_ratio
    
    # C4v-symmetric operations
    def quadrant_to_cell(quadrant):
        """Convert 16x16 quadrant to full 32x32 cell with C4v symmetry."""
        cell = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(half):
            for j in range(half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][size - 1 - j] = val
                cell[size - 1 - i][j] = val
                cell[size - 1 - i][size - 1 - j] = val
        return cell
    
    # Create diverse initial population
    def create_initial_population(pop_size):
        population = []
        cx, cy = 7.5, 7.5
        
        # 1. Super-ellipse (rounded square)
        q1 = [[0 for _ in range(half)] for _ in range(half)]
        a, n_exp = 5.0, 4.0
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                dx, dy = abs(x - cx)/a, abs(y - cy)/a
                if dx**n_exp + dy**n_exp < 1.0:
                    q1[i][j] = 1
        population.append(q1)
        
        # 2. Circle
        q2 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if (x-cx)**2 + (y-cy)**2 < 4.5**2:
                    q2[i][j] = 1
        population.append(q2)
        
        # 3. Square
        q3 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 4.5 and abs(y-cy) < 4.5:
                    q3[i][j] = 1
        population.append(q3)
        
        # 4. Ring
        q4 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                r = math.sqrt((x-cx)**2 + (y-cy)**2)
                if 2.0 < r < 5.5:
                    q4[i][j] = 1
        population.append(q4)
        
        # 5. Cross
        q5 = [[0 for _ in range(half)] for _ in range(half)]
        for i in range(half):
            for j in range(half):
                x, y = i + 0.5, j + 0.5
                if abs(x-cx) < 2.5 or abs(y-cy) < 2.5:
                    q5[i][j] = 1
        population.append(q5)
        
        # 6-12. Random perturbations
        rng = random.Random(42)
        for _ in range(pop_size - 5):
            base = population[rng.randint(0, 4)]
            q = [row[:] for row in base]
            for i in range(half):
                for j in range(half):
                    if rng.random() < 0.15:
                        q[i][j] = 1 - q[i][j]
            population.append(q)
        
        return population[:pop_size]
    
    # Crossover
    def crossover(q1, q2, rate=0.25):
        child1 = [row[:] for row in q1]
        child2 = [row[:] for row in q2]
        for i in range(half):
            for j in range(half):
                if random.random() < rate:
                    child1[i][j], child2[i][j] = child2[i][j], child1[i][j]
        return child1, child2
    
    # Mutation
    def mutate(quadrant, rate=0.04):
        q = [row[:] for row in quadrant]
        for i in range(half):
            for j in range(half):
                if random.random() < rate:
                    q[i][j] = 1 - q[i][j]
        return q
    
    # Evaluate
    def evaluate_quadrant(quadrant):
        cell = quadrant_to_cell(quadrant)
        cell_arr = np.array(cell, dtype=np.float64)
        return fast_tm_gap(cell_arr)
    
    # Main GA
    random.seed(123)
    pop_size = 15
    population = create_initial_population(pop_size)
    
    # Evaluate initial population
    scores = [evaluate_quadrant(q) for q in population]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_quadrant = [row[:] for row in population[best_idx]]
    
    generation = 0
    max_generations = 40
    
    while generation < max_generations:
        if time.time() > deadline - 25:
            break
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Elitism: keep top half
        elite_count = pop_size // 2
        new_population = [population[i][:] for i in sorted_indices[:elite_count]]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 25:
                break
            
            # Tournament selection from top
            t_size = min(5, len(sorted_indices))
            p1_idx = sorted_indices[random.randint(0, t_size - 1)]
            p2_idx = sorted_indices[random.randint(0, t_size - 1)]
            
            c1, c2 = crossover(population[p1_idx], population[p2_idx])
            c1 = mutate(c1, rate=0.03)
            c2 = mutate(c2, rate=0.03)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        population = new_population[:pop_size]
        scores = [evaluate_quadrant(q) for q in population]
        
        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_quadrant = [row[:] for row in population[gen_best_idx]]
        
        generation += 1
    
    # Local refinement: single-pixel flips around best
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(half):
                if time.time() > deadline - 5:
                    break
                for j in range(half):
                    if time.time() > deadline - 5:
                        break
                    q = [row[:] for row in best_quadrant]
                    q[i][j] = 1 - q[i][j]
                    score = evaluate_quadrant(q)
                    if score > best_score:
                        best_score = score
                        best_quadrant = q
                        improved = True
    
    # Convert to final cell
    best_cell = quadrant_to_cell(best_quadrant)
    
    # Verify symmetry
    for i in range(size):
        for j in range(size):
            if best_cell[i][j] != best_cell[size-1-i][j]:
                raise ValueError("Row symmetry failed")
            if best_cell[i][j] != best_cell[i][size-1-j]:
                raise ValueError("Col symmetry failed")
            if best_cell[i][j] != best_cell[j][i]:
                raise ValueError("Transpose symmetry failed")
    
    return best_cell
# EVOLVE-BLOCK-END
