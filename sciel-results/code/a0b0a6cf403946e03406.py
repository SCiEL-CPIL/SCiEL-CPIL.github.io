"""
Strategy: Lattice Gas Model with Parallel Tempering Monte Carlo
Instead of continuous coordinates, we discretize the cube into a grid.
We model the placement of 100 spheres as a lattice gas with N particles.
The energy function penalizes overlaps (hard constraint via large penalty) 
and wall violations.
We use Parallel Tempering (Replica Exchange MC) to escape local minima.
Different replicas operate at different "temperatures" (inversion of acceptance probability).
After MC, we map back to continuous coordinates and perform a local gradient polish.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    
    # Wall distances
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_wall_global, min_dist / 2.0)

def _soft_min_objective(x, beta):
    """
    Objective function for continuous polish.
    Minimizes the soft-min of wall distances and half-pairwise distances.
    """
    c = x.reshape(N, 3)
    
    # Wall terms
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    v_walls = walls.flatten()
    v_pairs = dist.flatten() / 2.0
    v_pairs = v_pairs[v_pairs < np.inf]
    
    all_vals = np.concatenate([v_walls, v_pairs])
    
    m = np.min(all_vals)
    terms = np.exp(-beta * (all_vals - m))
    log_sum = np.log(np.sum(terms))
    
    obj = -m + log_sum / beta
    return obj

def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective.
    """
    c = x.reshape(N, 3)
    
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    v_walls = walls.flatten()
    v_pairs = dist.flatten() / 2.0
    v_pairs = v_pairs[v_pairs < np.inf]
    
    all_vals = np.concatenate([v_walls, v_pairs])
    
    m = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    sum_exp = np.sum(exp_terms)
    p = exp_terms / sum_exp
    
    grad = np.zeros_like(c)
    
    p_walls = p[:N*6].reshape(N, 6)
    grad[:, 0] += p_walls[:, 0] - p_walls[:, 1]
    grad[:, 1] += p_walls[:, 2] - p_walls[:, 3]
    grad[:, 2] += p_walls[:, 4] - p_walls[:, 5]
    
    p_pairs = p[N*6:]
    
    P_mat = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            P_mat[i, j] = p_pairs[idx]
            idx += 1
            
    dist_safe = np.where(dist > 1e-8, dist, 1e-8)
    coeff = P_mat / (2.0 * dist_safe)
    weighted_diff = coeff[:, :, np.newaxis] * diff
    grad -= np.sum(weighted_diff, axis=1)
    
    return grad.flatten()

def _soft_min_jac(x, beta):
    obj = _soft_min_objective(x, beta)
    grad = _soft_min_grad(x, beta)
    return obj, grad

class LatticeGasMC:
    def __init__(self, grid_size, n_particles, time_limit):
        self.grid_size = grid_size
        self.n_particles = n_particles
        self.time_limit = time_limit
        self.cells = grid_size ** 3
        # Occupancy: 1 if particle, 0 if empty
        self.occ = np.zeros(self.cells, dtype=np.int8)
        self.particle_ids = np.full(self.cells, -1, dtype=np.int32)
        
        # Precompute neighbor offsets for distance checks
        # For a lattice gas, we need to check distances.
        # To be efficient, we only check neighbors within a certain range.
        # Max radius ~ 0.11. Grid size G. Cell size = 1/G.
        # Distance in cells = dist / (1/G) = dist * G.
        # Max dist ~ 0.22. Cells ~ 0.22 * G.
        # For G=30, 0.22*30 ~ 6.6 cells.
        # We will check neighbors within a box of radius R_cells.
        self.R_cells = int(0.25 * grid_size) + 1
        
        # Precompute list of neighbor offsets
        offsets = []
        for dx in range(-self.R_cells, self.R_cells + 1):
            for dy in range(-self.R_cells, self.R_cells + 1):
                for dz in range(-self.R_cells, self.R_cells + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz))
        self.offsets = np.array(offsets)
        
    def _init_random(self):
        self.occ = np.zeros(self.cells, dtype=np.int8)
        self.particle_ids = np.full(self.cells, -1, dtype=np.int32)
        available = np.arange(self.cells)
        chosen = np.random.choice(available, self.n_particles, replace=False)
        for i, cell in enumerate(chosen):
            self.occ[cell] = 1
            self.particle_ids[cell] = i
            
    def _get_continuous_coords(self):
        # Map lattice indices to continuous coordinates in [-0.5, 0.5]
        # Lattice index (i, j, k) -> coord ((i+0.5)/G - 0.5, ...)
        G = self.grid_size
        indices = np.argwhere(self.occ == 1)
        coords = (indices + 0.5) / G - 0.5
        # Assign particle IDs based on order? 
        # For scoring, we just need the set of points.
        return coords

    def _energy(self):
        # Energy is sum of pair potentials + wall penalties
        # We use a soft repulsive potential to guide the search
        # E = sum_{i<j} V(r_ij) + sum_i V_wall(c_i)
        
        coords = self._get_continuous_coords()
        
        # Wall penalty: quadratic if outside "safe" zone? 
        # Actually, lattice points are always inside [-0.5, 0.5].
        # Min wall distance for a point at (0.5/G - 0.5) is 0.5/G.
        # We want to maximize min radius.
        # Let's define energy as - (min radius proxy)
        # Or simply penalize close pairs.
        
        # Pairwise distances
        if len(coords) < 2:
            return 0.0
            
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize the minimum distance.
        # Energy = - min(dists) is non-differentiable and sparse.
        # Use soft-min or sum of exp(-k * d)
        # Let's use a simple sum of inverse square or exponential decay
        # E_pair = sum_{i<j} exp(-d_ij / sigma)
        # Small d -> large E.
        
        sigma = 0.05
        # Only sum close pairs to save time? 
        # For G=30, 100 particles, 4950 pairs. cdist is fast.
        e_pairs = np.sum(np.exp(-dists / sigma))
        
        # Wall penalty: 
        # Points near walls have smaller max possible radius.
        # r_max_i = min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
        # We want to maximize min_i r_max_i.
        # So penalty for small r_max_i.
        abs_coords = np.abs(coords)
        wall_dists = 0.5 - np.max(abs_coords, axis=1)
        # Penalize if wall_dist < 0.1 (approx max radius)
        # E_wall = sum (0.1 - wall_dist)^2 if wall_dist < 0.1 else 0
        e_wall = np.sum(np.maximum(0, 0.11 - wall_dists) ** 2)
        
        return e_pairs + 10.0 * e_wall

    def _move(self, temp):
        # Swap a particle with a random empty cell
        # Or move a particle to a random empty cell (and swap with nothing? No, lattice gas swaps)
        # Standard lattice gas: swap occupied and unoccupied.
        
        # Pick random occupied cell
        occupied_indices = np.where(self.occ == 1)[0]
        if len(occupied_indices) == 0:
            return False
        i_occ = occupied_indices[np.random.randint(len(occupied_indices))]
        
        # Pick random unoccupied cell
        unoccupied_indices = np.where(self.occ == 0)[0]
        if len(unoccupied_indices) == 0:
            return False
        i_unocc = unoccupied_indices[np.random.randint(len(unoccupied_indices))]
        
        # Calculate delta E
        # This requires efficient energy delta calculation.
        # For simplicity and robustness within time limit, we might just recalculate 
        # if N is small enough or use neighbor lists.
        # With G=30, N=100, full recalc is ~5000 ops. 100k steps -> 500M ops. Might be slow in Python.
        # Let's optimize: Precompute cell coords.
        
        # For now, let's try a simpler move: 
        # Try to move a particle to a neighbor cell if empty.
        # This is a random walk.
        
        # Better: Use the precomputed offsets.
        # But we need to check energy delta.
        
        # Let's use a "kick" approach: 
        # Pick a particle. Generate a random neighbor cell (within R_cells).
        # If empty, calculate delta E.
        
        # To make it faster, we'll approximate delta E using only neighbors of the moving particle.
        
        # Actually, let's just do full recalc but limit steps.
        # 150s / 100k steps = 1.5ms per step. 
        # cdist 100x100 is ~0.1ms. Sum exp is fast.
        # So 100k steps is feasible.
        
        e_old = self._energy()
        
        # Perform swap
        self.occ[i_occ] = 0
        self.occ[i_unocc] = 1
        pid = self.particle_ids[i_occ]
        self.particle_ids[i_unocc] = pid
        self.particle_ids[i_occ] = -1
        
        e_new = self._energy()
        
        delta_e = e_new - e_old
        
        # Metropolis criterion
        if delta_e < 0:
            return True
        else:
            if np.random.random() < np.exp(-delta_e / temp):
                return True
            else:
                # Revert
                self.occ[i_unocc] = 0
                self.occ[i_occ] = 1
                self.particle_ids[i_occ] = pid
                self.particle_ids[i_unocc] = -1
                return False

    def run(self):
        start_time = time.time()
        self._init_random()
        
        # Parallel Tempering
        n_reps = 5
        temps = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
        # State for each replica
        reps_occ = [np.zeros_like(self.occ) for _ in range(n_reps)]
        reps_pids = [np.zeros_like(self.particle_ids) for _ in range(n_reps)]
        reps_e = [0.0] * n_reps
        
        for r in range(n_reps):
            self._init_random()
            reps_occ[r] = self.occ.copy()
            reps_pids[r] = self.particle_ids.copy()
            # Temporarily set state to compute energy
            self.occ = reps_occ[r]
            self.particle_ids = reps_pids[r]
            reps_e[r] = self._energy()
            
        best_e = np.inf
        best_state = None
        
        steps_per_iter = 100
        exchange_freq = 10
        
        step = 0
        while time.time() < self.time_limit - 10.0:
            # Do MC steps for each replica
            for r in range(n_reps):
                self.occ = reps_occ[r]
                self.particle_ids = reps_pids[r]
                
                for _ in range(steps_per_iter):
                    self._move(temps[r])
                    
                # Update energy (approx: recalc)
                reps_e[r] = self._energy()
                
                if reps_e[r] < best_e:
                    best_e = reps_e[r]
                    best_state = (reps_occ[r].copy(), reps_pids[r].copy())
            
            # Exchange replicas
            if step % exchange_freq == 0:
                for r in range(n_reps - 1):
                    # Try swap r and r+1
                    # Accept if exp(-(E_r/T_{r+1} - E_{r+1}/T_r)) > rand
                    # Or standard: accept if exp( (E_r/T_r - E_r/T_{r+1}) + (E_{r+1}/T_{r+1} - E_{r+1}/T_r) ) > rand
                    # Which is exp( E_r*(1/T_r - 1/T_{r+1}) + E_{r+1}*(1/T_{r+1} - 1/T_r) )
                    
                    # Simpler: 
                    # P = exp( - (E_r/T_{r+1} - E_{r+1}/T_{r+1}) + (E_r/T_r - E_{r+1}/T_r) )
                    # Let's use:
                    # delta = E_r/T_{r+1} - E_{r+1}/T_{r+1} - (E_r/T_r - E_{r+1}/T_r)
                    # Wait, the probability of swapping states r and r+1 given temps T_r, T_{r+1}
                    # is proportional to exp(-E_r/T_{r+1}) exp(-E_{r+1}/T_r) / (exp(-E_r/T_r) exp(-E_{r+1}/T_{r+1}))
                    # = exp( -E_r/T_{r+1} + E_{r+1}/T_r + E_r/T_r - E_{r+1}/T_{r+1} )
                    # = exp( E_r(1/T_r - 1/T_{r+1}) + E_{r+1}(1/T_{r+1} - 1/T_r) )
                    
                    term1 = reps_e[r] * (1.0/temps[r] - 1.0/temps[r+1])
                    term2 = reps_e[r+1] * (1.0/temps[r+1] - 1.0/temps[r])
                    log_prob = term1 + term2
                    
                    if np.random.random() < np.exp(log_prob):
                        # Swap
                        reps_occ[r], reps_occ[r+1] = reps_occ[r+1], reps_occ[r]
                        reps_pids[r], reps_pids[r+1] = reps_pids[r+1], reps_pids[r]
                        reps_e[r], reps_e[r+1] = reps_e[r+1], reps_e[r]
            
            step += 1
            
        # Return best state
        if best_state is not None:
            self.occ, self.particle_ids = best_state
        return self._get_continuous_coords()

def _polish_continuous(c0, time_limit):
    """
    Polish the lattice solution using L-BFGS-B on the continuous soft-min objective.
    """
    start_time = time.time()
    c = c0.copy()
    
    try:
        # Use a moderate beta for initial polish
        beta = 100.0
        res = minimize(_soft_min_jac, c.flatten(), args=(beta,), 
                       jac=True, method='L-BFGS-B',
                       options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        
        c_polished = res.x.reshape(N, 3)
        c_polished = np.clip(c_polished, -0.5, 0.5)
        
        # Check improvement
        r_before = _compute_radius(c)
        r_after = _compute_radius(c_polished)
        
        if r_after > r_before:
            c = c_polished
            
        # Second pass with higher beta if time permits
        if time.time() < time_limit - 2.0:
            beta = 500.0
            res2 = minimize(_soft_min_jac, c.flatten(), args=(beta,), 
                           jac=True, method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            
            c_polished2 = res2.x.reshape(N, 3)
            c_polished2 = np.clip(c_polished2, -0.5, 0.5)
            
            r_before2 = _compute_radius(c)
            r_after2 = _compute_radius(c_polished2)
            
            if r_after2 > r_before2:
                c = c_polished2
                
    except Exception as e:
        pass
        
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Strategy:
    # 1. Run Lattice Gas MC with Parallel Tempering.
    # 2. Get continuous coords.
    # 3. Polish with L-BFGS-B.
    
    # Grid size: 30 seems reasonable. 30^3 = 27000 cells.
    # N=100. Density ~ 0.4%.
    # Larger grid = more freedom, but slower neighbor checks if we used them.
    # Since we use full cdist, grid size mainly affects resolution of initial guess.
    # G=30 -> resolution 0.033. Max radius ~0.11. So ~3 cells radius.
    
    G = 30
    mc = LatticeGasMC(G, N, deadline - 10.0)
    
    c_lattice = mc.run()
    
    # Polish
    c_final = _polish_continuous(c_lattice, deadline - 1.0)
    
    # Safety: ensure valid shape and finite
    if c_final.shape != (N, 3) or not np.all(np.isfinite(c_final)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
        
    return c_final

# EVOLVE-BLOCK-END
