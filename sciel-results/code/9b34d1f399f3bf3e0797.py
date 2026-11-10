"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n.
    Uses a simple random walk with backtracking if stuck.
    Returns list of (x, y) tuples.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    def gen(length, current, occupied):
        if length == 0:
            return [current]
        
        dirs = list(range(4))
        rng.shuffle(dirs)
        
        for d in dirs:
            nx, ny = current[0] + DIRS[d][0], current[1] + DIRS[d][1]
            new_pos = (nx, ny)
            if new_pos not in occupied:
                occupied.add(new_pos)
                suffix = gen(length - 1, new_pos, occupied)
                if suffix:
                    return [current] + suffix
                occupied.remove(new_pos)
        
        return None

    occupied = {(0, 0)}
    result = gen(n - 1, (0, 0), occupied)
    if result is None:
        return [(i, 0) for i in range(n)]
    return result

def _regrow_suffix(chain, sequence, start_idx, rng, temperature=0.0):
    """
    Regrow the suffix of the chain starting from start_idx.
    Keeps chain[0:start_idx] fixed.
    Returns new suffix or None if stuck.
    """
    prefix = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    new_suffix = []
    last = prefix[-1]
    
    for i in range(start_idx, len(chain)):
        res = sequence[i]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        new_suffix.append(best_site)
        last = best_site
        
    return new_suffix

def _local_improve_pivot(sequence, chain, rng, max_iters=5000):
    """
    Local improvement using Regrow moves.
    """
    n = len(chain)
    if n < 5:
        return chain, _compute_energy(sequence, chain)
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = 2.0
    cool_rate = 0.995
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        if temp < 0.05:
            break
            
        move_type = rng.random()
        
        if move_type < 0.6:
            min_split = 3
            max_split = max(min_split, n - 2)
            if max_split <= min_split:
                temp *= cool_rate
                continue
            split = rng.randint(min_split, max_split)
            
            new_suffix = _regrow_suffix(chain, sequence, split, rng, temperature=temp)
            if new_suffix is None:
                temp *= cool_rate
                continue
            
            new_chain = chain[:split] + new_suffix
            new_energy = _compute_energy(sequence, new_chain)
            
            delta = new_energy - current_energy
            if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    temp = min(temp * 1.05, 3.0)
                else:
                    temp *= cool_rate
            else:
                temp *= cool_rate
                
        else:
            split = max(3, n - rng.randint(5, 10))
            new_suffix = _regrow_suffix(chain, sequence, split, rng, temperature=temp * 0.5)
            if new_suffix is None:
                temp *= cool_rate
                continue
            
            new_chain = chain[:split] + new_suffix
            new_energy = _compute_energy(sequence, new_chain)
            
            delta = new_energy - current_energy
            if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    temp = min(temp * 1.05, 3.0)
                else:
                    temp *= cool_rate
            else:
                temp *= cool_rate

    return best_chain, best_energy

def _continuous_fold(sequence, n, rng):
    """
    Formulate folding as continuous 2D vector field optimization.
    1. Initialize random continuous coordinates.
    2. Gradient descent to minimize energy (maximize H-H contacts) + self-avoidance penalty.
    3. Snap to nearest integer lattice.
    4. Repair self-avoidance if necessary.
    """
    # 1. Initialize
    # Start with a loose spiral or random points
    # Let's try a random walk in continuous space, but scaled up
    coords = np.zeros((n, 2))
    for i in range(1, n):
        angle = rng.uniform(0, 2 * math.pi)
        step = 1.0
        coords[i] = coords[i-1] + np.array([math.cos(angle), math.sin(angle)]) * step
    
    # 2. Gradient Descent
    # We want to maximize H-H contacts.
    # Energy E = - sum_{i<j, |i-j|>1} I(seq[i]=H, seq[j]=H) * I(dist(i,j) < 1.5)
    # We use a soft potential for distance: V(r) = exp(-(r-1)^2 / sigma^2) or similar.
    # Actually, we want to pull H-H pairs together if they are close, and push non-H-H or all pairs apart if overlapping.
    
    # Let's define a loss function:
    # L = - lambda_contact * sum_{i<j, H,H} exp(-(||r_i - r_j||^2 - 1)^2 / 0.5) 
    #     + lambda_avoid * sum_{i!=j} exp(-||r_i - r_j||^2 / 0.1)  # Repulsive for close pairs
    
    # This is computationally heavy for N=100 (100*100 interactions).
    # But N is small enough for a few iterations?
    # 100*100 = 10k pairs. Gradient descent for 100 steps = 1M ops. Very fast.
    
    lambda_contact = 5.0
    lambda_avoid = 1.0
    
    lr = 0.05
    num_steps = 200
    
    # Precompute H indices
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    
    for step in range(num_steps):
        if time.time() > _DEADLINE - 10:
            break
            
        grad = np.zeros_like(coords)
        
        # Calculate forces
        # 1. Repulsion between all pairs (i != j)
        # We can use numpy broadcasting for speed
        # dist_sq[i, j] = ||coords[i] - coords[j]||^2
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Mask out self
        mask = ~np.eye(n, dtype=bool)
        
        # Repulsive force: d/dx_i ( exp(-dist_sq / sigma^2) ) = -2 * (r_i - r_j) / sigma^2 * exp(...)
        # We want to push apart, so force is attractive if we minimize repulsive energy?
        # Energy_rep = sum exp(-dist_sq / sigma^2).
        # Grad w.r.t r_i: sum_j -2/r_j * exp(...) * (r_i - r_j)
        # To minimize energy, we move in direction -grad.
        # So if close, force pushes apart.
        
        sigma_rep = 0.5
        rep_energy = np.exp(-dist_sq / (2 * sigma_rep**2))
        rep_energy[~mask] = 0
        
        # Gradient of rep_energy w.r.t coords[i]
        # grad_i = -2 / sigma_rep^2 * sum_j rep_energy[i,j] * (coords[i] - coords[j])
        # We can compute this with matrix multiplication
        # grad_rep[i] = -2 / sigma_rep^2 * (rep_energy[i] @ (coords[i] - coords)) ? No.
        
        # Let's use explicit loop for clarity and safety, N=100 is small.
        # Or vectorized:
        # grad_rep = -2 / sigma_rep^2 * (rep_energy * diff).sum(axis=1)
        # Wait, diff[i,j] = r_i - r_j.
        # d/d r_i ( exp(-||r_i-r_j||^2 / 2s^2) ) = exp(...) * (-1/2s^2) * 2(r_i-r_j) = -1/s^2 * (r_i-r_j) * exp(...)
        # So grad_rep[i] = sum_j -1/s^2 * (r_i-r_j) * exp(...)
        
        grad_rep = -1.0 / (sigma_rep**2) * (rep_energy * diff).sum(axis=1)
        
        # 2. Attractive force for H-H pairs
        # We want to maximize contacts.
        # Contact potential: V(r) = - exp(-(dist_sq - 1)^2 / 0.5)
        # We want to minimize -V, i.e., maximize V.
        # So we subtract the gradient of V from the total gradient (or add to loss).
        # Loss_contact = - sum_{H,H} exp(-(dist_sq - 1)^2 / 0.5)
        # Grad Loss_contact w.r.t r_i = - sum_{j in H, j!=i} d/dr_i exp(-(dist_sq - 1)^2 / 0.5)
        # d/dr_i exp(-(d-1)^2/0.5) = exp(...) * (-1/0.5) * 2(d-1) * d/d r_i (d)
        # d/d r_i (||r_i-r_j||^2) = 2(r_i-r_j)
        # So term = -2 * exp(...) * (d-1) * (r_i-r_j) / 0.5 ?
        
        sigma_att = 0.7
        grad_att = np.zeros_like(coords)
        
        for i in h_indices:
            for j in h_indices:
                if i == j:
                    continue
                # Avoid double counting? Gradient is symmetric.
                # If we sum over all j, each pair is counted twice in the global sum, which is fine for gradient.
                rij = coords[i] - coords[j]
                dist_sq_ij = np.dot(rij, rij)
                d = dist_sq_ij - 1.0
                # Derivative of exp(-d^2 / (2 * sigma_att^2)) w.r.t r_i
                # Let f = exp(-d^2 / (2 s^2))
                # df/dr_i = f * (-1 / (2 s^2)) * 2d * 2r_ij = -f * d * r_ij / s^2
                # Wait, d = dist_sq - 1. d(dist_sq)/dr_i = 2 r_ij.
                # So d(d)/dr_i = 2 r_ij.
                # df/dr_i = exp(...) * (-1/(2s^2)) * 2d * 2r_ij = -exp(...) * d * r_ij / s^2
                
                val = math.exp(-d*d / (2 * sigma_att**2))
                grad_i = -val * d * rij / (sigma_att**2)
                
                # We are minimizing Loss = -V.
                # Grad Loss = - Grad V.
                # So we add -grad_i to grad_att?
                # Let's just compute grad of V and subtract it from total grad (since we want to minimize -V).
                # Total Grad = Grad_Rep + Grad_Att_Loss
                # Grad_Att_Loss = - Grad_V
                # So we add -grad_i to grad_att.
                grad_att[i] += -grad_i
        
        # Total Gradient
        grad = lambda_avoid * grad_rep + lambda_contact * grad_att
        
        # Update
        coords -= lr * grad
        
        # Optional: Normalize step size
        # norm = np.linalg.norm(grad)
        # if norm > 0:
        #     coords -= lr * grad / norm
        
    # 3. Snap to nearest integer
    snapped = np.round(coords).astype(int)
    
    # 4. Repair self-avoidance
    # This is the hard part. The snapped chain might have overlaps or non-unit steps.
    # We need to convert it to a valid SAW.
    # Strategy: Use the snapped coordinates as a "target" for a regrow-based repair.
    # Or, simpler: If the continuous fold produced a compact blob, the snapped version is likely close to a valid fold.
    # We can try to fix it by regrowing from the start, but guided by the snapped positions.
    
    # Let's try a simple repair:
    # Iterate through the chain. If a point is occupied or not unit step, move it to the nearest valid spot.
    # This is heuristic and might fail.
    
    # Better: Use the continuous solution to initialize a Regrow-SA.
    # The continuous solution gives us a good "shape".
    # We can use it to bias the Regrow moves.
    
    # For now, let's just return the snapped chain if it's valid, else a random SAW.
    # Checking validity:
    valid = True
    occupied = set()
    for i in range(n):
        p = tuple(snapped[i])
        if p in occupied:
            valid = False
            break
        occupied.add(p)
        if i > 0:
            prev = tuple(snapped[i-1])
            if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                valid = False
                break
    
    if valid:
        return [list(p) for p in snapped]
    else:
        # Fallback: Return a random SAW, but maybe we can do better?
        # Let's return None to signal failure, and let the main loop try other methods.
        return None

def fold(sequence, dimension):
    """
    Fold the sequence using a hybrid approach:
    1. Try Continuous Vector Field Optimization.
    2. If it fails or yields a poor result, fall back to Regrow-SA.
    3. Run multiple independent instances of both.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # 170s total.
    # Let's allocate 50% to Continuous, 50% to Discrete SA.
    # Or run them in parallel? No, single thread.
    # Let's run 5 Continuous attempts and 5 Discrete attempts.
    
    # Continuous attempts
    for i in range(5):
        if time.time() > _DEADLINE - 20:
            break
        
        # Add some noise to initialization to get diverse results
        # The _continuous_fold function uses rng, so it will be different each time?
        # Yes, because it calls rng.uniform.
        
        chain = _continuous_fold(sequence, n, rng)
        
        if chain is not None:
            energy = _compute_energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            
            # Refine the continuous result with a short SA run
            # This can fix minor issues and improve energy
            refined_chain, refined_energy = _local_improve_pivot(sequence, chain, rng, max_iters=2000)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_chain = refined_chain
    
    # Discrete SA attempts
    for i in range(5):
        if time.time() > _DEADLINE - 10:
            break
        
        init_chain = _generate_random_saw(n, rng)
        
        # Run SA
        sa_chain, sa_energy = _local_improve_pivot(sequence, init_chain, rng, max_iters=5000)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain
    
    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
