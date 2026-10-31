"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
import time

N = 21


def _energy_and_grad(pos, aspect, r):
    """
    Compute energy and gradient for circle packing.
    pos: (N, 2) positions
    aspect: scalar, width/height ratio (perimeter=4 => w+h=2)
    r: (N,) radii
    Returns: energy, dE/dpos, dE/daspect
    """
    # Rectangle dimensions from perimeter 4: 2(w+h)=4 => w+h=2
    # w = 2*aspect/(1+aspect), h = 2/(1+aspect)
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    n = len(pos)
    energy = 0.0
    grad_pos = np.zeros_like(pos)
    grad_aspect = 0.0
    
    # Pairwise repulsion: soft overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            min_dist = r[i] + r[j]
            if dist < 1e-10:
                dist = 1e-10
                d = np.array([1e-10, 0.0])
            overlap = min_dist - dist
            if overlap > 0:
                # Quadratic penalty for overlap
                energy += 0.5 * overlap ** 2
                # Gradient wrt pos[i]: push i away from j
                dir_ij = d / dist
                grad_pos[i] += overlap * dir_ij
                grad_pos[j] -= overlap * dir_ij
    
    # Wall repulsion: circles must stay inside rectangle
    for i in range(n):
        xi, yi = pos[i]
        ri = r[i]
        
        # Left wall
        overlap_l = ri - xi
        if overlap_l > 0:
            energy += 0.5 * overlap_l ** 2
            grad_pos[i, 0] += overlap_l
        
        # Right wall
        overlap_r = (xi + ri) - w
        if overlap_r > 0:
            energy += 0.5 * overlap_r ** 2
            grad_pos[i, 0] -= overlap_r
        
        # Bottom wall
        overlap_b = ri - yi
        if overlap_b > 0:
            energy += 0.5 * overlap_b ** 2
            grad_pos[i, 1] += overlap_b
        
        # Top wall
        overlap_t = (yi + ri) - h
        if overlap_t > 0:
            energy += 0.5 * overlap_t ** 2
            grad_pos[i, 1] -= overlap_t
    
    # Aspect ratio gradient: compute dE/daspect via finite difference
    # w = 2a/(1+a), dw/da = 2/(1+a)^2
    # h = 2/(1+a), dh/da = -2/(1+a)^2
    dw_da = 2.0 / (1.0 + aspect) ** 2
    dh_da = -2.0 / (1.0 + aspect) ** 2
    
    # dE/daspect = sum over wall overlaps of d(overlap)/d(w or h) * dw/da or dh/da
    for i in range(n):
        xi, yi = pos[i]
        ri = r[i]
        
        # Left wall: overlap = ri - xi, no direct aspect dependence
        # Right wall: overlap = xi + ri - w, d(overlap)/dw = -1
        overlap_r = (xi + ri) - w
        if overlap_r > 0:
            grad_aspect += overlap_r * (-1.0) * dw_da
        
        # Bottom wall: no direct aspect dependence
        # Top wall: overlap = yi + ri - h, d(overlap)/dh = -1
        overlap_t = (yi + ri) - h
        if overlap_t > 0:
            grad_aspect += overlap_t * (-1.0) * dh_da
    
    return energy, grad_pos, grad_aspect


def _simulate(pos0, aspect0, r, n_steps=2000, lr_pos=0.01, lr_aspect=0.001):
    """
    Run gradient descent to minimize energy.
    Returns final pos, aspect, energy.
    """
    pos = pos0.copy()
    aspect = aspect0
    
    for step in range(n_steps):
        energy, grad_pos, grad_aspect = _energy_and_grad(pos, aspect, r)
        
        # Gradient descent
        pos -= lr_pos * grad_pos
        aspect -= lr_aspect * grad_aspect
        
        # Keep aspect in reasonable range
        aspect = np.clip(aspect, 0.2, 5.0)
        
        # Clamp positions to stay roughly in bounds
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        for i in range(len(pos)):
            pos[i, 0] = np.clip(pos[i, 0], 0.0, w)
            pos[i, 1] = np.clip(pos[i, 1], 0.0, h)
        
        # Adaptive learning rate
        if step % 200 == 199:
            if energy < 1e-6:
                break
            lr_pos *= 0.9
            lr_aspect *= 0.9
    
    energy, _, _ = _energy_and_grad(pos, aspect, r)
    return pos, aspect, energy


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    best_score = -1.0
    best_result = None
    
    # Try multiple aspect ratios and initial configurations
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Vary aspect ratio
        aspect0 = 0.5 + attempt * 0.15  # Range from 0.5 to ~4.85
        aspect0 = np.clip(aspect0, 0.3, 5.0)
        
        w0 = 2.0 * aspect0 / (1.0 + aspect0)
        h0 = 2.0 / (1.0 + aspect0)
        
        # Grid-based initial positions
        cols = int(np.ceil(np.sqrt(N * aspect0)))
        rows = int(np.ceil(N / cols))
        
        pos0 = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            # Center of each cell
            pos0[i, 0] = (col + 0.5) * w0 / cols
            pos0[i, 1] = (row + 0.5) * h0 / rows
        
        # Estimate radii: minimum of cell size / 2 and distance to walls
        cell_w = w0 / cols
        cell_h = h0 / rows
        r = np.full(N, 0.5 * min(cell_w, cell_h) * 0.95)
        
        # Run simulation
        pos, aspect, energy = _simulate(pos0, aspect0, r, n_steps=1500, lr_pos=0.005, lr_aspect=0.0005)
        
        # Check validity and compute score
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Verify no overlaps and all circles inside
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
            # Check walls
            if pos[i, 0] < r[i] - 1e-9 or pos[i, 0] + r[i] > w + 1e-9:
                valid = False
                break
            if pos[i, 1] < r[i] - 1e-9 or pos[i, 1] + r[i] > h + 1e-9:
                valid = False
                break
        
        if not valid:
            # Try to fix by shrinking radii slightly
            scale = 0.9
            r = r * scale
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                if pos[i, 0] < r[i] - 1e-9 or pos[i, 0] + r[i] > w + 1e-9:
                    valid = False
                    break
                if pos[i, 1] < r[i] - 1e-9 or pos[i, 1] + r[i] > h + 1e-9:
                    valid = False
                    break
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                for i in range(N):
                    result[i] = [pos[i, 0], pos[i, 1], r[i]]
                best_result = result
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
