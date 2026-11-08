"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    # Boundary checks
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    # Pairwise overlap check
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _build_grid(n, jitter=0.0):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.45 / side
    radii = np.full(n, r)
    return pos, radii


def _build_hex(n, jitter=0.0):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.55 / cols
            y = (row + 0.5) * 0.5 / (int(np.ceil(n / cols)) + 1)
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    if jitter > 0:
        pos += np.random.uniform(-jitter, jitter, size=pos.shape)
        pos = np.clip(pos, 0.01, 0.99)
    r = 0.22
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=None):
    """Build a random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.08)
    return pos, radii


def _relax_and_grow(pos, radii, max_iters=5000, time_budget=30.0, start_time=None):
    """Physics-based relaxation with simultaneous radius growth."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    # Precompute pair indices
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Check boundary violations
        for i in range(n):
            # Left boundary
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            # Right boundary
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            # Bottom boundary
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            # Top boundary
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Resolve overlaps
        for i, j in pairs:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            
            if dist < min_dist:
                if dist < 1e-10:
                    # Random small separation
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = (min_dist - dist) / 2
                    pos[i, 0] -= push * np.cos(angle)
                    pos[i, 1] -= push * np.sin(angle)
                    pos[j, 0] += push * np.cos(angle)
                    pos[j, 1] += push * np.sin(angle)
                else:
                    # Push apart along the connecting line
                    push = (min_dist - dist) / 2
                    nx = dx / dist
                    ny = dy / dist
                    pos[i, 0] -= push * nx
                    pos[i, 1] -= push * ny
                    pos[j, 0] += push * nx
                    pos[j, 1] += push * ny
        
        # Re-check boundaries after overlap resolution
        for i in range(n):
            if pos[i, 0] < radii[i]:
                pos[i, 0] = radii[i]
            if pos[i, 0] > 1 - radii[i]:
                pos[i, 0] = 1 - radii[i]
            if pos[i, 1] < radii[i]:
                pos[i, 1] = radii[i]
            if pos[i, 1] > 1 - radii[i]:
                pos[i, 1] = 1 - radii[i]
        
        # Try to grow radii
        if it % 50 == 0:
            grow_amount = 0.001
            new_radii = radii + grow_amount
            # Check if growth is feasible
            feasible = True
            for i in range(n):
                if new_radii[i] > 0.5 or new_radii[i] < 0.01:
                    feasible = False
                    break
                if pos[i, 0] < new_radii[i] or pos[i, 0] > 1 - new_radii[i]:
                    feasible = False
                    break
                if pos[i, 1] < new_radii[i] or pos[i, 1] > 1 - new_radii[i]:
                    feasible = False
                    break
            if feasible:
                for i, j in pairs:
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < new_radii[i] + new_radii[j] - 1e-9:
                        feasible = False
                        break
            if feasible:
                radii = new_radii
    
    return pos, radii


def _local_search(pos, radii, time_budget=10.0, start_time=None):
    """Local search: try small moves to improve packing."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() - start_time < time_budget:
        # Try to grow all radii slightly
        grow = 0.0005
        new_radii = radii + grow
        feasible = True
        for i in range(n):
            if new_radii[i] > 0.5:
                feasible = False
                break
            if pos[i, 0] < new_radii[i] - 1e-9 or pos[i, 0] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
            if pos[i, 1] < new_radii[i] - 1e-9 or pos[i, 1] > 1 - new_radii[i] + 1e-9:
                feasible = False
                break
        if feasible:
            for i, j in pairs:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < new_radii[i] + new_radii[j] - 1e-9:
                    feasible = False
                    break
        if feasible:
            radii = new_radii
            score = np.sum(radii)
            if score > best_score:
                best_score = score
            continue
        
        # Try small random moves
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            idx = np.random.randint(n)
            old_pos = pos[idx].copy()
            pos[idx] += np.random.normal(0, 0.005, size=2)
            pos[idx] = np.clip(pos[idx], 0, 1)
            
            # Check validity
            valid = True
            if pos[idx, 0] < radii[idx] - 1e-9 or pos[idx, 0] > 1 - radii[idx] + 1e-9:
                valid = False
            if pos[idx, 1] < radii[idx] - 1e-9 or pos[idx, 1] > 1 - radii[idx] + 1e-9:
                valid = False
            if valid:
                for j in range(n):
                    if j == idx:
                        continue
                    dx = pos[j, 0] - pos[idx, 0]
                    dy = pos[j, 1] - pos[idx, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[idx] + radii[j] - 1e-9:
                        valid = False
                        break
            if not valid:
                pos[idx] = old_pos
    
    return pos, radii


def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # Multiple initial configurations
    configs = []
    
    # Grid
    pos, radii = _build_grid(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Grid with jitter
    pos, radii = _build_grid(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Hexagonal
    pos, radii = _build_hex(N, jitter=0.0)
    configs.append((pos, radii))
    
    # Hexagonal with jitter
    pos, radii = _build_hex(N, jitter=0.02)
    configs.append((pos, radii))
    
    # Random starts
    for seed in [42, 123, 456]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos, radii))
    
    # Process each configuration
    for ci, (pos, radii) in enumerate(configs):
        if time.time() - start_time > deadline:
            break
        
        # Relax and grow
        pos, radii = _relax_and_grow(pos, radii, max_iters=3000, 
                                      time_budget=min(20.0, deadline - (time.time() - start_time)),
                                      start_time=start_time)
        
        # Local search
        remaining = deadline - (time.time() - start_time)
        if remaining > 2:
            pos, radii = _local_search(pos, radii, time_budget=min(8.0, remaining - 1),
                                         start_time=start_time)
        
        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        
        # Use this as starting point for next if better
        if ci < len(configs) - 1 and best_pos is not None:
            # Optionally, could use best as next start, but keep independent for diversity
            pass
    
    # Final polish if time remains
    if best_pos is not None and time.time() - start_time < deadline - 3:
        remaining = deadline - (time.time() - start_time) - 2
        best_pos, best_radii = _local_search(best_pos, best_radii, 
                                               time_budget=min(10.0, remaining),
                                               start_time=start_time)
        if _check_validity(best_pos, best_radii):
            best_score = np.sum(best_radii)
    
    # Return best result
    if best_pos is None:
        # Fallback: simple grid
        pos, radii = _build_grid(N, jitter=0.0)
        best_pos, best_radii = pos, radii
    
    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]
    
    return result_arr
# EVOLVE-BLOCK-END
