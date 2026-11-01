# So I should define `construct` inside the block or just define it globally if the block is just for evolution?
# The prompt says: "Output the COMPLETE program in a single python code block."
# And "Keep the program's entry-point function name(s) ... EXACTLY as in the current program above".
# The current program has `def construct():` inside the EVOLVE-BLOCK.
# I will structure my code to have `construct` defined.

# Since the seed code puts `construct` inside the block, I will do the same.
# However, the helper functions can be outside or inside. 
# To be safe and clean, I'll put everything in the block or just the main logic.
# Actually, the seed has imports outside. I will keep imports outside.
# The seed has `N=50` outside.
# The seed has `def construct():` inside the block.

# I will rewrite the whole file structure to match the seed's style but with my new logic.

# Wait, the prompt asks for "COMPLETE program".
# I will output the full file.

# Let's re-verify the "EVOLVE-BLOCK" requirement.
# "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program above; do not rename functions or change what they return."
# The current program has `construct` defined in the block.
# I will define `construct` in the block.

# One detail: The seed program uses `minimize` from scipy. My new code doesn't need it.
# I will remove unused imports if I'm rewriting from scratch, but the prompt says "Keep the program's entry-point...". 
# It doesn't say I must keep unused imports. 
# However, to be safe, I'll include the necessary imports.

# Let's assemble the final code.

import numpy as np
import time
import random

N = 50
TIME_LIMIT = 140.0

def _compute_radius(centers):
    c = np.asarray(centers)
    n = c.shape[0]
    abs_c = np.abs(c)
    max_coord = np.max(abs_c, axis=1)
    min_wall = 0.5 - np.max(max_coord)
    
    if n < 2:
        return min_wall
        
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)
    min_dist2 = np.min(dist2)
    min_dist = np.sqrt(min_dist2) / 2.0
    
    return min(min_wall, min_dist)

def _init_fcc():
    x = np.linspace(-0.45, 0.45, 4)
    y = np.linspace(-0.45, 0.45, 4)
    z = np.linspace(-0.45, 0.45, 4)
    grid = np.array([(x, y, z) for z in z for y in y for x in x])
    idx = np.random.choice(len(grid), 50, replace=False)
    centers = grid[idx]
    centers += np.random.normal(0, 0.01, size=centers.shape)
    return centers

def _init_bcc():
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x0 = -0.5 + i * 0.25
                y0 = -0.5 + j * 0.25
                z0 = -0.5 + k * 0.25
                pts.append([x0, y0, z0])
                pts.append([x0 + 0.125, y0 + 0.125, z0 + 0.125])
    pts = np.array(pts)
    idx = np.random.choice(len(pts), 50, replace=False)
    centers = pts[idx]
    centers += np.random.normal(0, 0.01, size=centers.shape)
    return centers

def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(50, 3))

def _repulsion_anneal(centers, time_limit, steps=2000, t_start=0.1, t_end=0.001):
    c = centers.copy()
    n = len(c)
    iu = np.triu_indices(n, k=1)
    t = t_start
    t_ratio = t_end / t_start
    step_ratio = t_ratio ** (1.0 / steps)
    best_c = c.copy()
    best_score = _compute_radius(c)
    
    for i in range(steps):
        if time.time() > time_limit:
            break
        diff = c[iu[0]] - c[iu[1]]
        dist2 = np.sum(diff * diff, axis=1)
        dist2[dist2 < 1e-10] = 1e-10
        dist = np.sqrt(dist2)
        forces = diff / (dist2 * dist)[:, np.newaxis]
        new_c = c.copy()
        np.add.at(new_c, iu[0], forces)
        np.add.at(new_c, iu[1], -forces)
        noise = np.random.normal(0, t, size=c.shape)
        new_c += noise
        new_c = np.clip(new_c, -0.5 + 0.01, 0.5 - 0.01)
        new_score = _compute_radius(new_c)
        old_score = _compute_radius(c)
        
        if new_score >= old_score:
            c = new_c
            if new_score > best_score:
                best_score = new_score
                best_c = c.copy()
        else:
            delta = new_score - old_score
            if delta < -1e-6:
                if np.random.random() < np.exp(delta / t):
                    c = new_c
            else:
                c = new_c
        t *= step_ratio
    return best_c

def _lloyd_step(c, r):
    n = len(c)
    new_c = c.copy()
    for i in range(n):
        dist2 = np.sum((c - c[i])**2, axis=1)
        dist2[i] = np.inf
        j = np.argmin(dist2)
        d = np.sqrt(dist2[j])
        if d < 2 * r:
            direction = (c[i] - c[j]) / d
            overlap = 2 * r - d
            step = 0.1 * overlap
            new_c[i] += direction * step
    new_c = np.clip(new_c, -0.5 + r, 0.5 - r)
    return new_c

def _lloyd_optimize(c, r, max_iter=100, time_limit=None):
    for i in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
        new_c = _lloyd_step(c, r)
        if np.allclose(c, new_c, atol=1e-7):
            break
        c = new_c
    return c

def _local_search(c, time_limit, steps=100):
    best_c = c.copy()
    best_score = _compute_radius(c)
    for i in range(steps):
        if time.time() > time_limit:
            break
        c_try = best_c + np.random.normal(0, 0.01, size=best_c.shape)
        c_try = np.clip(c_try, -0.5 + 0.01, 0.5 - 0.01)
        score_try = _compute_radius(c_try)
        if score_try > best_score:
            best_c = c_try
            best_score = score_try
    return best_c

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = [
        _init_fcc(),
        _init_bcc(),
        _init_random()
    ]
    
    best_centers = None
    best_score = -1.0
    
    for init in inits:
        if time.time() > deadline - 20:
            break
        annealed = _repulsion_anneal(init, time_limit=deadline - 10)
        r_current = _compute_radius(annealed)
        lloyded = _lloyd_optimize(annealed, r_current, max_iter=50, time_limit=deadline - 5)
        polished = _local_search(lloyded, time_limit=deadline - 2)
        
        score = _compute_radius(polished)
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
            
    if best_centers is not None and time.time() < deadline - 5:
        final = _local_search(best_centers, time_limit=deadline - 1, steps=50)
        final_score = _compute_radius(final)
        if final_score > best_score:
            best_centers = final
    
    return best_centers
# EVOLVE-BLOCK-END

