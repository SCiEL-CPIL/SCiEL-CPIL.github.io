"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill

    # 1. Ring-based initialization matching 1+5+13+31 structure
    def generate_ring_init():
        """Generate initial positions using concentric rings matching known optimal structure."""
        centers = np.zeros((N, 2))
        idx = 0

        # Center circle
        centers[idx] = [0.0, 0.0]
        idx += 1

        # Ring 1: 5 circles
        r1 = 0.25
        for j in range(5):
            angle = 2 * np.pi * j / 5
            centers[idx] = [r1 * np.cos(angle), r1 * np.sin(angle)]
            idx += 1

        # Ring 2: 13 circles
        r2 = 0.50
        for j in range(13):
            angle = 2 * np.pi * j / 13 + np.pi / 13  # offset
            centers[idx] = [r2 * np.cos(angle), r2 * np.sin(angle)]
            idx += 1

        # Ring 3: 31 circles
        r3 = 0.75
        for j in range(31):
            angle = 2 * np.pi * j / 31
            centers[idx] = [r3 * np.cos(angle), r3 * np.sin(angle)]
            idx += 1

        return centers

    current_centers = generate_ring_init()

    # 2. Grid-based collision detection for O(N) neighbor lookups
    class GridMap:
        """Spatial grid for fast neighbor lookups."""
        def __init__(self, cell_size=0.15):
            self.cell_size = cell_size
            self.grid = {}

        def _cell_key(self, x, y):
            return (int(x / self.cell_size), int(y / self.cell_size))

        def _neighbor_cells(self, x, y):
            cx, cy = self._cell_key(x, y)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    yield (cx + dx, cy + dy)

        def build(self, centers):
            self.grid = {}
            for i, (x, y) in enumerate(centers):
                key = self._cell_key(x, y)
                if key not in self.grid:
                    self.grid[key] = []
                self.grid[key].append(i)

        def get_neighbors(self, x, y, exclude_idx):
            """Get indices of points within cell_size of (x,y), excluding exclude_idx."""
            neighbors = []
            for cell in self._neighbor_cells(x, y):
                if cell in self.grid:
                    for idx in self.grid[cell]:
                        if idx != exclude_idx:
                            neighbors.append(idx)
            return neighbors

    # Helper: Calculate radius using grid for faster neighbor lookups
    def calc_min_radius(centers, grid_map=None):
        if grid_map is None:
            grid_map = GridMap(cell_size=0.15)
            grid_map.build(centers)

        min_pair_dist = np.inf
        # Check each point against its grid neighbors
        for i in range(len(centers)):
            x, y = centers[i]
            neighbors = grid_map.get_neighbors(x, y, i)
            for j in neighbors:
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < min_pair_dist:
                    min_pair_dist = d

        # Boundary constraint
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)

        return min(min_pair_dist / 2.0, min_boundary)

    current_r = calc_min_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r

    # 3. Optimization: Simulated Annealing with grid-accelerated evaluation
    T = 0.015
    alpha = 0.9998
    T_min = 1e-5

    grid_map = GridMap(cell_size=0.15)
    grid_map.build(current_centers)

    t_start = time.time()
    max_sa_time = time_limit - 8.0  # Reserve time for polish

    for it in range(10000000):  # Large cap, time will stop it
        if time.time() - t_start > max_sa_time or time.time() - start_time > time_limit - 2.0:
            break

        # Pick a random center
        i = np.random.randint(N)

        # Move size proportional to T
        delta = np.random.normal(0, T, 2)

        old_center = current_centers[i].copy()
        new_center = old_center + delta
        new_centers = current_centers.copy()
        new_centers[i] = new_center

        # Update grid efficiently: only need to check neighbors of moved point
        new_r = calc_min_radius(new_centers, grid_map)

        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
            grid_map.build(current_centers)
        else:
            delta_r = current_r - new_r
            if np.random.rand() < np.exp(-delta_r / T):
                current_centers = new_centers
                current_r = new_r
                grid_map.build(current_centers)

        T *= alpha
        if T < T_min:
            T = T_min

    # 4. Final Polish: Deterministic local optimization
    T = 5e-5
    alpha = 0.999
    t_start_polish = time.time()
    polish_budget = 6.0

    grid_map.build(current_centers)

    for it in range(200000):
        if time.time() - t_start_polish > polish_budget:
            break
        if time.time() - start_time > time_limit - 2.0:
            break

        i = np.random.randint(N)
        delta = np.random.normal(0, T, 2)

        new_centers = current_centers.copy()
        new_centers[i] = current_centers[i] + delta

        new_r = calc_min_radius(new_centers, grid_map)

        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
            grid_map.build(current_centers)
        else:
            delta_r = current_r - new_r
            if np.random.rand() < np.exp(-delta_r / T):
                current_centers = new_centers
                current_r = new_r
                grid_map.build(current_centers)

        T *= alpha
        if T < 1e-6:
            T = 1e-6

    # Final verification with full pairwise check
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.linalg.norm(best_centers, axis=1)
    min_boundary = 1.0 - np.max(norms)
    final_r = min(min_pair, min_boundary)

    if final_r > best_r:
        best_r = final_r

    return best_centers

# EVOLVE-BLOCK-END
