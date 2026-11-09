import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Generate all 6 orientations for a box
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique
    
    # Pre-compute orientations
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
    
    # Sort boxes by volume descending (larger first)
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    # Spatial grid for fast overlap checking
    class SpatialGrid:
        def __init__(self, W, H, D, cell_size=10):
            self.cell_size = cell_size
            self.cells = {}
            self.W = W
            self.H = H
            self.D = D
            
        def _get_cells(self, x, y, z, w, h, d):
            x0 = x // self.cell_size
            y0 = y // self.cell_size
            z0 = z // self.cell_size
            x1 = (x + w - 1) // self.cell_size
            y1 = (y + h - 1) // self.cell_size
            z1 = (z + d - 1) // self.cell_size
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    for cz in range(z0, z1 + 1):
                        yield (cx, cy, cz)
                        
        def add(self, x, y, z, w, h, d, box_id):
            for cell in self._get_cells(x, y, z, w, h, d):
                if cell not in self.cells:
                    self.cells[cell] = []
                self.cells[cell].append(box_id)
                
        def get_candidates(self, x, y, z, w, h, d):
            candidates = set()
            for cell in self._get_cells(x, y, z, w, h, d):
                if cell in self.cells:
                    candidates.update(self.cells[cell])
            return candidates
            
        def is_empty(self, x, y, z, w, h, d):
            candidates = self.get_candidates(x, y, z, w, h, d)
            for box_id in candidates:
                px, py, pz, pw, ph, pd = self.boxes[box_id]
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    return False
            return True
            
        def __init__(self, W, H, D, cell_size=10):
            self.cell_size = cell_size
            self.cells = {}
            self.W = W
            self.H = H
            self.D = D
            self.boxes = {}
            self.next_id = 0
            
        def add(self, x, y, z, w, h, d):
            box_id = self.next_id
            self.next_id += 1
            self.boxes[box_id] = (x, y, z, w, h, d)
            for cell in self._get_cells(x, y, z, w, h, d):
                if cell not in self.cells:
                    self.cells[cell] = []
                self.cells[cell].append(box_id)
            return box_id
            
        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > self.W or y + h > self.H or z + d > self.D:
                return False
            candidates = self.get_candidates(x, y, z, w, h, d)
            for box_id in candidates:
                px, py, pz, pw, ph, pd = self.boxes[box_id]
                if (x < px + pw and x + w > px and
                    y < py + ph and y + h > py and
                    z < pz + pd and z + d > pz):
                    return False
            return True
    
    def find_position_fast(grid, bw, bh, bd):
        """Find the best position for a box using dense candidate positions."""
        candidates = [(0, 0, 0)]
        
        # Add positions from all placed boxes
        for box_id, (px, py, pz, pw, ph, pd) in grid.boxes.items():
            # 8 corners
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
            
            # Additional positions along edges
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz))
            
        # Remove duplicates
        candidates = list(set(candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if not grid.can_place(cx, cy, cz, bw, bh, bd):
                continue
            
            # Score: prefer lower z, then lower y, then lower x
            score = cz * 100 + cy * 10 + cx
            
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    def run_pack(order, grid=None):
        """Try to pack boxes in the given order."""
        if grid is None:
            grid = SpatialGrid(W, H, D)
        else:
            grid.boxes = {}
            grid.cells = {}
            grid.next_id = 0
        
        results = [None] * n
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position_fast(grid, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                grid.add(cx, cy, cz, ow, oh, od)
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results, grid
    
    def reinsert_leftout(results, grid):
        """Try to re-insert left-out boxes into gaps."""
        leftout = [i for i in range(n) if results[i][0] == -1]
        if not leftout:
            return results, grid
        
        # Sort left-out by volume descending
        leftout.sort(key=lambda i: box_volumes[i], reverse=True)
        
        for idx in leftout:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position_fast(grid, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                grid.add(cx, cy, cz, ow, oh, od)
                results[idx] = (0, cx, cy, cz, ow, oh, od)
        
        return results, grid
    
    # Try multiple packing orders
    best_results = None
    best_volume = -1
    best_grid = None
    
    # Order 1: Original order
    order1 = list(range(n))
    results1, grid1 = run_pack(order1)
    vol1 = sum(w*h*d for (b, x, y, z, w, h, d) in results1 if b == 0)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = results1
        best_grid = grid1
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 2: Sorted by volume descending
    order2 = box_indices
    results2, grid2 = run_pack(order2)
    vol2 = sum(w*h*d for (b, x, y, z, w, h, d) in results2 if b == 0)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = results2
        best_grid = grid2
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 3: Sorted by max dimension descending
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    order3 = sorted(range(n), key=lambda i: box_max_dim[i], reverse=True)
    results3, grid3 = run_pack(order3)
    vol3 = sum(w*h*d for (b, x, y, z, w, h, d) in results3 if b == 0)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = results3
        best_grid = grid3
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 4: Sorted by aspect ratio (more cubic first)
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else float('inf') for w, h, d in boxes]
    order4 = sorted(range(n), key=lambda i: box_aspect[i])
    results4, grid4 = run_pack(order4)
    vol4 = sum(w*h*d for (b, x, y, z, w, h, d) in results4 if b == 0)
    if vol4 > best_volume:
        best_volume = vol4
        best_results = results4
        best_grid = grid4
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try many random orders if time permits
    random.seed(42)
    for trial in range(300):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        results, grid = run_pack(order)
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results
            best_grid = grid
    
    # Try local improvement: perturbed orders based on best
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        base_orders = [box_indices, order3, order4]
        for base_order in base_orders:
            for trial in range(80):
                if time.time() - start_time > DEADLINE - 5:
                    break
                order = list(base_order)
                num_swaps = random.randint(1, 8)
                for _ in range(num_swaps):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    order[i], order[j] = order[j], order[i]
                
                results, grid = run_pack(order)
                vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
                if vol > best_volume:
                    best_volume = vol
                    best_results = results
                    best_grid = grid
    
    # Reinsert left-out boxes
    if best_results is not None and best_grid is not None and time.time() - start_time < DEADLINE - 5:
        best_results, _ = reinsert_leftout(best_results, best_grid)
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
