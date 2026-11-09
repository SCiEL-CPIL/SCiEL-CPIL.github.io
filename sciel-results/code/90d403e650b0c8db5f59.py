"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 150s limit (using 115s)
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
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
    
    def run_pack(order, placement_strategy):
        """
        Attempt to pack boxes in the given order.
        placement_strategy: 'skyline' or 'random'
        """
        # Use integer grid for fast collision checking if dimensions allow
        # For general floats, we'd need a different structure, but assuming typical integer-ish or manageable floats
        # To be safe and fast, we use a list of placed boxes and do O(N) check, 
        # but we optimize by only checking against boxes that could potentially overlap.
        # Actually, for 60 boxes, O(N^2) is 3600 checks, which is trivial. 
        # The bottleneck in the previous code was generating candidates.
        # Here, we use a deterministic "best fit" heuristic which is very fast.
        
        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                if placement_strategy == 'skyline':
                    # Skyline heuristic: Try to place at the lowest possible z, then lowest y, then lowest x
                    # We scan the space in a structured way to find the first valid spot.
                    # Since N is small, we can just check a set of "promising" spots.
                    # Promising spots: (0,0,0) and spots adjacent to existing boxes.
                    candidates = [(0, 0, 0)]
                    for (px, py, pz, pw, ph, pd) in placed:
                        candidates.append((px + pw, py, pz))
                        candidates.append((px, py + ph, pz))
                        candidates.append((px, py, pz + pd))
                        candidates.append((px + pw, py + ph, pz))
                        candidates.append((px + pw, py, pz + pd))
                        candidates.append((px, py + ph, pz + pd))
                        candidates.append((px + pw, py + ph, pz + pd))
                    
                    # Sort candidates by (z, y, x) to prioritize bottom-front-left
                    candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                    
                    for (cx, cy, cz) in candidates:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        if cx < 0 or cy < 0 or cz < 0:
                            continue
                        
                        # Check overlap
                        overlap = False
                        for (px, py, pz, pw, ph, pd) in placed:
                            if (cx < px + pw and cx + ow > px and
                                cy < py + ph and cy + oh > py and
                                cz < pz + pd and cz + od > pz):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Valid position found
                            # Score: prefer lower z, then lower y, then lower x
                            score = cz * 10000 + cy * 100 + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orientation = (ow, oh, od)
                            break # Found the best for this orientation (lowest z,y,x)
                else:
                    # Random strategy: Try a few random positions
                    # This is less likely to be the primary driver, but adds diversity
                    pass

            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results

    # Try multiple packing orders
    best_results = None
    best_volume = -1
    
    # Generate diverse orders
    orders = []
    
    # 1. Volume descending
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    orders.append(sorted(range(n), key=lambda i: box_volumes[i], reverse=True))
    
    # 2. Max dimension descending
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: box_max_dim[i], reverse=True))
    
    # 3. Min dimension descending (thick boxes first)
    box_min_dim = [min(w, h, d) for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: box_min_dim[i], reverse=True))
    
    # 4. Aspect ratio (cubic first)
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else float('inf') for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: box_aspect[i]))
    
    # 5. Surface area descending
    box_surface = [2*(w*h + w*d + h*d) for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: box_surface[i], reverse=True))
    
    # Add random orders
    random.seed(42)
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        orders.append(order)
    
    # Run packers
    for order in orders:
        if time.time() - start_time > DEADLINE:
            break
        
        results = run_pack(order, 'skyline')
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    # If we have time left, do more random restarts
    while time.time() - start_time < DEADLINE - 1:
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order, 'skyline')
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
