"""3D Container Loading - Multi-start Best-Fit with Skyline Anchors and ILS."""
import time
import random
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    t0 = time.time()
    DEADLINE = 145  # seconds, leave margin for safety (limit is 150s)
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute orientations for each box
    # boxes[i] is (w, h, d)
    box_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = list(set(permutations([w, h, d])))
        box_orients.append(orients)
    
    def get_volume(b):
        return b[0] * b[1] * b[2]
    
    def check_overlap(x, y, z, ow, oh, od, placed):
        """Check if a box at (x,y,z) with dims (ow,oh,od) overlaps with any placed box."""
        for px, py, pz, pw, ph, pd in placed:
            # Overlap if: x < px+pw AND px < x+ow
            #             y < py+ph AND py < y+oh
            #             z < pz+pd AND pz < z+od
            if (x < px + pw and px < x + ow and
                y < py + ph and py < y + oh and
                z < pz + pd and pz < z + od):
                return True
        return False

    def pack(order):
        """
        Greedy pack given a specific order of box indices.
        Returns (placements, total_volume_placed)
        """
        placed = []
        candidates = [(0, 0, 0)]
        placements = [None] * n
        total_vol = 0
        
        for idx in order:
            if time.time() - t0 > DEADLINE:
                # Emergency exit: mark rest as -1 and return what we have
                break
                
            w, h, d = boxes[idx]
            orients = box_orients[idx]
            
            best_pos = None
            best_score = float('inf')
            
            # We iterate orientations and candidates
            # Score: minimize (z, y, x) to fill bottom-up, then front-to-back
            # This heuristic tends to create a "wall" of boxes, reducing fragmentation
            
            for ow, oh, od in orients:
                if ow > W or oh > H or od > D:
                    continue
                    
                for cx, cy, cz in candidates:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                        
                    if check_overlap(cx, cy, cz, ow, oh, od, placed):
                        continue
                        
                    score = (cz, cy, cx)
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz, ow, oh, od)
            
            if best_pos is not None:
                bx, by, bz, bw, bh, bd = best_pos
                placements[idx] = (0, bx, by, bz, bw, bh, bd)
                placed.append((bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd
                
                # Generate new candidates: corners of the placed box
                # (bx, by, bz) is the corner, so the new potential floor/corner positions are:
                # (bx+bw, by, bz) -> right
                # (bx, by+bh, bz) -> back
                # (bx, by, bz+bd) -> top
                # and combinations
                new_candidates = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                
                # Add unique candidates
                # Using a set is O(N), list check is O(N^2) worst case but N is small-ish
                # However, candidates can grow. Let's use a set for candidate management internally
                # but keep a list for iteration stability?
                # Actually, order of candidates doesn't matter for correctness, only for which one is "first" found if we broke early.
                # Since we check ALL candidates to find the min score, order doesn't matter.
                # So we can just use a set for membership and a list for iteration.
                
                for nc in new_candidates:
                    if nc not in candidates:
                        candidates.append(nc)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill in any remaining None slots (in case of timeout)
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        return placements, total_vol

    # Strategy:
    # 1. Start with a few different heuristics for ordering.
    # 2. Use Iterated Local Search (ILS) to perturb the best ordering found.
    
    # Base orderings
    # 1. Volume Descending
    order_vol = list(range(n))
    order_vol.sort(key=lambda i: -get_volume(boxes[i]))
    
    # 2. Max Dimension Descending (often good for fitting)
    order_max_dim = list(range(n))
    order_max_dim.sort(key=lambda i: -max(boxes[i]))
    
    # 3. Sum of Dimensions Descending
    order_sum_dim = list(range(n))
    order_sum_dim.sort(key=lambda i: -(boxes[i][0] + boxes[i][1] + boxes[i][2]))
    
    # 4. Random
    order_rand = list(range(n))
    random.shuffle(order_rand)
    
    initial_orders = [order_vol, order_max_dim, order_sum_dim, order_rand]
    
    best_placements = None
    best_volume = -1
    
    # Evaluate initial orders
    for order in initial_orders:
        if time.time() - t0 > DEADLINE:
            break
        placements, vol = pack(order)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            best_order = order[:] # Store the order that produced the best result
    
    # ILS Loop
    # If we have time, try to improve the best_order by swapping pairs of boxes
    
    if time.time() - t0 < DEADLINE - 5: # Need at least 5s to finish safely
        current_order = best_order[:]
        
        # Simple ILS: Swap two random indices in the order, re-pack.
        # Keep the best solution found.
        
        while time.time() - t0 < DEADLINE - 2:
            # Perturbation: Swap two random elements
            i, j = random.sample(range(n), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            placements, vol = pack(current_order)
            
            if vol > best_volume:
                best_volume = vol
                best_placements = placements
                best_order = current_order[:]
            else:
                # If we didn't improve, revert the swap?
                # Or keep exploring? Standard ILS often accepts worse solutions to escape local minima,
                # but since our "state" is just the order, and we track the global best,
                # we can just keep mutating from the current state or reset to best.
                # Let's try a mix: mostly mutate from best, sometimes from current.
                # For simplicity and stability, let's mutate from the best_order found so far.
                # This is a "hill climbing" on the order space.
                current_order = best_order[:]
            
    # Finalize
    if best_placements is None:
        # Fallback: just try volume order one last time if something went wrong
        placements, _ = pack(order_vol)
        best_placements = placements

    return best_placements
