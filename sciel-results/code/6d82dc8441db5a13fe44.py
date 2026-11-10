import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 20.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    # Build lines
    # Line ID: m*49 + c for non-vertical (2401), 2401 + x0 for vertical (49)
    point_lines = [[] for _ in range(q2)]
    line_points = [[] for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].append(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].append(pi)
    
    def compute_witness(in_C):
        """Compute witness sets and counts for all points."""
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                valid = True
                for q_idx in line_points[lid]:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        return W, wcount
    
    def is_valid(wcount):
        return np.all(wcount > 0)
    
    def try_greedy_expand(in_C, W, wcount, rng, max_steps=100000):
        """Greedy expansion using priority queue with lazy evaluation."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        if not is_valid(wcount):
            return in_C, W, wcount, 0
        
        # Build initial heap
        heap = []
        for i in range(q2):
            if not in_C[i]:
                # Cost: number of points that would lose a witness
                cost = 0
                for lid in point_lines[i]:
                    for x_idx in line_points[lid]:
                        if x_idx != i and lid in W[x_idx]:
                            if wcount[x_idx] == 1:
                                cost += 1
                heapq.heappush(heap, (cost, i))
        
        added = 0
        while heap:
            if time.time() > _DEADLINE - _MARGIN:
                break
            
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
            
            # Verify cost is still accurate
            current_cost = 0
            for lid in point_lines[p_idx]:
                for x_idx in line_points[lid]:
                    if x_idx != p_idx and lid in W[x_idx]:
                        if wcount[x_idx] == 1:
                            current_cost += 1
            
            if current_cost != cost:
                heapq.heappush(heap, (current_cost, p_idx))
                continue
            
            if current_cost > 0:
                continue
            
            # Safe to add
            in_C[p_idx] = True
            added += 1
            
            # Update witnesses for affected points
            for lid in point_lines[p_idx]:
                for x_idx in line_points[lid]:
                    if x_idx != p_idx and lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
                        # If this point now has 0 witnesses, we have a problem
                        # But we checked cost==0, so this shouldn't happen
                        if wcount[x_idx] == 0:
                            # This is invalid - rollback not implemented, skip
                            pass
            
            # Push neighbors back into heap with updated costs
            for lid in point_lines[p_idx]:
                for x_idx in line_points[lid]:
                    if x_idx != p_idx and not in_C[x_idx]:
                        new_cost = 0
                        for lid2 in point_lines[x_idx]:
                            for y_idx in line_points[lid2]:
                                if y_idx != x_idx and lid2 in W[y_idx]:
                                    if wcount[y_idx] == 1:
                                        new_cost += 1
                        heapq.heappush(heap, (new_cost, x_idx))
        
        return in_C, W, wcount, added
    
    def local_search_swap(in_C, W, wcount, rng, max_iters=5000):
        """Try to improve by swapping: remove one C-point, add another."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        improved = True
        while improved and time.time() < _DEADLINE - _MARGIN:
            improved = False
            
            # Try random swaps
            for _ in range(100):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                
                # Pick a random point to remove from C
                c_points = [i for i in range(q2) if in_C[i]]
                if not c_points:
                    break
                
                rem = rng.choice(c_points)
                
                # Tentatively remove
                in_C[rem] = False
                
                # Update W: removing from C can only ADD witnesses
                # Recompute affected witnesses
                for lid in point_lines[rem]:
                    for x_idx in line_points[lid]:
                        # Check if lid is now a valid witness for x_idx
                        valid = True
                        for q_idx in line_points[lid]:
                            if q_idx != x_idx and in_C[q_idx]:
                                valid = False
                                break
                        if valid and lid not in W[x_idx]:
                            W[x_idx].add(lid)
                            wcount[x_idx] += 1
                
                # Now try to add a point that was previously blocked
                # Find points with low cost
                best_add = -1
                best_cost = float('inf')
                
                for add in range(q2):
                    if in_C[add]:
                        continue
                    cost = 0
                    for lid in point_lines[add]:
                        for x_idx in line_points[lid]:
                            if x_idx != add and lid in W[x_idx]:
                                if wcount[x_idx] == 1:
                                    cost += 1
                                    if cost >= best_cost:
                                        break
                        if cost >= best_cost:
                            break
                    if cost < best_cost:
                        best_cost = cost
                        best_add = add
                
                if best_add == -1 or best_cost > 0:
                    # Rollback
                    in_C[rem] = True
                    for lid in point_lines[rem]:
                        for x_idx in line_points[lid]:
                            # Remove witnesses that were added
                            valid = True
                            for q_idx in line_points[lid]:
                                if q_idx != x_idx and in_C[q_idx]:
                                    valid = False
                                    break
                            if not valid and lid in W[x_idx]:
                                W[x_idx].discard(lid)
                                wcount[x_idx] -= 1
                    continue
                
                # Perform the swap
                in_C[best_add] = True
                for lid in point_lines[best_add]:
                    for x_idx in line_points[lid]:
                        if x_idx != best_add and lid in W[x_idx]:
                            W[x_idx].discard(lid)
                            wcount[x_idx] -= 1
                
                improved = True
                break
        
        return in_C, W, wcount
    
    def init_parabola():
        """Initialize C with the parabola y = x^2 in F_49."""
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                x_idx = xa * p + xb
                # y = x^2
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C
    
    def init_random(density, rng):
        """Initialize C with random points."""
        in_C = np.zeros(q2, dtype=bool)
        n_points = int(q2 * density)
        indices = rng.sample(range(q2), n_points)
        for i in indices:
            in_C[i] = True
        return in_C
    
    best_in_C = None
    best_size = -1
    
    rng = random.Random(42)
    
    # Try multiple initializations
    initializations = []
    
    # Parabola
    initializations.append(init_parabola())
    
    # Random with various densities
    for density in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        initializations.append(init_random(density, rng))
    
    # More parabola variants with random perturbations
    for _ in range(5):
        base = init_parabola()
        # Add some random points
        extra = rng.sample(range(q2), 20)
        for i in extra:
            base[i] = True
        initializations.append(base)
    
    for init_C in initializations:
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        W, wcount = compute_witness(init_C)
        
        if not is_valid(wcount):
            # Try to repair by removing points
            valid = False
            for _ in range(200):
                if time.time() > _DEADLINE - _MARGIN:
                    break
                bad = np.where(wcount == 0)[0]
                if len(bad) == 0:
                    valid = True
                    break
                # Remove a random C-point
                c_pts = np.where(init_C)[0]
                if len(c_pts) == 0:
                    break
                rem = rng.choice(c_pts)
                init_C[rem] = False
                W, wcount = compute_witness(init_C)
            
            if not valid:
                continue
        
        # Greedy expand
        in_C, W, wcount, added = try_greedy_expand(init_C, W, wcount, rng)
        
        # Local search
        in_C, W, wcount = local_search_swap(in_C, W, wcount, rng)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
    
    if best_in_C is None:
        # Fallback: empty C
        best_in_C = np.zeros(q2, dtype=bool)
    
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
